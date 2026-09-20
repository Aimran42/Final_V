"""Small, schema-aware constrained decoder."""

import json
from typing import Any, Callable

try:
    import numpy as np
    from llm_sdk.llm_sdk import Small_LLM_Model
except ImportError as err:
    print(f"Import Error:{err}")

from src.models import FunctionDefinition


class ConstrainedDecodingError(RuntimeError):
    """Raised when the model cannot continue within the required grammar."""


class TokenTrie:
    """A prefix tree used to limit output to a known list of names."""

    def __init__(self, words: list[str]) -> None:
        self.root: dict[str, Any] = {}
        for word in words:
            node = self.root
            for character in word:
                node = node.setdefault(character, {})
            node["end"] = True

    def accepts_prefix(self, text: str) -> bool:
        node = self.root
        for character in text:
            if character not in node:
                return False
            node = node[character]
        return True

    def accepts_word(self, text: str) -> bool:
        node = self.root
        for character in text:
            if character not in node:
                return False
            node = node[character]
        return bool(node.get("end"))


class ConstrainedDecoder:
    """Generate a valid function name and a valid typed JSON object."""

    def __init__(self, model: Small_LLM_Model) -> None:
        self.model = model
        self.token_text: dict[int, str] = {}

    def _decode_token(self, token_id: int) -> str:
        """Decode a token once and cache the result."""
        if token_id not in self.token_text:
            self.token_text[token_id] = self.model.decode([token_id])
        return self.token_text[token_id]

    def _next_token(
        self,
        input_ids: list[int],
        generated: str,
        accepts_prefix: Callable[[str], bool],
    ) -> tuple[int, str]:
        """Return the highest-scoring token which preserves the grammar."""
        logits = np.asarray(self.model.get_logits_from_input_ids(input_ids))
        for token_id in np.argsort(logits)[::-1]:
            token = self._decode_token(int(token_id))
            if token and accepts_prefix(generated + token):
                return int(token_id), token
        raise ConstrainedDecodingError("no token satisfies the constraint")

    @staticmethod
    def _function_prompt(
        prompt: str, functions: list[FunctionDefinition]
    ) -> str:
        definitions = json.dumps(
            [function.model_dump() for function in functions],
            separators=(",", ":"),
        )
        return (
            "Choose the best function for the user request. "
            "Return only its exact name.\n"
            f"Functions: {definitions}\nUser request: {prompt}\nFunction name:"
        )

    def select_function(
        self, prompt: str, functions: list[FunctionDefinition]
    ) -> FunctionDefinition:
        if not functions:
            raise ConstrainedDecodingError(
                "no function definitions were supplied")

        names = TokenTrie([function.name for function in functions])
        input_ids = self.model.encode(
            self._function_prompt(prompt, functions)).tolist()[0]
        generated = ""
        for _ in range(64):
            token_id, token = self._next_token(
                input_ids, generated, names.accepts_prefix
            )
            generated += token
            input_ids.append(token_id)
            if names.accepts_word(generated):
                return next(
                    function for function in functions
                    if function.name == generated
                )
        raise ConstrainedDecodingError("could not complete a function name")

    @staticmethod
    def _literal_prefix(
        text: str, position: int, literal: str
    ) -> tuple[bool, int]:
        """Match all or part of a fixed grammar literal."""
        fragment = text[position:]
        if literal.startswith(fragment):
            if fragment != literal:
                return True, len(text)
            return True, position + len(literal)
        if fragment.startswith(literal):
            return True, position + len(literal)
        return False, position

    @staticmethod
    def _string_prefix(text: str, position: int) -> tuple[bool, bool, int]:
        if position == len(text):
            return True, False, position
        if text[position] != '"':
            return False, False, position
        escaped = False
        for index in range(position + 1, len(text)):
            character = text[index]
            if ord(character) < 0x20:
                return False, False, index
            if not escaped and character == '"':
                return True, True, index + 1
            if escaped and character not in '"\\/bfnrtu':
                return False, False, index
            escaped = not escaped if character == "\\" else False
        return True, False, len(text)

    @classmethod
    def _number_prefix(cls, value: str, integer: bool) -> tuple[bool, bool]:
        """Return whether ``value`` can become a JSON number and is complete.

        This deliberately uses a small state machine instead of a regular
        expression so every accepted character is easy to see.
        """
        index = 0
        if value.startswith("-"):
            index = 1
        if index == len(value):
            return True, False
        if not value[index].isdigit():
            return False, False

        if value[index] == "0":
            index += 1
            if index < len(value) and value[index].isdigit():
                return False, False
        else:
            while index < len(value) and value[index].isdigit():
                index += 1

        if integer:
            return index == len(value), index == len(value)
        if index == len(value):
            return True, True
        if value[index] == ".":
            index += 1
            decimal_start = index
            while index < len(value) and value[index].isdigit():
                index += 1
            if index == len(value):
                return True, index > decimal_start
            if index == decimal_start:
                return False, False
        if value[index] not in "eE":
            return False, False
        index += 1
        if index < len(value) and value[index] in "+-":
            index += 1
        exponent_start = index
        while index < len(value) and value[index].isdigit():
            index += 1
        return index == len(value), index > exponent_start

    @classmethod
    def _value_prefix(
        cls, text: str, position: int, value_type: str
    ) -> tuple[bool, bool, int]:
        if value_type == "string":
            return cls._string_prefix(text, position)

        end = position
        while end < len(text) and text[end] not in ",}":
            end += 1
        value = text[position:end]
        if value_type == "boolean":
            valid = value in {"", "t", "tr", "tru",
                              "true", "f", "fa", "fal", "fals", "false"}
            complete = value in {"true", "false"}
        else:
            valid, complete = cls._number_prefix(
                value, value_type == "integer"
            )
        return valid and (end == len(text) or complete), complete, end

    @classmethod
    def _arguments_prefix_valid(
        cls, text: str, function: FunctionDefinition
    ) -> bool:
        """Check a partial object against its parameter order and types."""
        position = 0
        for index, (key, parameter) in enumerate(function.parameters.items()):
            prefix = ("{" if index == 0 else ",") + json.dumps(key) + ":"
            valid, position = cls._literal_prefix(text, position, prefix)
            if not valid:
                return False
            if position == len(text):
                return True
            valid, complete, position = cls._value_prefix(
                text, position, parameter.type)
            if not valid:
                return False
            if not complete:
                return True

        valid, position = cls._literal_prefix(text, position, "}")
        return valid and position == len(text)

    @classmethod
    def _arguments_complete(
        cls, text: str, function: FunctionDefinition
    ) -> bool:
        if not cls._arguments_prefix_valid(text, function):
            return False
        try:
            return isinstance(json.loads(text), dict)
        except json.JSONDecodeError:
            return False

    @staticmethod
    def _arguments_prompt(prompt: str, function: FunctionDefinition) -> str:
        definition = json.dumps(function.model_dump(), separators=(",", ":"))
        return (
            "Extract the arguments for this function. "
            "Return only one JSON object.\n"
            f"Function: {definition}\nUser request: {prompt}\nArguments:"
        )

    @staticmethod
    def _validate_arguments(
        arguments: dict[str, Any], function: FunctionDefinition
    ) -> dict[str, Any]:
        if set(arguments) != set(function.parameters):
            raise ConstrainedDecodingError(
                "generated JSON has invalid parameter keys")
        validated: dict[str, Any] = {}
        for name, parameter in function.parameters.items():
            value = arguments[name]
            if parameter.type == "string" and not isinstance(value, str):
                raise ConstrainedDecodingError(
                    f"parameter '{name}' must be a string")
            if parameter.type == "boolean" and not isinstance(value, bool):
                raise ConstrainedDecodingError(
                    f"parameter '{name}' must be a boolean")
            if parameter.type == "integer" and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise ConstrainedDecodingError(
                    f"parameter '{name}' must be an integer")
            if parameter.type in {"float", "number"} and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise ConstrainedDecodingError(
                    f"parameter '{name}' must be numeric")
            validated[name] = float(value) if parameter.type in {
                "float", "number"} else value
        return validated

    def decode_arguments(
        self, prompt: str, function: FunctionDefinition
    ) -> dict[str, Any]:
        if not function.parameters:
            return {}
        input_ids = self.model.encode(
            self._arguments_prompt(prompt, function)).tolist()[0]
        generated = ""
        for _ in range(256):
            token_id, token = self._next_token(
                input_ids,
                generated,
                lambda candidate: self._arguments_prefix_valid(
                    candidate, function),
            )
            generated += token
            input_ids.append(token_id)
            if self._arguments_complete(generated, function):
                parsed = json.loads(generated)
                return self._validate_arguments(parsed, function)
        raise ConstrainedDecodingError(
            "could not complete valid function arguments")
