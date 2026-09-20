"""Token-level constrained decoding for function names and arguments."""

import json
import re
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from llm_sdk.llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition


class ConstrainedDecodingError(RuntimeError):
    """Raised when no token can satisfy a decoding constraint."""


class TokenTrie:
    """Prefix trie over a finite set of candidate strings."""

    def __init__(self, words: List[str]) -> None:
        """Build a trie containing ``words``."""
        self.root: Dict[str, Any] = {}
        for word in words:
            node = self.root
            for character in word:
                node = node.setdefault(character, {})
            node["$"] = True

    def is_prefix(self, text: str) -> bool:
        """Return whether ``text`` is a prefix in this trie."""
        node = self.root
        for character in text:
            if character not in node:
                return False
            node = node[character]
        return True

    def is_complete(self, text: str) -> bool:
        """Return whether ``text`` is a complete word in this trie."""
        node = self.root
        for character in text:
            if character not in node:
                return False
            node = node[character]
        return "$" in node


class ConstrainedDecoder:
    """Decodes LLM logits with strict schema constraints."""

    def __init__(self, model: Small_LLM_Model) -> None:
        self.model = model
        self.vocab = self._load_vocab()
        self.token_text_cache: Dict[int, str] = {}

    def _load_vocab(self) -> Dict[int, str]:
        with open(
            self.model.get_path_to_vocab_file(), "r", encoding="utf-8"
        ) as vocab_file:
            vocab: Dict[str, int] = json.load(vocab_file)
        return {token_id: token for token, token_id in vocab.items()}

    def _token_text(self, token_id: int) -> str:
        if token_id not in self.token_text_cache:
            if token_id not in self.vocab:
                return ""
            self.token_text_cache[token_id] = self.model.decode([token_id])
        return self.token_text_cache[token_id]

    def _next_token(
        self,
        input_ids: List[int],
        generated: str,
        is_valid_prefix: Callable[[str], bool],
    ) -> Tuple[int, str]:
        logits = np.asarray(
            self.model.get_logits_from_input_ids(input_ids), dtype=float
        )
        masked_logits = logits.copy()
        for token_id in np.argsort(logits)[::-1]:
            token_id_int = int(token_id)
            token_text = self._token_text(token_id_int)
            if token_text and is_valid_prefix(generated + token_text):
                masked_logits[token_id_int] = logits[token_id_int]
                return token_id_int, token_text
            masked_logits[token_id_int] = -np.inf
        raise ConstrainedDecodingError(
            "no vocabulary token satisfies the schema"
        )

    @staticmethod
    def _function_prompt(
        prompt: str, functions: List[FunctionDefinition]
    ) -> str:
        definitions = json.dumps(
            [function.model_dump() for function in functions],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return (
            "Choose the best function for the user request. "
            "Return only its exact name.\n"
            f"Functions: {definitions}\nUser request: {prompt}\n"
            "Function name:"
        )

    def select_function(
        self, prompt: str, functions: List[FunctionDefinition]
    ) -> FunctionDefinition:
        if not functions:
            raise ConstrainedDecodingError(
                "no function definitions were supplied"
            )
        trie = TokenTrie([function.name for function in functions])
        input_ids = self.model.encode(
            self._function_prompt(prompt, functions)
        ).tolist()[0]
        generated = ""
        for _ in range(64):
            token_id, token_text = self._next_token(
                input_ids, generated, trie.is_prefix
            )
            generated += token_text
            input_ids.append(token_id)
            if trie.is_complete(generated):
                for function in functions:
                    if function.name == generated:
                        return function
        raise ConstrainedDecodingError(
            "could not complete a valid function name"
        )

    @staticmethod
    def _match_expected_syntax(
        text: str, start: int, literal: str
    ) -> Tuple[bool, bool, int]:
        remaining = text[start:]
        if literal.startswith(remaining):
            return True, len(remaining) == len(literal), len(text)
        if remaining.startswith(literal):
            return True, True, start + len(literal)
        return False, False, start

    @classmethod
    def _parse_json_string_prefix(
        cls, text: str, start: int
    ) -> Tuple[bool, bool, int]:
        if start == len(text):
            return True, False, start
        if text[start] != '"':
            return False, False, start
        index = start + 1
        while index < len(text):
            character = text[index]
            if character == '"':
                return True, True, index + 1
            if ord(character) < 0x20:
                return False, False, index
            if character == "\\":
                index += 1
                if index == len(text):
                    return True, False, index
                escaped = text[index]
                if escaped == "u":
                    unicode_end = index + 5
                    fragment = text[index + 1:unicode_end]
                    if not all(
                        char in "0123456789abcdefABCDEF" for char in fragment
                    ):
                        return False, False, index
                    if unicode_end > len(text):
                        return True, False, len(text)
                    index = unicode_end
                    continue
                if escaped not in '"\\/bfnrt':
                    return False, False, index
            index += 1
        return True, False, index

    _INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)$")
    _NUMBER = re.compile(
        r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$"
    )

    @classmethod
    def _parse_json_number_prefix(
        cls, text: str, start: int, integer: bool
    ) -> Tuple[bool, bool, int]:
        index = start
        while index < len(text) and text[index] not in ",}":
            if text[index] not in "-+0123456789.eE":
                return False, False, index
            index += 1
        fragment = text[start:index]
        expression = cls._INTEGER if integer else cls._NUMBER
        complete = bool(expression.fullmatch(fragment))
        if index < len(text):
            return complete, complete, index
        if complete:
            return True, True, index
        allowed_prefix = re.fullmatch(
            r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]*)?(?:[eE][+-]?[0-9]*)?",
            fragment,
        )
        if fragment in {"", "-", "0.", "-0."} or allowed_prefix:
            return True, False, index
        return False, False, index

    @staticmethod
    def _parse_json_boolean_prefix(
        text: str, start: int
    ) -> Tuple[bool, bool, int]:
        remaining = text[start:]
        for literal in ("true", "false"):
            if literal.startswith(remaining):
                return True, remaining == literal, len(text)
            if remaining.startswith(literal):
                return True, True, start + len(literal)
        return False, False, start

    @classmethod
    def _parse_typed_value_prefix(
        cls, text: str, start: int, value_type: str
    ) -> Tuple[bool, bool, int]:
        if value_type == "string":
            return cls._parse_json_string_prefix(text, start)
        if value_type == "boolean":
            return cls._parse_json_boolean_prefix(text, start)
        return cls._parse_json_number_prefix(
            text, start, value_type == "integer"
        )

    @classmethod
    def _arguments_prefix_valid(
        cls, text: str, function: FunctionDefinition
    ) -> bool:
        position = 0
        for index, (key, parameter) in enumerate(function.parameters.items()):
            separator = "{" if index == 0 else ","
            key_literal = json.dumps(key, ensure_ascii=False)
            valid, complete, position = cls._match_expected_syntax(
                text, position, f"{separator}{key_literal}:"
            )
            if not valid:
                return False
            if not complete:
                return True
            valid, complete, position = cls._parse_typed_value_prefix(
                text, position, parameter.type
            )
            if not valid:
                return False
            if not complete:
                return True
        valid, complete, position = cls._match_expected_syntax(
            text, position, "}"
        )
        return valid and (not complete or position == len(text))

    @classmethod
    def _arguments_complete(
        cls, text: str, function: FunctionDefinition
    ) -> bool:
        position = 0
        for index, (key, parameter) in enumerate(function.parameters.items()):
            separator = "{" if index == 0 else ","
            key_literal = json.dumps(key, ensure_ascii=False)
            valid, complete, position = cls._match_expected_syntax(
                text, position, f"{separator}{key_literal}:"
            )
            if not valid or not complete:
                return False
            valid, complete, position = cls._parse_typed_value_prefix(
                text, position, parameter.type
            )
            if not valid or not complete:
                return False
        valid, complete, position = cls._match_expected_syntax(
            text, position, "}"
        )
        return valid and complete and position == len(text)

    @staticmethod
    def _arguments_prompt(prompt: str, function: FunctionDefinition) -> str:
        definition = json.dumps(
            function.model_dump(), ensure_ascii=False, separators=(",", ":")
        )
        return (
            "Extract the arguments for this function. "
            "Return only one JSON object.\n"
            f"Function: {definition}\nUser request: {prompt}\nArguments:"
        )

    @staticmethod
    def _validate_arguments(
        arguments: Dict[str, Any], function: FunctionDefinition
    ) -> Dict[str, Any]:
        if set(arguments) != set(function.parameters):
            raise ConstrainedDecodingError(
                "generated JSON has invalid parameter keys"
            )
        validated: Dict[str, Any] = {}
        for key, parameter in function.parameters.items():
            value = arguments[key]
            if parameter.type == "string" and not isinstance(value, str):
                raise ConstrainedDecodingError(
                    f"parameter '{key}' must be a string"
                )
            if parameter.type == "boolean" and not isinstance(value, bool):
                raise ConstrainedDecodingError(
                    f"parameter '{key}' must be a boolean"
                )
            if parameter.type == "integer" and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise ConstrainedDecodingError(
                    f"parameter '{key}' must be an integer"
                )
            if parameter.type in {"number", "float"} and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise ConstrainedDecodingError(
                    f"parameter '{key}' must be numeric"
                )
            if parameter.type in {"number", "float"}:
                validated[key] = float(value)
            else:
                validated[key] = value
        return validated

    def decode_arguments(
        self, prompt: str, function: FunctionDefinition
    ) -> Dict[str, Any]:
        if not function.parameters:
            return {}
        input_ids = self.model.encode(
            self._arguments_prompt(prompt, function)
        ).tolist()[0]
        generated = ""
        for _ in range(256):
            token_id, token_text = self._next_token(
                input_ids,
                generated,
                lambda candidate: self._arguments_prefix_valid(
                    candidate, function
                ),
            )
            generated += token_text
            input_ids.append(token_id)
            if self._arguments_complete(generated, function):
                parsed = json.loads(generated)
                if not isinstance(parsed, dict):
                    raise ConstrainedDecodingError(
                        "arguments must be a JSON object"
                    )
                return self._validate_arguments(parsed, function)
        raise ConstrainedDecodingError(
            "could not complete valid function arguments"
        )
