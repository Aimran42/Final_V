from src.models import FunctionDefinition, TestPrompt
from typing import Any, List, Tuple, Dict
from pathlib import Path
import argparse
import json


class ParsingErr(Exception):
    """Custom exception raised when file loading or parsing fails."""
    pass


class Parser:
    """Parser class for handling arguments, loading JSON files """

    def parse_args(self) -> argparse.Namespace:
        """Parse command-line arguments for input and output file paths.

        Returns:
            argparse.Namespace: Parsed command-line arguments.
        """
        parser = argparse.ArgumentParser(
            prog="src",
            description="Run function-calling tests against function def",
        )
        parser.add_argument(
            "--functions_definition",
            type=str,
            default="data/input/functions_definition.json",
            help="Path to the functions definition JSON file.",
        )
        parser.add_argument(
            "--input",
            type=str,
            default="data/input/function_calling_tests.json",
            help="Path to the function calling tests JSON file.",
        )
        parser.add_argument(
            "--output",
            type=str,
            default="data/output/function_calls.json",
            help="Path to write the output JSON file.",
        )
        return parser.parse_args()

    @staticmethod
    def check_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        """Cheeking duplicate keys"""
        seen: set = set()
        for key, _ in pairs:
            if not key:
                raise ParsingErr(f"Invalid key: '{key}'")
            if key in seen:
                raise ParsingErr(f"Duplicate key: '{key}'")
            seen.add(key)
        return dict(pairs)

    @staticmethod
    def load_json_file(path: str) -> Any:
        """Load and parse content from a JSON file.
        Args:
            path (str): File system path to the JSON file.

        Returns:
            Any: Parsed JSON data structure (dict or list).
        """
        file_path = Path(path)
        if not file_path.is_file():
            raise ParsingErr(f"File not found: {path}")
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f, object_pairs_hook=Parser.check_keys)

    @staticmethod
    def parse_functions_definition(path: str) -> List[FunctionDefinition]:
        """Parse and validate function definitions from a JSON file.
        Args:
            path (str): Path to the function definitions JSON file.

        Returns:
            List of validated FunctionDefinition models.
        """
        raw_data = Parser.load_json_file(path)
        objects = [FunctionDefinition(**item) for item in raw_data]
        fn_names = [obj.name for obj in objects]
        duplicates = [fn for fn in fn_names if fn_names.count(fn) > 1]
        if duplicates:
            raise ParsingErr(f"duplicate function name {duplicates[0]}")
        if not isinstance(raw_data, list):
            raise ParsingErr(f"{path} must contain a JSON array")
        return [FunctionDefinition(**item) for item in raw_data]

    @staticmethod
    def parse_function_calling_tests(path: str) -> List[TestPrompt]:
        """Parse and validate test prompts from a JSON file.
        Args:
            path (str): Path to the test prompts JSON file.

        Returns:
            List[TestPrompt]: List of validated TestPrompt models.
        """
        raw_data = Parser.load_json_file(path)
        if isinstance(raw_data, dict):
            raw_data = [raw_data]
        if not isinstance(raw_data, list):
            raise ParsingErr(f"{path} must contain a JSON array")
        return [TestPrompt(**item) for item in raw_data]

    def parsing(self) -> Tuple[List[FunctionDefinition],
                               List[TestPrompt], Path]:
        """Execute the complete parsing pipeline: parse CLI arguments,
                load inputs, and initialize the output file.
        """
        args = Parser.parse_args(self)

        ft = Parser.parse_functions_definition(args.functions_definition)
        tests = Parser.parse_function_calling_tests(args.input)

        results = [test.model_dump() for test in tests]

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return (ft, tests, output_path)
