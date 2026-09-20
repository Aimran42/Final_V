import json
from pathlib import Path
from typing import List, Optional

from llm_sdk.llm_sdk import Small_LLM_Model
from src.constrained_decoding import ConstrainedDecoder
from src.models import FunctionDefinition, TestPrompt, FunctionCallResult


class Pipeline:
    def __init__(self, functions: List[FunctionDefinition],
                 tests: List[TestPrompt],
                 output_path: Path) -> None:
        self.functions = functions
        self.tests = tests
        self.output_path = output_path
        self.decoder: Optional[ConstrainedDecoder] = None

    def load_model(self) -> Small_LLM_Model:
        return Small_LLM_Model()

    def load_decoder(self) -> Optional[ConstrainedDecoder]:
        self.decoder = ConstrainedDecoder(self.load_model())
        return self.decoder

    def process_test(self, test: TestPrompt) -> Optional[FunctionCallResult]:
        if self.decoder is None:
            return None
        function = self.decoder.select_function(test.prompt, self.functions)
        parameters = self.decoder.decode_arguments(test.prompt, function)
        return FunctionCallResult(
            prompt=test.prompt,
            name=function.name,
            parameters=parameters,
        )

    def process_all_tests(self) -> List[FunctionCallResult]:
        results: List[FunctionCallResult] = []
        for test in self.tests:
            result = self.process_test(test)
            if result is not None:
                results.append(result)
        return results

    def write_results(self, results: List[FunctionCallResult]) -> None:
        with open(self.output_path, "w", encoding="utf-8") as output_file:
            json.dump(
                [result.model_dump() for result in results],
                output_file,
                indent=2,
            )

    def run(self) -> List[FunctionCallResult]:
        if self.load_decoder() is None:
            return []
        results = self.process_all_tests()
        if len(results) != len(self.tests):
            print("Error: not all prompts produced valid function calls")
            return []
        self.write_results(results)
        return results
