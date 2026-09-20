*This project has been created as part of the 42 curriculum by momahdam*


# Call_me_maybe

## Description

`Call_me_maybe` is a function-calling engine built on top of a small (0.6B parameter)
local language model, `Qwen/Qwen3-0.6B`. Given a natural-language prompt such as
*"What is the sum of 2 and 3?"*, the goal is not to answer the question in prose,
but to translate it into a structured, machine-executable function call:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

Small language models are notoriously unreliable at producing valid structured
output when simply prompted for it — success rates well below what production
systems require. This project solves that problem with **constrained decoding**:
instead of hoping the model spontaneously emits correct JSON, we intervene at
every generation step, masking out any token that would break the required
structure or schema, so the model can only ever produce output that is valid.

The project reads a list of available functions and a list of test prompts,
selects the correct function for each prompt using the model, extracts the
correct arguments, and writes the results to a JSON file.

## Instructions

### Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- The `llm_sdk` package copied into the project root (sibling of `src/`)

### Installation

```bash
make install
```

This resolves and installs project dependencies (`pydantic`, `numpy`) via `uv`.

### Running

```bash
make run
```

or directly:
```bash
make
```
Alternatively
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

All three arguments are optional; by default the program reads from
`data/input/` and writes to `data/output/function_calls.json`.

### Other Makefile targets

| Target | Description |
|---|---|
| `make install` | Install dependencies |
| `make run` | Run the program with default paths |
| `make debug` | Run the program under `pdb` |
| `make lint` | Run `flake8` and `mypy` |
| `make lint-strict` | Run `mypy --strict` |
| `make clean` | Remove caches and temporary files to keep the project environment clean.|


### Example usage

```bash
$ make run

_____________________SUMMARY_____________________

Loaded 5 function definitions
Loaded 11 test prompts
Wrote 11 results to data/output/function_calls.json
```

The resulting `data/output/function_calls.json` contains one object per
prompt, each with the selected function name and its extracted arguments.

## Algorithm Explanation

The core technique is **constrained decoding**: at each step of text
generation, before a token is chosen, we determine which tokens are
*allowed* given what has been generated so far, and only pick among those.

The pipeline uses two complementary strategies depending on what is being
generated:

**1. Enum-constrained decoding (function selection), via `TokenTrie`.**
The set of valid function names is finite and known in advance. We build a
character-level prefix trie from the candidate names. At each generation
step, we ask the model for its next-token logits, rank all candidate tokens
by score, and walk down that ranked list until we find the first token whose
text keeps us on a valid path through the trie (`is_prefix`). We stop once
the accumulated text exactly matches a complete function name
(`is_complete`). This guarantees the selected name is always one of the
functions actually provided — never a hallucinated one.

**2. Schema-constrained JSON decoding (argument extraction).**
Argument values (numbers, strings) are not drawn from a finite set, so a
trie doesn't apply. Instead, at each step we track whether the partial JSON
string generated so far (a) is still syntactically well-formed (balanced
braces/brackets, correctly paired quotes) and (b) would still, once
completed, only ever use keys present in the target function's parameter
schema. Any candidate token that would violate either condition has its
logit effectively masked out (never selected). Generation stops as soon as
the object is syntactically closed.

**Why not just prompt the model and hope?** Because unconstrained
generation on a 0.6B model produces invalid or malformed JSON often enough
to be unusable in production. Constrained decoding moves the correctness
guarantee from "the model's training" to "the decoding algorithm," which
is deterministic and enforceable regardless of model size.

## Design Decisions

- **Two decoding strategies, not one.** An enum/trie approach for finite
  candidate sets (function names) is both faster and stricter than treating
  name selection as free-form JSON generation with post-hoc string matching.
- **Sorted-logit, early-exit token selection.** Rather than scanning the
  entire ~150,000-token vocabulary against a validity check at every step
  (prohibitively slow), we sort candidate tokens by model score once per
  step and stop at the first valid one. In practice the model's top
  prediction is usually already valid, so this is close to constant-time
  per step in the common case.
- **Pydantic models for all structured data** (`FunctionDefinition`,
  `TestPrompt`, `FunctionCallResult`), giving free validation and clear,
  typed error messages when input files don't match the expected schema.
- **Clear separation of concerns.** `Parser` only handles CLI args and
  input file loading; `ConstrainedDecoder` only handles model-level token
  decisions; `Pipeline` only orchestrates the two and writes output. No
  module needs to understand the internals of another.
- **Fail loud, fail gracefully.** Every I/O and model-interaction boundary
  is wrapped in explicit error handling that prints a clear message and
  returns a non-zero exit code, rather than letting a raw traceback reach
  the user.

## Performance Analysis

- **Reliability:** because token choices are filtered at generation time
  rather than validated afterward, output is JSON-parseable by
  construction — invalid structure is never a possible outcome, regardless
  of model confidence.
- **Function-selection accuracy:** constrained to the exact set of
  provided function names, so accuracy depends only on whether the model's
  logits favor the semantically correct name among the (small) set of
  valid completions — a much easier task than free-form generation.
- **Speed:** the dominant cost is the number of forward passes through the
  model (one per generated token), since the SDK does not expose KV-caching
  and recomputes over the growing sequence each step. The sorted-logit,
  early-exit search removes the vocabulary-scan bottleneck, leaving model
  inference itself as the remaining cost. Step budgets are kept small
  (32 steps for function-name generation) to stay within a reasonable
  time budget across all test prompts.
- **Time**
```python
    if pl.processing_seconds is not None:
        print(f"Processed all prompts in {pl.processing_seconds:.2f} seconds")
```
## Challenges Faced

- **Initial constrained decoding was far too slow.** The first
  implementation checked every token in the ~150,000-entry vocabulary
  against a JSON-validity function at every single generation step. This
  produced correct but practically unusable performance (effectively
  hanging on a handful of prompts). The fix was to sort tokens by logit
  score once per step and stop at the first valid candidate, rather than
  exhaustively checking the entire vocabulary.
- **No KV-cache in the provided SDK.** Each call to
  `get_logits_from_input_ids` recomputes attention over the whole sequence
  so far, since only the public SDK surface could be used (private/internal
  attributes were off-limits). This bounds how large step budgets can be
  kept without exceeding the time budget.
- **Balancing correctness with simplicity.** Early versions duplicated
  validity-checking logic across multiple classes; the code was refactored
  so that `ConstrainedDecoder` is the single place owning all token-level
  decisions, while `Pipeline` only orchestrates.

## Testing Strategy

- Manual testing against the provided `function_calling_tests.json` and
  `functions_definition.json`, comparing selected function names and
  argument values against the expected mapping for each prompt.
- Deliberate fault injection: read-permission removal on input files,
  malformed/missing JSON, and empty function/test lists, to verify the
  program reports clear errors and exits with a non-zero status rather
  than crashing.
- Edge cases checked with peers: prompts with no unambiguous matching function,
  numeric parameters with large values, and functions with zero or
  multiple parameters, duplicates.

## Resources

- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers)
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Constrained decoding / guided generation overview (Outlines library docs)](https://dottxt-ai.github.io/outlines/) — read for conceptual background only; the `outlines` package itself is not used or imported anywhere in this project, per the assignment's restrictions.
- [pydantic documentation](https://docs.pydantic.dev/)
- [uv documentation](https://docs.astral.sh/uv/)

### AI usage

AI assistance (Claude) was used during this project for:

- Writing this README, based on the actual
  implementation and design decisions made throughout development.
- Documentation like docs strings Numpy Style describing each module, function, method or class ...
