"""Source code-test data and Python input adapters."""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict


STDIN_TEST_TYPE = "stdin"
FUNCTIONAL_TEST_TYPE = "functional"
FUNCTION_NAME_KEY = "func_name"
BINARY_REWARD_MODE = "binary"
FRACTIONAL_REWARD_MODE = "fractional"
LCB_REWARD_MODES = frozenset({BINARY_REWARD_MODE, FRACTIONAL_REWARD_MODE})
CodeGroundTruth = str | Mapping[str, Any] | list[Mapping[str, Any]]
ParsedCodeGroundTruth = Mapping[str, Any] | list[Any] | str | int | float | bool | None
_STDIN_SOURCE_TEST_TYPES = {STDIN_TEST_TYPE, "stdin_stdout"}
_FUNCTIONAL_SOURCE_TEST_TYPES = {FUNCTIONAL_TEST_TYPE, "call_based"}
_SOURCE_FUNCTION_NAME_KEY = "fn_name"


class CanonicalCodeCase(TypedDict):
    input: str
    output: str
    testtype: str
    metadata: NotRequired[dict[str, str]]


def _parse_code_ground_truth(ground_truth: CodeGroundTruth) -> ParsedCodeGroundTruth:
    if not isinstance(ground_truth, str):
        return ground_truth
    try:
        return json.loads(ground_truth)
    except json.JSONDecodeError as exc:
        raise ValueError("LCB ground_truth must be valid JSON.") from exc


def _canonical_code_case(case: Mapping[str, Any]) -> CanonicalCodeCase:
    test_input = case.get("input")
    expected_output = case.get("output")
    test_type = case.get("testtype", case.get("type"))
    if test_type in _STDIN_SOURCE_TEST_TYPES:
        if not isinstance(test_input, str) or not isinstance(expected_output, str):
            raise TypeError("Standard-input LCB test-case input and output must be strings.")
        return {"input": test_input, "output": expected_output, "testtype": STDIN_TEST_TYPE}
    if test_type in _FUNCTIONAL_SOURCE_TEST_TYPES:
        function_name = case.get(_SOURCE_FUNCTION_NAME_KEY)
        metadata = case.get("metadata")
        if function_name is None and isinstance(metadata, Mapping):
            function_name = metadata.get(FUNCTION_NAME_KEY)
        if not isinstance(function_name, str) or not function_name:
            raise ValueError("Functional LCB test cases require a function name.")
        if isinstance(test_input, list):
            test_input = "\n".join(json.dumps(argument) for argument in test_input)
        elif not isinstance(test_input, str):
            raise TypeError("Functional LCB test-case input must be a list of arguments or an encoded string.")
        if not isinstance(expected_output, str):
            expected_output = json.dumps(expected_output)
        return {
            "input": test_input,
            "output": expected_output,
            "testtype": FUNCTIONAL_TEST_TYPE,
            "metadata": {FUNCTION_NAME_KEY: function_name},
        }
    raise ValueError(f"Unsupported LCB test-case type: {test_type!r}.")


def _canonical_apps_cases(ground_truth: Mapping[str, Any]) -> list[CanonicalCodeCase]:
    inputs = ground_truth.get("inputs")
    outputs = ground_truth.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        raise TypeError("APPS ground_truth requires input and output lists.")
    if not inputs or len(inputs) != len(outputs):
        raise ValueError("APPS ground_truth requires equally sized, non-empty input and output lists.")
    function_name = ground_truth.get(_SOURCE_FUNCTION_NAME_KEY)
    test_type = FUNCTIONAL_TEST_TYPE if function_name is not None else STDIN_TEST_TYPE
    return [
        _canonical_code_case(
            {
                "input": test_input,
                "output": expected_output,
                "testtype": test_type,
                _SOURCE_FUNCTION_NAME_KEY: function_name,
            }
        )
        for test_input, expected_output in zip(inputs, outputs)
    ]


def normalize_lcb_ground_truth(ground_truth: CodeGroundTruth) -> str:
    """Return a canonical JSON-encoded LCB case list.

    Accepts canonical case lists, APPS ``inputs``/``outputs`` mappings, and open-r1
    ``test_cases`` mappings, either directly or as JSON text. All cases must use one
    execution mode and, for functional tests, one function name.
    """
    parsed = _parse_code_ground_truth(ground_truth)
    if isinstance(parsed, Mapping) and "test_cases" in parsed:
        if parsed.get("language") not in {None, "python"}:
            raise ValueError("LCB supports only Python verification specs.")
        cases = parsed["test_cases"]
    elif isinstance(parsed, Mapping):
        cases = _canonical_apps_cases(parsed)
    else:
        cases = parsed
    if not isinstance(cases, list) or not cases:
        raise ValueError("LCB ground_truth must contain at least one test case.")
    if not all(isinstance(case, Mapping) for case in cases):
        raise TypeError("LCB test cases must be mappings.")
    canonical_cases = [_canonical_code_case(case) for case in cases]
    test_types = {case["testtype"] for case in canonical_cases}
    if len(test_types) != 1:
        raise ValueError("LCB ground_truth cannot mix standard-input and functional test cases.")
    if FUNCTIONAL_TEST_TYPE in test_types:
        function_names = {case["metadata"][FUNCTION_NAME_KEY] for case in canonical_cases}
        if len(function_names) != 1:
            raise ValueError("All functional LCB test cases must use the same function name.")
    return json.dumps(canonical_cases, sort_keys=True)


def postprocess_lcb_sample(sample):
    sample_inputs = [sample["input"] for sample in sample]
    sample_outputs = [sample["output"] for sample in sample]

    sample_dict = {
        "inputs": sample_inputs,
        "outputs": sample_outputs,
    }

    if sample[0].get("testtype") == FUNCTIONAL_TEST_TYPE:
        metadata = sample[0].get("metadata", {})
        fn_name = metadata.get(FUNCTION_NAME_KEY, None)
        assert fn_name is not None, (
            f"Function name is not found, check if your LCB data is preprocessed correctly: {metadata}"
        )
        # Fill in the blank
        sample_dict["fn_name"] = fn_name

    sample = {
        "input_output": json.dumps(sample_dict),
    }
    return sample


@dataclass(frozen=True)
class VerifierLimits:
    """Resource bounds for one candidate interpreter.

    ``max_memory_bytes`` sets hard limits on address space and data size.
    ``total_timeout_seconds`` limits the full grading attempt, including startup.
    A ``None`` value removes the corresponding bound.
    """

    max_memory_bytes: int | None = 4 * 1024**3
    total_timeout_seconds: float | None = 300.0


DEFAULT_LIMITS = VerifierLimits()


def extract_code_from_model(model_response: str):
    """Return the last fenced code block, or None if the response has no code block."""
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", model_response, re.DOTALL)
    if not code_blocks:
        return None
    return code_blocks[-1].strip()
