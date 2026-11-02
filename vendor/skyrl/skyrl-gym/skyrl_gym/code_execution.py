"""Execute candidate programs in Shellbox and compare hidden references on the worker."""

import asyncio
import json
import math
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

from shellbox.machine import ExitReason, Machine
from verifyit.modes.grade_exact import grade_exact_candidate
from verifyit.spec import ExactSpec

from skyrl_gym.code_candidate import CANDIDATE_RESULT_PREFIX
from skyrl_gym.envs.lcb import livecodebench as runtime
from skyrl_gym.python_execution import PythonKernel

CANDIDATE_SCRIPT = Path(__file__).with_name("code_candidate.py")


async def validate_code_example(
    machine: Machine, ground_truth: runtime.CodeGroundTruth, positive_response: str, negative_response: str
) -> str:
    """Validate source tests against two candidates in the supplied Shellbox machine."""
    normalized = runtime.normalize_lcb_ground_truth(ground_truth)
    tests = json.loads(normalized)
    for response, accepted in ((positive_response, True), (negative_response, False)):
        reward, _ = await execute_code(machine, tests, runtime.extract_code_from_model(response) or "")
        if (reward == 1.0) != accepted:
            raise ValueError("Code preflight candidates do not match the supplied tests")
    return normalized


def _typed(value):
    if value is None or isinstance(value, (str, bool)):
        return [type(value).__name__, value]
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite result")
        number = Fraction(value)
        return ["number", number.numerator, number.denominator]
    if isinstance(value, list):
        return ["list", [_typed(item) for item in value]]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("unsupported result key")
        return ["dict", [[key, _typed(value[key])] for key in sorted(value)]]
    raise ValueError("unsupported result type")


def _unwire(value):
    kind, payload = value
    if kind == "tuple":
        return tuple(_unwire(item) for item in payload)
    if kind == "list":
        return [_unwire(item) for item in payload]
    if kind == "dict":
        return {_unwire(key): _unwire(item) for key, item in payload}
    if kind in {"NoneType", "str", "bool", "int", "float"} and type(payload).__name__ == kind:
        return payload
    raise ValueError("malformed output")


def _stdio(value: str):
    lines = []
    for line in value.strip().split("\n"):
        line = line.strip()
        try:
            numbers = [Decimal(token) for token in line.split()]
        except InvalidOperation:
            lines.append(["literal", line])
        else:
            if any(not number.is_finite() for number in numbers):
                raise ValueError("nonfinite output")
            lines.append(
                ["numbers", [[Fraction(number).numerator, Fraction(number).denominator] for number in numbers]]
            )
    return lines


async def execute_code(
    machine: Machine,
    tests: list,
    code: str,
    *,
    timeout: float = 6,
    reward_mode: str = runtime.BINARY_REWARD_MODE,
    limits: runtime.VerifierLimits = runtime.DEFAULT_LIMITS,
) -> tuple[float, dict]:
    """Return source-compatible test results without sending expected answers to candidate execution."""
    if reward_mode not in runtime.LCB_REWARD_MODES:
        raise ValueError(f"Unsupported LCB reward_mode: {reward_mode!r}")
    reference = json.loads(runtime.postprocess_lcb_sample(tests)["input_output"])
    function = reference.get("fn_name")
    inputs = reference["inputs"]
    targets = [json.loads(value) for value in reference["outputs"]] if function else reference["outputs"]
    expected = [_typed(value) if function else _stdio(value) for value in targets]
    if function:
        inputs = [[json.loads(line) for line in value.split("\n")] for value in inputs]
    if len(inputs) != len(expected) or not expected:
        raise RuntimeError("Code verification requires aligned tests")
    if not code:
        return 0.0, {"test_results": [], "total_tests": len(expected), "reason": "missing_code"}
    deadline = (timeout + 1) * len(expected) + 5
    if limits.total_timeout_seconds is not None:
        deadline = min(deadline, limits.total_timeout_seconds)
    kernel = PythonKernel(machine, memory_bytes=limits.max_memory_bytes)
    try:
        # Runtime setup failures are infrastructure errors, not candidate deadline failures.
        await kernel.start()
        script = f"{kernel.directory}/code_candidate.py"
        await kernel.machine.upload(CANDIDATE_SCRIPT, script)
        ready = await kernel.execute("from code_candidate import candidate_method, evaluate", timeout=timeout)
        if (
            ready.reason != ExitReason.EXITED
            or ready.exit_code != 0
            or ready.stdout_truncated
            or ready.stderr_truncated
        ):
            raise RuntimeError("Code verifier runtime setup failed")
        try:
            async with asyncio.timeout(deadline):
                return await _candidate_results(kernel, inputs, expected, code, function, timeout, reward_mode)
        except TimeoutError:
            return 0.0, {"test_results": [-3], "total_tests": len(expected), "reason": "candidate_timeout"}
    finally:
        await kernel.close()


async def _candidate_results(kernel, inputs, expected, code, function, timeout, reward_mode):
    startup = (
        f"try:\n    _candidate_method = candidate_method({code!r}, {function!r})\n"
        "    print('READY')\nexcept (Exception, SystemExit):\n    print('CANDIDATE_FAILURE')\n"
    )
    initialized = await kernel.execute(startup, timeout=timeout)
    if initialized.reason == ExitReason.TIMED_OUT:
        return 0.0, {"test_results": [-3], "total_tests": len(expected)}
    if (
        initialized.exit_code != 0
        or initialized.stdout_truncated
        or initialized.stderr_truncated
        or initialized.stdout.strip() != b"READY"
    ):
        return 0.0, {"test_results": [-4], "total_tests": len(expected)}
    result_prefix = CANDIDATE_RESULT_PREFIX.encode()
    results = []
    for arguments, target in zip(inputs, expected):
        call = f"evaluate(_candidate_method, {arguments!r}, {function!r})"
        output = await kernel.execute(call, timeout=timeout)
        if output.reason == ExitReason.TIMED_OUT:
            results.append(-3)
        elif output.exit_code != 0 or output.stdout_truncated or output.stderr_truncated:
            results.append(-4)
        elif not output.stdout.strip().startswith(result_prefix):
            results.append(-4)
        else:
            try:
                prediction = _unwire(json.loads(output.stdout.strip()[len(result_prefix) :]))
                actual = _typed(prediction) if function else _stdio(prediction)
            except (ValueError, TypeError):
                results.append(-2)
            else:
                verdict = grade_exact_candidate(
                    ExactSpec(
                        expected=(json.dumps(target, ensure_ascii=False),),
                        ignore_case=False,
                        ignore_whitespace=False,
                        strip_outer_whitespace=False,
                    ),
                    json.dumps(actual, ensure_ascii=False),
                )
                results.append(True if verdict.reward == 1.0 else -2)
        if kernel.failure is not None:
            return 0.0, {"test_results": results, "total_tests": len(expected), "reason": "candidate_kernel_terminated"}
        if results[-1] is not True and reward_mode == runtime.BINARY_REWARD_MODE:
            break
    reward = (
        sum(result is True for result in results) / len(results)
        if reward_mode == runtime.FRACTIONAL_REWARD_MODE
        else float(all(result is True for result in results))
    )
    return reward, {"test_results": results, "total_tests": len(expected)}
