# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Metrics for Kelp tree diffusion evaluation.

Provides functions to compute:
- Syntactic validity rate: fraction of generated programs that parse as valid Python
- pass@k: probability that at least one of k samples passes all tests
"""

import ast
import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidityResult:
    """Result of syntactic validity evaluation."""

    total: int
    valid: int
    validity_rate: float
    invalid_samples: list[tuple[int, str, str]]  # (index, code, error)


def check_python_syntax(code: str) -> tuple[bool, str | None]:
    """Check if code is syntactically valid Python.

    Args:
        code: Python source code to check.

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


def compute_validity_rate(
    generated_codes: list[str],
    *,
    return_details: bool = False,
) -> float | ValidityResult:
    """Compute the syntactic validity rate for generated programs.

    Args:
        generated_codes: List of generated Python code strings.
        return_details: If True, return detailed ValidityResult instead of just rate.

    Returns:
        Validity rate as a float between 0 and 1, or ValidityResult if return_details=True.
    """
    if not generated_codes:
        if return_details:
            return ValidityResult(total=0, valid=0, validity_rate=0.0, invalid_samples=[])
        return 0.0

    valid_count = 0
    invalid_samples = []

    for i, code in enumerate(generated_codes):
        is_valid, error = check_python_syntax(code)
        if is_valid:
            valid_count += 1
        else:
            invalid_samples.append((i, code, error or "Unknown error"))

    rate = valid_count / len(generated_codes)

    if return_details:
        return ValidityResult(
            total=len(generated_codes),
            valid=valid_count,
            validity_rate=rate,
            invalid_samples=invalid_samples,
        )
    return rate


def _estimator(n: int, c: int, k: int) -> float:
    """Compute pass@k estimator.

    This is the unbiased estimator from the Codex paper:
    pass@k = 1 - C(n-c, k) / C(n, k)

    where C(a, b) is the binomial coefficient "a choose b".

    Args:
        n: Total number of samples.
        c: Number of correct (passing) samples.
        k: k value for pass@k.

    Returns:
        Estimated pass@k probability.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(range(n - c - k + 1, n - c + 1)) / math.prod(range(n - k + 1, n + 1))


@dataclass
class PassAtKResult:
    """Result of pass@k evaluation."""

    k: int
    pass_rate: float
    total_problems: int
    per_problem_scores: list[float]


def compute_pass_at_k(
    results: list[list[bool]],
    k: int,
) -> PassAtKResult:
    """Compute pass@k metric for code generation.

    Uses the unbiased estimator from the Codex paper (Chen et al., 2021).

    Args:
        results: List of lists of test results per problem.
                 results[i][j] = True if j-th sample for i-th problem passes all tests.
        k: Number of attempts (k in pass@k).

    Returns:
        PassAtKResult with pass@k rate and per-problem scores.
    """
    if not results:
        return PassAtKResult(k=k, pass_rate=0.0, total_problems=0, per_problem_scores=[])

    per_problem_scores = []

    for problem_results in results:
        n = len(problem_results)
        if n == 0:
            per_problem_scores.append(0.0)
            continue

        c = sum(problem_results)  # number of correct samples
        actual_k = min(k, n)  # can't use more samples than we have
        score = _estimator(n, c, actual_k)
        per_problem_scores.append(score)

    pass_rate = float(np.mean(per_problem_scores))

    return PassAtKResult(
        k=k,
        pass_rate=pass_rate,
        total_problems=len(results),
        per_problem_scores=per_problem_scores,
    )


def _run_test_in_subprocess(code: str, test: str, queue: "multiprocessing.Queue") -> None:
    """Execute a single test in a subprocess. Must be module-level for pickling."""
    try:
        exec_globals: dict = {}
        exec(code, exec_globals)
        exec(test, exec_globals)
        queue.put(True)
    except Exception:
        queue.put(False)


def execute_python_with_tests(code: str, test_cases: list[str], timeout: float = 5.0) -> list[bool]:
    """Execute generated code with test cases.

    Args:
        code: Generated Python code.
        test_cases: List of test assertion strings to execute.
        timeout: Timeout in seconds for each test.

    Returns:
        List of booleans indicating whether each test passed.
    """
    import multiprocessing

    results = []
    for test in test_cases:
        queue: multiprocessing.Queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=_run_test_in_subprocess, args=(code, test, queue))
        process.start()
        process.join(timeout=timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            results.append(False)  # Timeout
        elif queue.empty():
            results.append(False)  # Process died without result
        else:
            results.append(queue.get())

    return results
