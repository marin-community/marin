"""LiveCodeBench adapter to the shared bounded code grader.

Adapted from https://github.com/NovaSky-AI/SkyThought/blob/main/skythought/tools/util/livecodebench/testing_util.py.
"""

from eval.graders import livecodebench as _lcb_grader


def lcb_run(problem, completion, timeout, is_extracted):
    """Score an LCB problem by delegating to the shared bounded grader.

    Returns the per-test ``(passed, message, value, elapsed)`` tuples the vendored
    ``lcb_run`` contract has always returned, so the ``check_correctness`` consumers here are
    unchanged. See :func:`eval.graders.livecodebench.run_lcb_tests` for the bounds.
    """
    return _lcb_grader.run_lcb_tests(problem, completion, timeout, is_extracted)
