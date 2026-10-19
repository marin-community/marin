"""LiveCodeBench adapter to the shared bounded code grader.

Adapted from https://github.com/NovaSky-AI/SkyThought/blob/main/skythought/tools/util/livecodebench/testing_util.py.
"""

from eval.graders import livecodebench as _lcb_grader


def lcb_run(problem, completion, timeout):
    """Return per-test pass results from the shared bounded code grader."""
    return _lcb_grader.run_lcb_tests(problem, completion, timeout)
