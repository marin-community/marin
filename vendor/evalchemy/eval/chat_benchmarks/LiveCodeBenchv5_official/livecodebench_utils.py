"""
Code from https://github.com/NovaSky-AI/SkyThought/blob/main/skythought/tools/util/livecodebench/testing_util.py

The candidate-execution helpers (``reliability_guard``, ``run_test_func``,
``run_test_std``, ``prepare_test_input_output_*``, ``run_tests_for_one_example``)
were replaced by the bounded grader in ``eval/graders/livecodebench.py`` (see
issue #147); ``lcb_run`` below is a thin adapter that keeps the vendored call
signature and delegates there.
"""

import base64
import json
import pickle
import zlib

from eval.graders import livecodebench as _lcb_grader


def has_test_type(tests, type):  # helper to select specific type of problems
    """
    Check if any test in the test list has 'testtype' set to 'type'.
    """
    test_list = json.loads(tests)
    for test in test_list:
        if test.get("testtype") == type:
            return True
    return False


def translate_private_test_cases(encoded_data):
    decoded_data = base64.b64decode(encoded_data)
    decompressed_data = zlib.decompress(decoded_data)
    original_data = pickle.loads(decompressed_data)
    return json.loads(original_data)


def map_to_example(row):
    return {
        "prompt": row["question_content"],
        "test": row["private_test_cases"],
        "entry_point": row["starter_code"],
        "task_id": row["question_id"],
        "is_stdin": has_test_type(row["public_test_cases"], "stdin"),
        "public_test_cases": row["public_test_cases"],
        "difficulty": row["difficulty"],
    }


def post_process_code(code):
    code = code.split("</code>")[0]
    code = code.replace("```python", "")
    code = code.split("```")[0]
    code = code.replace("<code>", "")
    return code


def lcb_run(problem, completion, timeout, is_extracted):
    """Score an LCB problem by delegating to the shared bounded grader.

    Returns the per-test ``(passed, message, value, elapsed)`` tuples the vendored
    ``lcb_run`` contract has always returned, so the ``check_correctness`` consumers here are
    unchanged. See :func:`eval.graders.livecodebench.run_lcb_tests` for the bounds.
    """
    return _lcb_grader.run_lcb_tests(problem, completion, timeout, is_extracted)
