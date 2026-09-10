Problem:
I have a list of numpy arrays, and want to check if all the arrays have NaN. What is the quickest way of doing this?
Thanks,
A:
<code>
import numpy as np
a = [np.array([np.nan,2,3]),np.array([1,np.nan,3]),np.array([1,2,np.nan])]
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import pandas as pd
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = [
                np.array([np.nan, 2, 3]),
                np.array([1, np.nan, 3]),
                np.array([1, 2, np.nan]),
            ]
        elif test_case_id == 2:
            a = [
                np.array([np.nan, 2, 3]),
                np.array([1, np.nan, 3]),
                np.array([1, 2, 3]),
            ]
        elif test_case_id == 3:
            a = [np.array([10, 2, 3]), np.array([1, 9, 3]), np.array([1, 6, 3])]
        elif test_case_id == 4:
            a = [np.array([10, 4, 3]), np.array([1, np.nan, 3]), np.array([8, 6, 3])]
        elif test_case_id == 5:
            a = [
                np.array([np.nan, np.nan]),
                np.array([np.nan, np.nan]),
                np.array([np.nan, np.nan]),
            ]
        return a

    def generate_ans(data):
        _a = data
        a = _a
        result = True
        for arr in a:
            if any(np.isnan(arr)) == False:
                result = False
                break
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
a = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(5):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)

Return only the Python code snippet that replaces [insert]:
- No explanations, Markdown, comments, prints, or extra text; no non-English output
- No imports, function/class definitions, or placeholders; only the missing logic
- Do not reassign/reset provided variables/inputs; use them to compute the result
- The snippet must be syntactically valid and run as-is in this context; deterministic (assume temperature=0)
- Do not read/write files or call external services
- Output only the code lines for [insert]; no Markdown fences, no explanations, no bullets.
