Problem:
I want to figure out how to remove nan values from my array. 
For example, My array looks something like this:
x = [[1400, 1500, 1600, nan], [1800, nan, nan ,1700]] #Not in this exact configuration
How can I remove the nan values from x?
Note that after removing nan, the result cannot be np.array due to dimension mismatch, so I want to convert the result to list of lists.
x = [[1400, 1500, 1600], [1800, 1700]]
A:
<code>
import numpy as np
x = np.array([[1400, 1500, 1600, np.nan], [1800, np.nan, np.nan ,1700]])
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            x = np.array([[1400, 1500, 1600, np.nan], [1800, np.nan, np.nan, 1700]])
        elif test_case_id == 2:
            x = np.array([[1, 2, np.nan], [3, np.nan, np.nan]])
        elif test_case_id == 3:
            x = np.array([[5, 5, np.nan, np.nan, 2], [3, 4, 5, 6, 7]])
        return x

    def generate_ans(data):
        _a = data
        x = _a
        result = [x[i, row] for i, row in enumerate(~np.isnan(x))]
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    for arr1, arr2 in zip(ans, result):
        np.testing.assert_array_equal(arr1, arr2)
    return 1


exec_context = r"""
import numpy as np
x = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(3):
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
