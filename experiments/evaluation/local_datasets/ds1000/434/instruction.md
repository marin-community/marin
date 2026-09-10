Problem:
I have a 2D array `a` to represent a many-many mapping :
0   3   1   3
3   0   0   0
1   0   0   0
3   0   0   0
What is the quickest way to 'zero' out rows and column entries corresponding to particular indices (e.g. zero_rows = [0, 1], zero_cols = [0, 1] corresponds to the 1st and 2nd row / column) in this array?
A:
<code>
import numpy as np
a = np.array([[0, 3, 1, 3], [3, 0, 0, 0], [1, 0, 0, 0], [3, 0, 0, 0]])
zero_rows = [1, 3]
zero_cols = [1, 2]
</code>
a = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import pandas as pd
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.array([[0, 3, 1, 3], [3, 0, 0, 0], [1, 0, 0, 0], [3, 0, 0, 0]])
            zero_rows = [1, 3]
            zero_cols = [1, 2]
        return a, zero_rows, zero_cols

    def generate_ans(data):
        _a = data
        a, zero_rows, zero_cols = _a
        a[zero_rows, :] = 0
        a[:, zero_cols] = 0
        return a

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
a, zero_rows, zero_cols = test_input
[insert]
result = a
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
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
