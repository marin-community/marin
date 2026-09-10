Problem:
I have a file with arrays or different shapes. I want to zeropad all the array to match the largest shape. The largest shape is (93,13).
To test this I have the following code:
a = np.ones((41,12))
how can I zero pad this array to match the shape of (93,13)? And ultimately, how can I do it for thousands of rows? Specifically, I want to pad the array to left, right equally and top, bottom equally. If not equal, put the rest row/column to the bottom/right.
e.g. convert [[1]] into [[0,0,0],[0,1,0],[0,0,0]]
A:
<code>
import numpy as np
a = np.ones((41, 12))
shape = (93, 13)
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
            a = np.ones((41, 12))
            shape = (93, 13)
        elif test_case_id == 2:
            a = np.ones((41, 13))
            shape = (93, 13)
        elif test_case_id == 3:
            a = np.ones((93, 11))
            shape = (93, 13)
        elif test_case_id == 4:
            a = np.ones((42, 10))
            shape = (93, 13)
        return a, shape

    def generate_ans(data):
        _a = data
        a, shape = _a

        def to_shape(a, shape):
            y_, x_ = shape
            y, x = a.shape
            y_pad = y_ - y
            x_pad = x_ - x
            return np.pad(
                a,
                (
                    (y_pad // 2, y_pad // 2 + y_pad % 2),
                    (x_pad // 2, x_pad // 2 + x_pad % 2),
                ),
                mode="constant",
            )

        result = to_shape(a, shape)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
a, shape = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(4):
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
