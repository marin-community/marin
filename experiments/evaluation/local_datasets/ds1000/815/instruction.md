Problem:
How to find relative extrema of a 2D array? An element is a relative extrema if it is less or equal to the neighbouring n (e.g. n = 2) elements forwards and backwards in the row. 
The result should be a list of indices of those elements, [0, 1] stands for arr[0][1]. It should be arranged like
[[0, 1], [0, 5], [1, 1], [1, 4], [2, 3], [2, 5], ...]
A:
<code>
import numpy as np
from scipy import signal
arr = np.array([[-624.59309896, -624.59309896, -624.59309896,
                      -625., -625., -625.,], [3, 0, 0, 1, 2, 4]])
n = 2
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
from scipy import signal


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            arr = np.array(
                [
                    [
                        -624.59309896,
                        -624.59309896,
                        -624.59309896,
                        -625.0,
                        -625.0,
                        -625.0,
                    ],
                    [3, 0, 0, 1, 2, 4],
                ]
            )
            n = 2
        elif test_case_id == 2:
            np.random.seed(42)
            arr = (np.random.rand(50, 10) - 0.5) * 10
            n = np.random.randint(2, 4)
        return arr, n

    def generate_ans(data):
        _a = data
        arr, n = _a
        res = signal.argrelextrema(arr, np.less_equal, order=n, axis=1)
        result = np.zeros((res[0].shape[0], 2)).astype(int)
        result[:, 0] = res[0]
        result[:, 1] = res[1]
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    assert np.array(result).dtype == np.int64 or np.array(result).dtype == np.int32
    return 1


exec_context = r"""
import numpy as np
from scipy import signal
arr, n = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
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
