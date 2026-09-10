Problem:
I would like to resample a numpy array as suggested here Resampling a numpy array representing an image however this resampling will do so by a factor i.e.
x = np.arange(9).reshape(3,3)
print scipy.ndimage.zoom(x, 2, order=1)
Will create a shape of (6,6) but how can I resample an array to its best approximation within a (4,6),(6,8) or (6,10) shape for instance?
A:
<code>
import numpy as np
import scipy.ndimage
x = np.arange(9).reshape(3, 3)
shape = (6, 8)
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
import scipy.ndimage


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            shape = (6, 8)
        elif test_case_id == 2:
            shape = (6, 10)
        elif test_case_id == 3:
            shape = (4, 6)
        elif test_case_id == 4:
            np.random.seed(42)
            shape = np.random.randint(4, 20, (2,))
        x = np.arange(9).reshape(3, 3)
        return x, shape

    def generate_ans(data):
        _a = data
        x, shape = _a
        result = scipy.ndimage.zoom(
            x, zoom=(shape[0] / x.shape[0], shape[1] / x.shape[1]), order=1
        )
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
import scipy.ndimage
x, shape = test_input
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
