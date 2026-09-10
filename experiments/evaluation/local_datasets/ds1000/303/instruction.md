Problem:
I want to convert a 1-dimensional array into a 2-dimensional array by specifying the number of columns in the 2D array. Something that would work like this:
> import numpy as np
> A = np.array([1,2,3,4,5,6,7])
> B = vec2matrix(A,ncol=2)
> B
array([[1, 2],
       [3, 4],
       [5, 6]])
Note that when A cannot be reshaped into a 2D array, we tend to discard elements which are at the end of A.
Does numpy have a function that works like my made-up function "vec2matrix"? (I understand that you can index a 1D array like a 2D array, but that isn't an option in the code I have - I need to make this conversion.)
A:
<code>
import numpy as np
A = np.array([1,2,3,4,5,6,7])
ncol = 2
</code>
B = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import pandas as pd
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            A = np.array([1, 2, 3, 4, 5, 6, 7])
            ncol = 2
        elif test_case_id == 2:
            np.random.seed(42)
            A = np.random.rand(23)
            ncol = 5
        return A, ncol

    def generate_ans(data):
        _a = data
        A, ncol = _a
        col = (A.shape[0] // ncol) * ncol
        B = A[:col]
        B = np.reshape(B, (-1, ncol))
        return B

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
A, ncol = test_input
[insert]
result = B
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
