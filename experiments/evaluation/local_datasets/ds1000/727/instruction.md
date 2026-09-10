Problem:
How can I extract the main diagonal(1-d array) of a sparse matrix? The matrix is created in scipy.sparse. I want equivalent of np.diagonal(), but for sparse matrix.

A:
<code>
import numpy as np
from scipy.sparse import csr_matrix

arr = np.random.rand(4, 4)
M = csr_matrix(arr)
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
from scipy.sparse import csr_matrix


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            arr = np.array(
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
            )
        elif test_case_id == 2:
            np.random.seed(42)
            arr = np.random.rand(6, 6)
        return arr

    def generate_ans(data):
        _a = data
        arr = _a
        M = csr_matrix(arr)
        result = M.A.diagonal(0)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
from scipy.sparse import csr_matrix
arr = test_input
M = csr_matrix(arr)
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
