Problem:
I am using Python with numpy to do linear algebra.
I performed numpy SVD on a matrix `a` to get the matrices U,i, and V. However the i matrix is expressed as a 1x4 matrix with 1 row. i.e.: [ 12.22151125 4.92815942 2.06380839 0.29766152].
How can I get numpy to express the i matrix as a diagonal matrix like so: [[12.22151125, 0, 0, 0],[0,4.92815942, 0, 0],[0,0,2.06380839,0 ],[0,0,0,0.29766152]]
Code I am using:
a = np.matrix([[3, 4, 3, 1],[1,3,2,6],[2,4,1,5],[3,3,5,2]])
U, i, V = np.linalg.svd(a,full_matrices=True)
So I want i to be a full diagonal matrix. How an I do this?
A:
<code>
import numpy as np
a = np.matrix([[3, 4, 3, 1],[1,3,2,6],[2,4,1,5],[3,3,5,2]])
U, i, V = np.linalg.svd(a,full_matrices=True)
</code>
i = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import pandas as pd
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.matrix([[3, 4, 3, 1], [1, 3, 2, 6], [2, 4, 1, 5], [3, 3, 5, 2]])
        return a

    def generate_ans(data):
        _a = data
        a = _a
        U, i, V = np.linalg.svd(a, full_matrices=True)
        i = np.diag(i)
        return i

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
a = test_input
U, i, V = np.linalg.svd(a,full_matrices=True)
[insert]
result = i
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
