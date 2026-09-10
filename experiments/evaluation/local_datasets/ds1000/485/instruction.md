Problem:
I'm trying the following:
Given a matrix A (x, y ,3) and another matrix B (3, 3), I would like to return a (x, y, 3) matrix in which the 3rd dimension of A multiplies the values of B (similar when an RGB image is transformed into gray, only that those "RGB" values are multiplied by a matrix and not scalars)...
Here's what I've tried:
np.multiply(B, A)
np.einsum('ijk,jl->ilk', B, A)
np.einsum('ijk,jl->ilk', A, B)
All of them failed with dimensions not aligned.
What am I missing?
A:
<code>
import numpy as np
A = np.random.rand(5, 6, 3)
B = np.random.rand(3, 3)
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
            np.random.seed(42)
            A = np.random.rand(5, 6, 3)
            B = np.random.rand(3, 3)
        return A, B

    def generate_ans(data):
        _a = data
        A, B = _a
        result = np.tensordot(A, B, axes=((2), (0)))
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
A, B = test_input
[insert]
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
