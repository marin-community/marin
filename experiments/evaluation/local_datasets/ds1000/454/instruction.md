Problem:
Given a 2-dimensional array in python, I would like to normalize each row with L∞ Norm.
I have started this code:
from numpy import linalg as LA
X = np.array([[1, 2, 3, 6],
              [4, 5, 6, 5],
              [1, 2, 5, 5],
              [4, 5,10,25],
              [5, 2,10,25]])
print X.shape
x = np.array([LA.norm(v,ord=np.inf) for v in X])
print x
Output:
   (5, 4)             # array dimension
   [6, 6, 5, 25, 25]   # L∞ on each Row
How can I have the rows of the matrix L∞-normalized without using LOOPS?
A:
<code>
from numpy import linalg as LA
import numpy as np
X = np.array([[1, -2, 3, 6],
              [4, 5, -6, 5],
              [-1, 2, 5, 5],
              [4, 5,10,-25],
              [5, -2,10,25]])
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            X = np.array(
                [
                    [1, -2, 3, 6],
                    [4, 5, -6, 5],
                    [-1, 2, 5, 5],
                    [4, 5, 10, -25],
                    [5, -2, 10, 25],
                ]
            )
        elif test_case_id == 2:
            X = np.array(
                [
                    [-1, -2, 3, 6],
                    [4, -5, -6, 5],
                    [-1, 2, -5, 5],
                    [4, -5, 10, -25],
                    [5, -2, 10, -25],
                ]
            )
        return X

    def generate_ans(data):
        _a = data
        X = _a
        linf = np.abs(X).max(axis=1)
        result = X / linf.reshape(-1, 1)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert np.allclose(result, ans)
    return 1


exec_context = r"""
from numpy import linalg as LA
import numpy as np
X = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)


def test_string(solution: str):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(solution.encode("utf-8")).readline):
        tokens.append(token.string)
    assert "while" not in tokens and "for" not in tokens

Return only the Python code snippet that replaces [insert]:
- No explanations, Markdown, comments, prints, or extra text; no non-English output
- No imports, function/class definitions, or placeholders; only the missing logic
- Do not reassign/reset provided variables/inputs; use them to compute the result
- The snippet must be syntactically valid and run as-is in this context; deterministic (assume temperature=0)
- Do not read/write files or call external services
- Output only the code lines for [insert]; no Markdown fences, no explanations, no bullets.
