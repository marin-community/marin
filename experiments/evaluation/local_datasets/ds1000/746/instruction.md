Problem:
I have a sparse 988x1 vector (stored in col, a column in a csr_matrix) created through scipy.sparse. Is there a way to gets its max and min value without having to convert the sparse matrix to a dense one?
numpy.max seems to only work for dense vectors.

A:
<code>
import numpy as np
from scipy.sparse import csr_matrix

np.random.seed(10)
arr = np.random.randint(4,size=(988,988))
sA = csr_matrix(arr)
col = sA.getcol(0)
</code>
Max, Min = ... # put solution in these variables
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
import tokenize, io
from scipy.sparse import csr_matrix


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(10)
            arr = np.random.randint(4, size=(988, 988))
            A = csr_matrix(arr)
            col = A.getcol(0)
        elif test_case_id == 2:
            np.random.seed(42)
            arr = np.random.randint(4, size=(988, 988))
            A = csr_matrix(arr)
            col = A.getcol(0)
        elif test_case_id == 3:
            np.random.seed(80)
            arr = np.random.randint(4, size=(988, 988))
            A = csr_matrix(arr)
            col = A.getcol(0)
        elif test_case_id == 4:
            np.random.seed(100)
            arr = np.random.randint(4, size=(988, 988))
            A = csr_matrix(arr)
            col = A.getcol(0)
        return col

    def generate_ans(data):
        _a = data
        col = _a
        Max, Min = col.max(), col.min()
        return [Max, Min]

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert np.allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
from scipy.sparse import csr_matrix
col = test_input
[insert]
result = [Max, Min]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(4):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)


def test_string(solution: str):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(solution.encode("utf-8")).readline):
        tokens.append(token.string)
    assert (
        "toarray" not in tokens
        and "array" not in tokens
        and "todense" not in tokens
        and "A" not in tokens
    )

Return only the Python code snippet that replaces [insert]:
- No explanations, Markdown, comments, prints, or extra text; no non-English output
- No imports, function/class definitions, or placeholders; only the missing logic
- Do not reassign/reset provided variables/inputs; use them to compute the result
- The snippet must be syntactically valid and run as-is in this context; deterministic (assume temperature=0)
- Do not read/write files or call external services
- Output only the code lines for [insert]; no Markdown fences, no explanations, no bullets.
