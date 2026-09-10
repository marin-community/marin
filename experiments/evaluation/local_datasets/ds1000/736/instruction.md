Problem:
I want to remove diagonal elements from a sparse matrix. Since the matrix is sparse, these elements shouldn't be stored once removed.
Scipy provides a method to set diagonal elements values: setdiag
If I try it using lil_matrix, it works:
>>> a = np.ones((2,2))
>>> c = lil_matrix(a)
>>> c.setdiag(0)
>>> c
<2x2 sparse matrix of type '<type 'numpy.float64'>'
    with 2 stored elements in LInked List format>
However with csr_matrix, it seems diagonal elements are not removed from storage:
>>> b = csr_matrix(a)
>>> b
<2x2 sparse matrix of type '<type 'numpy.float64'>'
    with 4 stored elements in Compressed Sparse Row format>

>>> b.setdiag(0)
>>> b
<2x2 sparse matrix of type '<type 'numpy.float64'>'
    with 4 stored elements in Compressed Sparse Row format>

>>> b.toarray()
array([[ 0.,  1.],
       [ 1.,  0.]])
Through a dense array, we have of course:
>>> csr_matrix(b.toarray())
<2x2 sparse matrix of type '<type 'numpy.float64'>'
    with 2 stored elements in Compressed Sparse Row format>
Is that intended? If so, is it due to the compressed format of csr matrices? Is there any workaround else than going from sparse to dense to sparse again?
A:
<code>
from scipy import sparse
import numpy as np
a = np.ones((2, 2))
b = sparse.csr_matrix(a)
</code>
b = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
import tokenize, io
from scipy import sparse


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.ones((2, 2))
            b = sparse.csr_matrix(a)
        elif test_case_id == 2:
            a = []
            b = sparse.csr_matrix([])
        return a, b

    def generate_ans(data):
        _a = data
        a, b = _a
        b = sparse.csr_matrix(a)
        b.setdiag(0)
        b.eliminate_zeros()
        return b

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert type(result) == type(ans)
    assert len(sparse.find(result != ans)[0]) == 0
    assert result.nnz == ans.nnz
    return 1


exec_context = r"""
from scipy import sparse
import numpy as np
a, b = test_input
[insert]
result = b
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
