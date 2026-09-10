Problem:
I have problems using scipy.sparse.csr_matrix:
for instance:
a = csr_matrix([[1,2,3],[4,5,6]])
b = csr_matrix([[7,8,9],[10,11,12]])
how to merge them into
[[1,2,3,7,8,9],[4,5,6,10,11,12]]
I know a way is to transfer them into numpy array first:
csr_matrix(numpy.hstack((a.toarray(),b.toarray())))
but it won't work when the matrix is huge and sparse, because the memory would run out.
so are there any way to merge them together in csr_matrix?
any answers are appreciated!
A:
<code>
from scipy import sparse
sa = sparse.random(10, 10, density = 0.01, format = 'csr')
sb = sparse.random(10, 10, density = 0.01, format = 'csr')
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import copy
import tokenize, io
from scipy import sparse


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            sa = sparse.random(10, 10, density=0.01, format="csr", random_state=42)
            sb = sparse.random(10, 10, density=0.01, format="csr", random_state=45)
        return sa, sb

    def generate_ans(data):
        _a = data
        sa, sb = _a
        result = sparse.hstack((sa, sb)).tocsr()
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert type(result) == sparse.csr.csr_matrix
    assert len(sparse.find(result != ans)[0]) == 0
    return 1


exec_context = r"""
from scipy import sparse
sa, sb = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
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
