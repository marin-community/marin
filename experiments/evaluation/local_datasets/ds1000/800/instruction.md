Problem:
Basically, I am just trying to do a simple matrix multiplication, specifically, extract each column of it and normalize it by dividing it with its length.
    #csr sparse matrix
    self.__WeightMatrix__ = self.__WeightMatrix__.tocsr()
    #iterate through columns
    for Col in xrange(self.__WeightMatrix__.shape[1]):
       Column = self.__WeightMatrix__[:,Col].data
       List = [x**2 for x in Column]
       #get the column length
       Len = math.sqrt(sum(List))
       #here I assumed dot(number,Column) would do a basic scalar product
       dot((1/Len),Column)
       #now what? how do I update the original column of the matrix, everything that have been returned are copies, which drove me nuts and missed pointers so much
I've searched through the scipy sparse matrix documentations and got no useful information. I was hoping for a function to return a pointer/reference to the matrix so that I can directly modify its value. Thanks
A:
<code>
from scipy import sparse
import numpy as np
import math
sa = sparse.random(10, 10, density = 0.3, format = 'csr', random_state = 42)

</code>
sa = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
from scipy import sparse


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            sa = sparse.random(10, 10, density=0.3, format="csr", random_state=42)
        return sa

    def generate_ans(data):
        _a = data
        sa = _a
        sa = sparse.csr_matrix(
            sa.toarray() / np.sqrt(np.sum(sa.toarray() ** 2, axis=0))
        )
        return sa

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert type(result) == sparse.csr.csr_matrix
    assert len(sparse.find(result != ans)[0]) == 0
    return 1


exec_context = r"""
from scipy import sparse
import numpy as np
import math
sa = test_input
[insert]
result = sa
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
