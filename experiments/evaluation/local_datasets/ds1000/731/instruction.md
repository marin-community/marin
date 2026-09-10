Problem:
I have two csr_matrix, c1, c2.

I want a new matrix Feature = [c1, c2]. But if I directly concatenate them horizontally this way, there's an error that says the matrix Feature is a list. How can I achieve the matrix concatenation and still get the same type of matrix, i.e. a csr_matrix?

And it doesn't work if I do this after the concatenation: Feature = csr_matrix(Feature) It gives the error:

Traceback (most recent call last):
  File "yelpfilter.py", line 91, in <module>
    Feature = csr_matrix(Feature)
  File "c:\python27\lib\site-packages\scipy\sparse\compressed.py", line 66, in __init__
    self._set_self( self.__class__(coo_matrix(arg1, dtype=dtype)) )
  File "c:\python27\lib\site-packages\scipy\sparse\coo.py", line 185, in __init__
    self.row, self.col = M.nonzero()
TypeError: __nonzero__ should return bool or int, returned numpy.bool_

A:
<code>
from scipy import sparse
c1 = sparse.csr_matrix([[0, 0, 1, 0], [2, 0, 0, 0], [0, 0, 0, 0]])
c2 = sparse.csr_matrix([[0, 3, 4, 0], [0, 0, 0, 5], [6, 7, 0, 8]])
</code>
Feature = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import copy
from scipy import sparse


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            c1 = sparse.csr_matrix([[0, 0, 1, 0], [2, 0, 0, 0], [0, 0, 0, 0]])
            c2 = sparse.csr_matrix([[0, 3, 4, 0], [0, 0, 0, 5], [6, 7, 0, 8]])
        return c1, c2

    def generate_ans(data):
        _a = data
        c1, c2 = _a
        Feature = sparse.hstack((c1, c2)).tocsr()
        return Feature

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert type(result) == sparse.csr_matrix
    assert len(sparse.find(ans != result)[0]) == 0
    return 1


exec_context = r"""
from scipy import sparse
c1, c2 = test_input
[insert]
result = Feature
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
