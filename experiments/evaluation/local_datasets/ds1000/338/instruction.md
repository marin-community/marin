Problem:
So in numpy arrays there is the built in function for getting the diagonal indices, but I can't seem to figure out how to get the diagonal starting from the top right rather than top left.
This is the normal code to get starting from the top left, assuming processing on 5x5 array:
>>> import numpy as np
>>> a = np.arange(25).reshape(5,5)
>>> diagonal = np.diag_indices(5)
>>> a
array([[ 0,  1,  2,  3,  4],
   [ 5,  6,  7,  8,  9],
   [10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19],
   [20, 21, 22, 23, 24]])
>>> a[diagonal]
array([ 0,  6, 12, 18, 24])

so what do I use if I want it to return:
array([[0, 6, 12, 18, 24] [4,  8, 12, 16, 20])
How to get that in a general way, That is, can be used on other arrays with different shape?
A:
<code>
import numpy as np
a = np.array([[ 0,  1,  2,  3,  4],
   [ 5,  6,  7,  8,  9],
   [10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19],
   [20, 21, 22, 23, 24]])
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import pandas as pd
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.array(
                [
                    [0, 1, 2, 3, 4],
                    [5, 6, 7, 8, 9],
                    [10, 11, 12, 13, 14],
                    [15, 16, 17, 18, 19],
                    [20, 21, 22, 23, 24],
                ]
            )
        elif test_case_id == 2:
            np.random.seed(42)
            a = np.random.rand(8, 8)
        return a

    def generate_ans(data):
        _a = data
        a = _a
        result = np.vstack((np.diag(a), np.diag(np.fliplr(a))))
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
a = test_input
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
