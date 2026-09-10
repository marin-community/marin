Problem:
I want to process a gray image in the form of np.array. 
*EDIT: chose a slightly more complex example to clarify
Suppose:
im = np.array([ [0,0,0,0,0,0] [0,0,5,1,2,0] [0,1,8,0,1,0] [0,0,0,7,1,0] [0,0,0,0,0,0]])
I'm trying to create this:
[ [0,5,1,2], [1,8,0,1], [0,0,7,1] ]
That is, to remove the peripheral zeros(black pixels) that fill an entire row/column.
In extreme cases, an image can be totally black, and I want the result to be an empty array.
I can brute force this with loops, but intuitively I feel like numpy has a better means of doing this.
A:
<code>
import numpy as np
im = np.array([[0,0,0,0,0,0],
               [0,0,5,1,2,0],
               [0,1,8,0,1,0],
               [0,0,0,7,1,0],
               [0,0,0,0,0,0]])
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
            im = np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 5, 1, 2, 0],
                    [0, 1, 8, 0, 1, 0],
                    [0, 0, 0, 7, 1, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            )
        elif test_case_id == 2:
            np.random.seed(42)
            im = np.random.randint(0, 10, (10, 12))
            im[:, 0] = 0
            im[-1, :] = 0
        elif test_case_id == 3:
            im = np.zeros((10, 10))
        return im

    def generate_ans(data):
        _a = data
        im = _a
        mask = im == 0
        rows = np.flatnonzero((~mask).sum(axis=1))
        cols = np.flatnonzero((~mask).sum(axis=0))
        if rows.shape[0] == 0:
            result = np.array([])
        else:
            result = im[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    if ans.shape[0]:
        np.testing.assert_array_equal(result, ans)
    else:
        ans = ans.reshape(0)
        result = result.reshape(0)
        np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
im = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(3):
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
