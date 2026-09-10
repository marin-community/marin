Problem:
I have two 2D numpy arrays like this, representing the x/y distances between three points. I need the x/y distances as tuples in a single array.
So from:
x_dists = array([[ 0, -1, -2],
                 [ 1,  0, -1],
                 [ 2,  1,  0]])
y_dists = array([[ 0, 1, -2],
                 [ -1,  0, 1],
                 [ -2,  1,  0]])
I need:
dists = array([[[ 0,  0], [-1, 1], [-2, -2]],
               [[ 1,  -1], [ 0,  0], [-1, 1]],
               [[ 2,  -2], [ 1,  1], [ 0,  0]]])
I've tried using various permutations of dstack/hstack/vstack/concatenate, but none of them seem to do what I want. The actual arrays in code are liable to be gigantic, so iterating over the elements in python and doing the rearrangement "manually" isn't an option speed-wise.
A:
<code>
import numpy as np
x_dists = np.array([[ 0, -1, -2],
                 [ 1,  0, -1],
                 [ 2,  1,  0]])

y_dists = np.array([[ 0, 1, -2],
                 [ -1,  0, 1],
                 [ -2,  1,  0]])
</code>
dists = ... # put solution in this variable
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
            x_dists = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]])
            y_dists = np.array([[0, 1, -2], [-1, 0, 1], [-2, 1, 0]])
        elif test_case_id == 2:
            np.random.seed(42)
            x_dists = np.random.rand(3, 4)
            y_dists = np.random.rand(3, 4)
        return x_dists, y_dists

    def generate_ans(data):
        _a = data
        x_dists, y_dists = _a
        dists = np.vstack(([x_dists.T], [y_dists.T])).T
        return dists

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
x_dists, y_dists = test_input
[insert]
result = dists
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
