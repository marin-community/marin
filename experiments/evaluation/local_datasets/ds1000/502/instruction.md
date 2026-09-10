Problem:
I have two arrays:
•	a: a 3-dimensional source array (N x M x 2)
•	b: a 2-dimensional index array (N x M) containing 0 and 1s.
I want to use the indices in b to select the corresponding elements of a in its third dimension. The resulting array should have the dimensions N x M. Here is the example as code:
import numpy as np
a = np.array( # dims: 3x3x2
    [[[ 0,  1],
     [ 2,  3],
     [ 4,  5]],
    [[ 6,  7],
     [ 8,  9],
     [10, 11]],
    [[12, 13],
     [14, 15],
     [16, 17]]]
)
b = np.array( # dims: 3x3
    [[1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]]
)
# select the elements in a according to b
# to achieve this result:
desired = np.array(
  [[ 1,  3,  5],
   [ 7,  9, 11],
   [13, 15, 17]]
)

At first, I thought this must have a simple solution but I could not find one at all. Since I would like to port it to tensorflow, I would appreciate if somebody knows a numpy-type solution for this.
A:
<code>
import numpy as np
a = np.array( # dims: 3x3x2
    [[[ 0,  1],
     [ 2,  3],
     [ 4,  5]],
    [[ 6,  7],
     [ 8,  9],
     [10, 11]],
    [[12, 13],
     [14, 15],
     [16, 17]]]
)
b = np.array( # dims: 3x3
    [[1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]]
)
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
            a = np.array(
                [
                    [[0, 1], [2, 3], [4, 5]],
                    [[6, 7], [8, 9], [10, 11]],
                    [[12, 13], [14, 15], [16, 17]],
                ]
            )
            b = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        elif test_case_id == 2:
            np.random.seed(42)
            dim = np.random.randint(10, 15)
            a = np.random.rand(dim, dim, 2)
            b = np.zeros((dim, dim)).astype(int)
            b[[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]] = 1
        return a, b

    def generate_ans(data):
        _a = data
        a, b = _a
        result = np.take_along_axis(a, b[..., np.newaxis], axis=-1)[..., 0]
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
a, b = test_input
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
