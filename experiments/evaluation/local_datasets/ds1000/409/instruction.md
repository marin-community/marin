Problem:
I have two input arrays x and y of the same shape. I need to run each of their elements with matching indices through a function, then store the result at those indices in a third array z. What is the most pythonic way to accomplish this? Right now I have four four loops - I'm sure there is an easier way.
x = [[2, 2, 2],
     [2, 2, 2],
     [2, 2, 2]]
y = [[3, 3, 3],
     [3, 3, 3],
     [3, 3, 1]]
def elementwise_function(element_1,element_2):
    return (element_1 + element_2)
z = [[5, 5, 5],
     [5, 5, 5],
     [5, 5, 3]]
I am getting confused since my function will only work on individual data pairs. I can't simply pass the x and y arrays to the function.
A:
<code>
import numpy as np
x = [[2, 2, 2],
     [2, 2, 2],
     [2, 2, 2]]
y = [[3, 3, 3],
     [3, 3, 3],
     [3, 3, 1]]
</code>
z = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            x = [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
            y = [[3, 3, 3], [3, 3, 3], [3, 3, 1]]
        elif test_case_id == 2:
            np.random.seed(42)
            dim1 = np.random.randint(5, 10)
            dim2 = np.random.randint(6, 10)
            x = np.random.rand(dim1, dim2)
            y = np.random.rand(dim1, dim2)
        return x, y

    def generate_ans(data):
        _a = data
        x, y = _a
        x_new = np.array(x)
        y_new = np.array(y)
        z = x_new + y_new
        return z

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
x, y = test_input
[insert]
result = z
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
