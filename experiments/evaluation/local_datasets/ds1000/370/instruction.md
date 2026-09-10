Problem:
In numpy, is there a nice idiomatic way of testing if all rows are equal in a 2d array?
I can do something like
np.all([np.array_equal(a[0], a[i]) for i in xrange(1,len(a))])
This seems to mix python lists with numpy arrays which is ugly and presumably also slow.
Is there a nicer/neater way?
A:
<code>
import numpy as np
example_a = np.repeat(np.arange(1, 6).reshape(1, -1), 3, axis = 0)
def f(a = example_a):
    # return the solution in this function
    # result = f(a)
    ### BEGIN SOLUTION

Execution context (contains [insert] placeholder):
import numpy as np
import pandas as pd
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.repeat(np.arange(1, 6).reshape(1, -1), 3, axis=0)
        elif test_case_id == 2:
            np.random.seed(42)
            a = np.random.rand(3, 4)
        elif test_case_id == 3:
            a = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        elif test_case_id == 4:
            a = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 3]])
        elif test_case_id == 5:
            a = np.array([[1, 1, 1], [2, 2, 1], [1, 1, 1]])
        return a

    def generate_ans(data):
        _a = data
        a = _a
        result = np.isclose(a, a[0], atol=0).all()
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert result == ans
    return 1


exec_context = r"""
import numpy as np
a = test_input
def f(a):
[insert]
result = f(a)
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(5):
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
