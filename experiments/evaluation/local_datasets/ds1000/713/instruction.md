Problem:
I have a set of data and I want to compare which line describes it best (polynomials of different orders, exponential or logarithmic).
I use Python and Numpy and for polynomial fitting there is a function polyfit(). But I found no such functions for exponential and logarithmic fitting.
How do I fit y = A*exp(Bx) + C ? The result should be an np.array of [A, B, C]. I know that polyfit performs bad for this function, so I would like to use curve_fit to solve the problem, and it should start from initial guess p0.
A:
<code>
import numpy as np
import scipy.optimize
y = np.array([1, 7, 20, 50, 79])
x = np.array([10, 19, 30, 35, 51])
p0 = (4, 0.1, 1)
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
import scipy.optimize


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            y = np.array([1, 7, 20, 50, 79])
            x = np.array([10, 19, 30, 35, 51])
            p0 = (4, 0.1, 1)
        return x, y, p0

    def generate_ans(data):
        _a = data
        x, y, p0 = _a
        result = scipy.optimize.curve_fit(
            lambda t, a, b, c: a * np.exp(b * t) + c, x, y, p0=p0
        )[0]
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert np.allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
import scipy.optimize
x, y, p0 = test_input
[insert]
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
