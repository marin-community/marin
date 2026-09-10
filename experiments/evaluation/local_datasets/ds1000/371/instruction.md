Problem:
SciPy has three methods for doing 1D integrals over samples (trapz, simps, and romb) and one way to do a 2D integral over a function (dblquad), but it doesn't seem to have methods for doing a 2D integral over samples -- even ones on a rectangular grid.
The closest thing I see is scipy.interpolate.RectBivariateSpline.integral -- you can create a RectBivariateSpline from data on a rectangular grid and then integrate it. However, that isn't terribly fast.
I want something more accurate than the rectangle method (i.e. just summing everything up). I could, say, use a 2D Simpson's rule by making an array with the correct weights, multiplying that by the array I want to integrate, and then summing up the result.
However, I don't want to reinvent the wheel if there's already something better out there. Is there?
For instance, I want to do 2D integral over (cosx)^4 + (siny)^2, how can I do it? Perhaps using Simpson rule?
A:
<code>
import numpy as np
x = np.linspace(0, 1, 20)
y = np.linspace(0, 1, 30)
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
import scipy
from scipy.integrate import simpson


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            x = np.linspace(0, 1, 20)
            y = np.linspace(0, 1, 30)
        elif test_case_id == 2:
            x = np.linspace(3, 5, 30)
            y = np.linspace(0, 1, 20)
        return x, y

    def generate_ans(data):
        _a = data
        x, y = _a
        z = np.cos(x[:, None]) ** 4 + np.sin(y) ** 2
        result = simpson(simpson(z, y), x)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))

    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
x, y = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
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
