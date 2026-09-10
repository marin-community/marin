Problem:
Scipy offers many useful tools for root finding, notably fsolve. Typically a program has the following form:
def eqn(x, a, b):
    return x + 2*a - b**2
fsolve(eqn, x0=0.5, args = (a,b))
and will find a root for eqn(x) = 0 given some arguments a and b.
However, what if I have a problem where I want to solve for the b variable, giving the function arguments in a and b? Of course, I could recast the initial equation as
def eqn(b, x, a)
but this seems long winded and inefficient. Instead, is there a way I can simply set fsolve (or another root finding algorithm) to allow me to choose which variable I want to solve for?
Note that the result should be an array of roots for many (x, a) pairs. The function might have two roots for each setting, and I want to put the smaller one first, like this:
result = [[2, 5],
          [-3, 4]] for two (x, a) pairs
A:
<code>
import numpy as np
from scipy.optimize import fsolve
def eqn(x, a, b):
    return x + 2*a - b**2

xdata = np.arange(4)+3
adata = np.random.randint(0, 10, (4,))
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
from scipy.optimize import fsolve


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(42)
            xdata = np.arange(4) + 3
            adata = np.random.randint(0, 10, (4,))
        return xdata, adata

    def generate_ans(data):
        _a = data

        def eqn(x, a, b):
            return x + 2 * a - b**2

        xdata, adata = _a
        A = np.array(
            [
                fsolve(lambda b, x, a: eqn(x, a, b), x0=0, args=(x, a))[0]
                for x, a in zip(xdata, adata)
            ]
        )
        temp = -A
        result = np.zeros((len(A), 2))
        result[:, 0] = A
        result[:, 1] = temp
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
from scipy.optimize import fsolve
def eqn(x, a, b):
    return x + 2*a - b**2
xdata, adata = test_input
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
