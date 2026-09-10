Problem:
I'm searching for examples of using scipy.optimize.line_search. I do not really understand how this function works with multivariable functions. I wrote a simple example
import scipy as sp
import scipy.optimize
def test_func(x):
    return (x[0])**2+(x[1])**2

def test_grad(x):
    return [2*x[0],2*x[1]]

sp.optimize.line_search(test_func,test_grad,[1.8,1.7],[-1.0,-1.0])
And I've got
File "D:\Anaconda2\lib\site-packages\scipy\optimize\linesearch.py", line 259, in phi
return f(xk + alpha * pk, *args)
TypeError: can't multiply sequence by non-int of type 'float'
The result should be the alpha value of line_search
A:
<code>
import scipy
import scipy.optimize
import numpy as np
def test_func(x):
    return (x[0])**2+(x[1])**2

def test_grad(x):
    return [2*x[0],2*x[1]]
starting_point = [1.8, 1.7]
direction = [-1, -1]
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
            starting_point = [1.8, 1.7]
            direction = [-1, -1]
        elif test_case_id == 2:
            starting_point = [50, 37]
            direction = [-1, -1]
        return starting_point, direction

    def generate_ans(data):
        _a = data

        def test_func(x):
            return (x[0]) ** 2 + (x[1]) ** 2

        def test_grad(x):
            return [2 * x[0], 2 * x[1]]

        starting_point, direction = _a
        result = scipy.optimize.line_search(
            test_func, test_grad, np.array(starting_point), np.array(direction)
        )[0]
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert abs(result - ans) <= 1e-5
    return 1


exec_context = r"""
import scipy
import scipy.optimize
import numpy as np
def test_func(x):
    return (x[0])**2+(x[1])**2
def test_grad(x):
    return [2*x[0],2*x[1]]
starting_point, direction = test_input
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
