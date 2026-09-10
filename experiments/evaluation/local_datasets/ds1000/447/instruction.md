Problem:
What I am trying to achieve is a 'highest to lowest' ranking of a list of values, basically the reverse of rankdata
So instead of:
a = [1,2,3,4,3,2,3,4]
rankdata(a).astype(int)
array([1, 2, 5, 7, 5, 2, 5, 7])
I want to get this:
array([7, 6, 3, 1, 3, 6, 3, 1])
I wasn't able to find anything in the rankdata documentation to do this.
A:
<code>
import numpy as np
from scipy.stats import rankdata
example_a = [1,2,3,4,3,2,3,4]
def f(a = example_a):
    # return the solution in this function
    # result = f(a)
    ### BEGIN SOLUTION

Execution context (contains [insert] placeholder):
import numpy as np
import copy
from scipy.stats import rankdata


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = [1, 2, 3, 4, 3, 2, 3, 4]
        elif test_case_id == 2:
            np.random.seed(42)
            a = np.random.rand(np.random.randint(26, 30))
        return a

    def generate_ans(data):
        _a = data
        a = _a
        result = len(a) - rankdata(a).astype(int)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
from scipy.stats import rankdata
a = test_input
def f(a):
[insert]
result = f(a)
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
