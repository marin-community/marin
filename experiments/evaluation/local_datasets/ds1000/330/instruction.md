Problem:
I need to square a 2D numpy array (elementwise) and I have tried the following code:
import numpy as np
a = np.arange(4).reshape(2, 2)
print(a^2, '\n')
print(a*a)
that yields:
[[2 3]
[0 1]]
[[0 1]
[4 9]]
Clearly, the notation a*a gives me the result I want and not a^2.
I would like to know if another notation exists to raise a numpy array to power = 2 or power = N? Instead of a*a*a*..*a.
A:
<code>
import numpy as np
example_a = np.arange(4).reshape(2, 2)
def f(a = example_a, power = 5):
    # return the solution in this function
    # result = f(a, power)
    ### BEGIN SOLUTION

Execution context (contains [insert] placeholder):
import numpy as np
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.arange(4).reshape(2, 2)
            power = 5
        elif test_case_id == 2:
            np.random.seed(42)
            a = np.random.rand(np.random.randint(5, 10), np.random.randint(6, 10))
            power = np.random.randint(6, 10)
        return a, power

    def generate_ans(data):
        _a = data
        a, power = _a
        result = a**power
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert np.allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
a, power = test_input
def f(a, power):
[insert]
result = f(a, power)
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
