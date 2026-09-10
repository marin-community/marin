Problem:
Suppose I have a hypotetical function I'd like to approximate:
def f(x):
    return a * x ** 2 + b * x + c
Where a, b and c are the values I don't know.
And I have certain points where the function output is known, i.e.
x = [-1, 2, 5, 100]
y = [123, 456, 789, 1255]
(actually there are way more values)
I'd like to get a, b and c while minimizing the squared error .
What is the way to do that in Python? The result should be an array like [a, b, c], from highest order to lowest order.
There should be existing solutions in numpy or anywhere like that.
A:
<code>
import numpy as np
x = [-1, 2, 5, 100]
y = [123, 456, 789, 1255]
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import pandas as pd
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            x = [-1, 2, 5, 100]
            y = [123, 456, 789, 1255]
        elif test_case_id == 2:
            np.random.seed(42)
            x = (np.random.rand(100) - 0.5) * 10
            y = (np.random.rand(100) - 0.5) * 10
        return x, y

    def generate_ans(data):
        _a = data
        x, y = _a
        result = np.polyfit(x, y, 2)
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
