Problem:
Say, I have an array:
import numpy as np
a = np.array([0, 1, 2, 5, 6, 7, 8, 8, 8, 10, 29, 32, 45])
How can I calculate the 2nd standard deviation for it, so I could get the value of +2sigma ? Then I can get 2nd standard deviation interval, i.e., (μ-2σ, μ+2σ).
What I want is detecting outliers of 2nd standard deviation interval from array x. 
Hopefully result should be a bool array, True for outlier and False for not.
A:
<code>
import numpy as np
a = np.array([0, 1, 2, 5, 6, 7, 8, 8, 8, 10, 29, 32, 45])
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
            a = np.array([0, 1, 2, 5, 6, 7, 8, 8, 8, 10, 29, 32, 45])
        elif test_case_id == 2:
            np.random.seed(42)
            a = np.random.randn(30)
        elif test_case_id == 3:
            a = np.array([-1, -2, -10, 0, 1, 2, 2, 3])
        return a

    def generate_ans(data):
        _a = data
        a = _a
        interval = (a.mean() - 2 * a.std(), a.mean() + 2 * a.std())
        result = ~np.logical_and(a > interval[0], a < interval[1])
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
a = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(3):
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
