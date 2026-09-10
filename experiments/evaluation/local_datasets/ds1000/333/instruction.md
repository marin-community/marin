Problem:
Does Python have a function to reduce fractions?
For example, when I calculate 98/42 I want to get 7/3, not 2.3333333, is there a function for that using Python or Numpy?
The result should be a tuple, namely (7, 3), the first for numerator and the second for denominator.
IF the dominator is zero, result should be (NaN, NaN)
A:
<code>
import numpy as np
numerator = 98
denominator = 42
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            numerator = 98
            denominator = 42
        elif test_case_id == 2:
            np.random.seed(42)
            numerator = np.random.randint(2, 10)
            denominator = np.random.randint(2, 10)
        elif test_case_id == 3:
            numerator = -5
            denominator = 10
        elif test_case_id == 4:
            numerator = 5
            denominator = 0
        return numerator, denominator

    def generate_ans(data):
        _a = data
        numerator, denominator = _a
        if denominator == 0:
            result = (np.nan, np.nan)
        else:
            gcd = np.gcd(numerator, denominator)
            result = (numerator // gcd, denominator // gcd)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    if np.isnan(ans[0]):
        assert np.isnan(result[0]) and np.isnan(result[1])
    else:
        assert result[0] == ans[0] and result[1] == ans[1]
    return 1


exec_context = r"""
import numpy as np
numerator, denominator = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(4):
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
