Problem:
I want to figure out how to replace nan values from my array with np.inf. 
For example, My array looks something like this:
x = [1400, 1500, 1600, nan, nan, nan ,1700] #Not in this exact configuration
How can I replace the nan values from x?
A:
<code>
import numpy as np
x = np.array([1400, 1500, 1600, np.nan, np.nan, np.nan ,1700])
</code>
x = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            x = np.array([1400, 1500, 1600, np.nan, np.nan, np.nan, 1700])
        elif test_case_id == 2:
            np.random.seed(42)
            x = np.random.rand(20)
            x[np.random.randint(0, 20, 3)] = np.nan
        return x

    def generate_ans(data):
        _a = data
        x = _a
        x[np.isnan(x)] = np.inf
        return x

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
x = test_input
[insert]
result = x
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
