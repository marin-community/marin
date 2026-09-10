Problem:
I have data of sample 1 and sample 2 (`a` and `b`) – size is different for sample 1 and sample 2. I want to do a weighted (take n into account) two-tailed t-test.
I tried using the scipy.stat module by creating my numbers with np.random.normal, since it only takes data and not stat values like mean and std dev (is there any way to use these values directly). But it didn't work since the data arrays has to be of equal size.
For some reason, nans might be in original data, and we want to omit them.
Any help on how to get the p-value would be highly appreciated.
A:
<code>
import numpy as np
import scipy.stats
a = np.random.randn(40)
b = 4*np.random.randn(50)
</code>
p_value = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
import scipy.stats


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(42)
            a = np.random.randn(40)
            b = 4 * np.random.randn(50)
        elif test_case_id == 2:
            np.random.seed(43)
            a = np.random.randn(40)
            b = 4 * np.random.randn(50)
            a[10] = np.nan
            b[20] = np.nan
        return a, b

    def generate_ans(data):
        _a = data
        a, b = _a
        _, p_value = scipy.stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        return p_value

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert abs(ans - result) <= 1e-5
    return 1


exec_context = r"""
import numpy as np
import scipy.stats
a, b = test_input
[insert]
result = p_value
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
