Problem:
How does one convert a list of Z-scores from the Z-distribution (standard normal distribution, Gaussian distribution) to left-tailed p-values? I have yet to find the magical function in Scipy's stats module to do this, but one must be there.
A:
<code>
import numpy as np
import scipy.stats
z_scores = np.array([-3, -2, 0, 2, 2.5])
</code>
p_values = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
import scipy.stats


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.array([-3, -2, 0, 2, 2.5])
        return a

    def generate_ans(data):
        _a = data
        z_scores = _a
        temp = np.array(z_scores)
        p_values = scipy.stats.norm.cdf(temp)
        return p_values

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
import scipy.stats
z_scores = test_input
[insert]
result = p_values
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
