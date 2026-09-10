Problem:
I have only the summary statistics of sample 1 and sample 2, namely mean, variance, nobs(number of observations). I want to do a weighted (take n into account) two-tailed t-test.
Any help on how to get the p-value would be highly appreciated.
A:
<code>
import numpy as np
import scipy.stats
amean = -0.0896
avar = 0.954
anobs = 40
bmean = 0.719
bvar = 11.87
bnobs = 50
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
            amean = -0.0896
            avar = 0.954
            anobs = 40
            bmean = 0.719
            bvar = 11.87
            bnobs = 50
        return amean, avar, anobs, bmean, bvar, bnobs

    def generate_ans(data):
        _a = data
        amean, avar, anobs, bmean, bvar, bnobs = _a
        _, p_value = scipy.stats.ttest_ind_from_stats(
            amean, np.sqrt(avar), anobs, bmean, np.sqrt(bvar), bnobs, equal_var=False
        )
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
amean, avar, anobs, bmean, bvar, bnobs = test_input
[insert]
result = p_value
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
