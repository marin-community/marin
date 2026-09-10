Problem:
I could not find a built-in function in Python to generate a log uniform distribution given a min and max value (the R equivalent is here), something like: loguni[n, min, max, base] that returns n log uniformly distributed in the range min and max.
The closest I found though was numpy.random.uniform.
That is, given range of x, I want to get samples of given size (n) that suit log-uniform distribution. 
Any help would be appreciated!
A:
<code>
import numpy as np

min = 1
max = np.e
n = 10000
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
import tokenize, io
import scipy
from scipy.stats import ks_2samp


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            low = 1
            high = np.e
            size = 10000
        return low, high, size

    def generate_ans(data):
        _a = data
        min, max, n = _a
        result = scipy.stats.loguniform.rvs(a=min, b=max, size=n)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))

    return test_input, expected_result


def exec_test(result, ans):

    np.testing.assert_array_equal(result.shape, ans.shape)
    assert ks_2samp(result, ans)[0] <= 0.1

    return 1


exec_context = r"""
import numpy as np
min, max, n = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)


def test_string(solution: str):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(solution.encode("utf-8")).readline):
        tokens.append(token.string)
    assert "while" not in tokens and "for" not in tokens

Return only the Python code snippet that replaces [insert]:
- No explanations, Markdown, comments, prints, or extra text; no non-English output
- No imports, function/class definitions, or placeholders; only the missing logic
- Do not reassign/reset provided variables/inputs; use them to compute the result
- The snippet must be syntactically valid and run as-is in this context; deterministic (assume temperature=0)
- Do not read/write files or call external services
- Output only the code lines for [insert]; no Markdown fences, no explanations, no bullets.
