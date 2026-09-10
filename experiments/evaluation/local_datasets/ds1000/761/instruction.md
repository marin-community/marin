Problem:
How to calculate kurtosis (the fourth standardized moment, according to Pearson’s definition) without bias correction?
I have tried scipy.stats.kurtosis, but it gives a different result. I followed the definition in mathworld.
A:
<code>
import numpy as np
a = np.array([   1. ,    2. ,    2.5,  400. ,    6. ,    0. ])
</code>
kurtosis_result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.array([1.0, 2.0, 2.5, 400.0, 6.0, 0.0])
        elif test_case_id == 2:
            np.random.seed(42)
            a = np.random.randn(10)
        return a

    def generate_ans(data):
        _a = data
        a = _a
        kurtosis_result = (sum((a - np.mean(a)) ** 4) / len(a)) / np.std(a) ** 4
        return kurtosis_result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert np.allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
a = test_input
[insert]
result = kurtosis_result
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
