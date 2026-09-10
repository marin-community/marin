Problem:
I'd like to calculate element-wise maximum of numpy ndarrays. For example
In [56]: a = np.array([10, 20, 30])
In [57]: b = np.array([30, 20, 20])
In [58]: c = np.array([50, 20, 40])
What I want:
[50, 20, 40]
A:
<code>
import numpy as np
a = np.array([10, 20, 30])
b = np.array([30, 20, 20])
c = np.array([50, 20, 40])
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
            a = np.array([10, 20, 30])
            b = np.array([30, 20, 20])
            c = np.array([50, 20, 40])
        elif test_case_id == 2:
            np.random.seed(42)
            a = np.random.rand(50)
            b = np.random.rand(50)
            c = np.random.rand(50)
        return a, b, c

    def generate_ans(data):
        _a = data
        a, b, c = _a
        result = np.max([a, b, c], axis=0)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
a, b, c = test_input
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
