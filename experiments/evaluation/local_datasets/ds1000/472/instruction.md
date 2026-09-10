Problem:
For example, if I have a 2D array X, I can do slicing X[-1:, :]; if I have a 3D array Y, then I can do similar slicing for the first dimension like Y[-1:, :, :].
What is the right way to do the slicing when given an array `a` of unknown dimension?
Thanks!
A:
<code>
import numpy as np
a = np.random.rand(*np.random.randint(2, 10, (np.random.randint(2, 10))))
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
            np.random.seed(42)
            a = np.random.rand(*np.random.randint(2, 10, 4))
        elif test_case_id == 2:
            np.random.seed(43)
            a = np.random.rand(*np.random.randint(2, 10, 6))
        return a

    def generate_ans(data):
        _a = data
        a = _a
        result = a[-1:, ...]
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
a = test_input
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
