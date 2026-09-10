Problem:
How do I convert a numpy array to pytorch tensor?
A:
<code>
import torch
import numpy as np
a = np.ones(5)
</code>
a_pt = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import pandas as pd
import torch
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.ones(5)
        elif test_case_id == 2:
            a = np.array([1, 1, 4, 5, 1, 4])
        return a

    def generate_ans(data):
        _a = data
        a = _a
        a_pt = torch.Tensor(a)
        return a_pt

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert type(result) == torch.Tensor
    torch.testing.assert_close(result, ans, check_dtype=False)
    return 1


exec_context = r"""
import torch
import numpy as np
a = test_input
[insert]
result = a_pt
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
