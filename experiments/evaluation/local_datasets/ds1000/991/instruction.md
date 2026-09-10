Problem:

I may be missing something obvious, but I can't find a way to compute this.

Given two tensors, I want to keep elements with the maximum absolute values, in each one of them as well as the sign.

I thought about

sign_x = torch.sign(x)
sign_y = torch.sign(y)
max = torch.max(torch.abs(x), torch.abs(y))
in order to eventually multiply the signs with the obtained maximums, but then I have no method to multiply the correct sign to each element that was kept and must choose one of the two tensors.


A:

<code>
import numpy as np
import pandas as pd
import torch
x, y = load_data()
</code>
signed_max = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import torch
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            torch.random.manual_seed(42)
            x = torch.randint(-10, 10, (5,))
            y = torch.randint(-20, 20, (5,))
        return x, y

    def generate_ans(data):
        x, y = data
        maxs = torch.max(torch.abs(x), torch.abs(y))
        xSigns = (maxs == torch.abs(x)) * torch.sign(x)
        ySigns = (maxs == torch.abs(y)) * torch.sign(y)
        finalSigns = xSigns.int() | ySigns.int()
        signed_max = maxs * finalSigns
        return signed_max

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        torch.testing.assert_close(result, ans, check_dtype=False)
        return 1
    except:
        return 0


exec_context = r"""
import numpy as np
import pandas as pd
import torch
x, y = test_input
[insert]
result = signed_max
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
