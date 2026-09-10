Problem:

Consider I have 2D Tensor, index_in_batch * diag_ele. How can I get a 3D Tensor index_in_batch * Matrix (who is a diagonal matrix, construct by drag_ele)?

The torch.diag() construct diagonal matrix only when input is 1D, and return diagonal element when input is 2D.


A:

<code>
import numpy as np
import pandas as pd
import torch
Tensor_2D = load_data()
</code>
Tensor_3D = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import torch
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        torch.random.manual_seed(42)
        if test_case_id == 1:
            a = torch.rand(2, 3)
        elif test_case_id == 2:
            a = torch.rand(4, 5)
        return a

    def generate_ans(data):
        a = data
        Tensor_3D = torch.diag_embed(a)
        return Tensor_3D

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
Tensor_2D = test_input
[insert]
result = Tensor_3D
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
