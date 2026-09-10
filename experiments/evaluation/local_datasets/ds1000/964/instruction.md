Problem:

I have this code:

import torch

list_of_tensors = [ torch.randn(3), torch.randn(3), torch.randn(3)]
tensor_of_tensors = torch.tensor(list_of_tensors)
I am getting the error:

ValueError: only one element tensors can be converted to Python scalars

How can I convert the list of tensors to a tensor of tensors in pytorch?


A:

<code>
import numpy as np
import pandas as pd
import torch
list_of_tensors = load_data()
</code>
tensor_of_tensors = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import torch
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        torch.random.manual_seed(42)
        if test_case_id == 1:
            list_of_tensors = [torch.randn(3), torch.randn(3), torch.randn(3)]
        return list_of_tensors

    def generate_ans(data):
        list_of_tensors = data
        tensor_of_tensors = torch.stack((list_of_tensors))
        return tensor_of_tensors

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
list_of_tensors = test_input
[insert]
result = tensor_of_tensors
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
