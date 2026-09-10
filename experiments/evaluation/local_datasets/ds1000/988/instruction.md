Problem:

This question may not be clear, so please ask for clarification in the comments and I will expand.

I have the following tensors of the following shape:

mask.size() == torch.Size([1, 400])
clean_input_spectrogram.size() == torch.Size([1, 400, 161])
output.size() == torch.Size([1, 400, 161])
mask is comprised only of 0 and 1. Since it's a mask, I want to set the elements of output equal to clean_input_spectrogram where that relevant mask value is 1.

How would I do that?


A:

<code>
import numpy as np
import pandas as pd
import torch
mask, clean_input_spectrogram, output= load_data()
</code>
output = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import torch
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            torch.random.manual_seed(42)
            mask = torch.tensor([[0, 1, 0]]).to(torch.int32)
            clean_input_spectrogram = torch.rand((1, 3, 2))
            output = torch.rand((1, 3, 2))
        return mask, clean_input_spectrogram, output

    def generate_ans(data):
        mask, clean_input_spectrogram, output = data
        output[:, mask[0].to(torch.bool), :] = clean_input_spectrogram[
            :, mask[0].to(torch.bool), :
        ]
        return output

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
mask, clean_input_spectrogram, output = test_input
[insert]
result = output
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
