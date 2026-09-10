Problem:

How to batch convert sentence lengths to masks in PyTorch?
For example, from

lens = [3, 5, 4]
we want to get

mask = [[1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0]]
Both of which are torch.LongTensors.


A:

<code>
import numpy as np
import pandas as pd
import torch
lens = load_data()
</code>
mask = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import torch
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            lens = torch.LongTensor([3, 5, 4])
        elif test_case_id == 2:
            lens = torch.LongTensor([3, 2, 4, 6, 5])
        return lens

    def generate_ans(data):
        lens = data
        max_len = max(lens)
        mask = torch.arange(max_len).expand(len(lens), max_len) < lens.unsqueeze(1)
        mask = mask.type(torch.LongTensor)
        return mask

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        torch.testing.assert_close(result, ans)
        return 1
    except:
        return 0


exec_context = r"""
import numpy as np
import pandas as pd
import torch
lens = test_input
[insert]
result = mask
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
