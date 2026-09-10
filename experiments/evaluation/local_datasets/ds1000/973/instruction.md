Problem:

I have the tensors:

ids: shape (70,3) containing indices like [[0,1,0],[1,0,0],[0,0,1],...]

x: shape(70,3,2)

ids tensor encodes the index of bold marked dimension of x which should be selected (1 means selected, 0 not). I want to gather the selected slices in a resulting vector:

result: shape (70,2)

Background:

I have some scores (shape = (70,3)) for each of the 3 elements and want only to select the one with the highest score.
Therefore, I made the index with the highest score to be 1, and rest indexes to be 0


A:

<code>
import numpy as np
import pandas as pd
import torch
ids, x = load_data()
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import torch
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        torch.random.manual_seed(42)
        if test_case_id == 1:
            x = torch.arange(70 * 3 * 2).view(70, 3, 2)
            select_ids = torch.randint(0, 3, size=(70, 1))
            ids = torch.zeros(size=(70, 3))
            for i in range(3):
                ids[i][select_ids[i]] = 1
        return ids, x

    def generate_ans(data):
        ids, x = data
        ids = torch.argmax(ids, 1, True)
        idx = ids.repeat(1, 2).view(70, 1, 2)
        result = torch.gather(x, 1, idx)
        result = result.squeeze(1)
        return result

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
ids, x = test_input
[insert]
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
