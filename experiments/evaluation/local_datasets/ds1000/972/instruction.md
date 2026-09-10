Problem:

I have the tensors:

ids: shape (30,1) containing indices like [[2],[1],[0],...]

x: shape(30,3,114)

ids tensor encodes the index of bold marked dimension of x which should be selected. I want to gather the selected slices in a resulting vector:

result: shape (30,114)

Background:

I have some scores (shape = (30,3)) for each of the 3 elements and want only to select the one with the highest score. Therefore, I used the function

ids = torch.argmax(scores,1,True)
giving me the maximum ids. I already tried to do it with gather function:

result = x.gather(1,ids)
but that didn't work.


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
            x = torch.arange(30 * 3 * 114).view(30, 3, 114)
            ids = torch.randint(0, 3, size=(30, 1))
        return ids, x

    def generate_ans(data):
        ids, x = data
        idx = ids.repeat(1, 114).view(30, 1, 114)
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
