Problem:

I have the following torch tensor:

tensor([[-0.2,  0.3],
    [-0.5,  0.1],
    [-0.4,  0.2]])
and the following numpy array: (I can convert it to something else if necessary)

[1 0 1]
I want to get the following tensor:

tensor([-0.2, 0.1, -0.4])
i.e. I want the numpy array to index each sub-element of my tensor (note the detail here, 0 means to select index 1, and 1 means to select index 0). Preferably without using a loop.

Thanks in advance


A:

<code>
import numpy as np
import pandas as pd
import torch
t, idx = load_data()
assert type(t) == torch.Tensor
assert type(idx) == np.ndarray
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import torch
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            t = torch.tensor([[-0.2, 0.3], [-0.5, 0.1], [-0.4, 0.2]])
            idx = np.array([1, 0, 1], dtype=np.int32)
        elif test_case_id == 2:
            t = torch.tensor([[-0.2, 0.3], [-0.5, 0.1], [-0.4, 0.2]])
            idx = np.array([1, 1, 0], dtype=np.int32)
        return t, idx

    def generate_ans(data):
        t, idx = data
        idx = 1 - idx
        idxs = torch.from_numpy(idx).long().unsqueeze(1)
        result = t.gather(1, idxs).squeeze(1)
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
t, idx = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)


def test_string(solution: str):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(solution.encode("utf-8")).readline):
        tokens.append(token.string)
    assert "for" not in tokens and "while" not in tokens

Return only the Python code snippet that replaces [insert]:
- No explanations, Markdown, comments, prints, or extra text; no non-English output
- No imports, function/class definitions, or placeholders; only the missing logic
- Do not reassign/reset provided variables/inputs; use them to compute the result
- The snippet must be syntactically valid and run as-is in this context; deterministic (assume temperature=0)
- Do not read/write files or call external services
- Output only the code lines for [insert]; no Markdown fences, no explanations, no bullets.
