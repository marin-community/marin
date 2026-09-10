Problem:

Is it possible in PyTorch to change the learning rate of the optimizer in the middle of training dynamically (I don't want to define a learning rate schedule beforehand)?

So let's say I have an optimizer:

optim = torch.optim.SGD(..., lr=0.005)
Now due to some tests which I perform during training, I realize my learning rate is too high so I want to change it to say 0.0005. There doesn't seem to be a method optim.set_lr(0.0005) but is there some way to do this?


A:

<code>
import numpy as np
import pandas as pd
import torch
optim = load_data()
</code>
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import torch
import copy
from torch import nn


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:

            class MyAttentionBiLSTM(nn.Module):
                def __init__(self):
                    super(MyAttentionBiLSTM, self).__init__()
                    self.lstm = nn.LSTM(
                        input_size=20,
                        hidden_size=20,
                        num_layers=1,
                        batch_first=True,
                        bidirectional=True,
                    )
                    self.attentionW = nn.Parameter(torch.randn(5, 20 * 2))
                    self.softmax = nn.Softmax(dim=1)
                    self.linear = nn.Linear(20 * 2, 2)

            model = MyAttentionBiLSTM()
            optim = torch.optim.SGD(
                [{"params": model.lstm.parameters()}, {"params": model.attentionW}],
                lr=0.01,
            )
        return optim

    def generate_ans(data):
        optim = data
        for param_group in optim.param_groups:
            param_group["lr"] = 0.0005
        return optim

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        assert ans.defaults == result.defaults
        for param_group in result.param_groups:
            assert param_group["lr"] == 0.0005
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
import torch
optim = test_input
[insert]
result = optim
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
