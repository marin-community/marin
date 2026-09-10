Problem:

I have a logistic regression model using Pytorch, where my input is high-dimensional and my output must be a scalar - 0, 1 or 2.

I'm using a linear layer combined with a softmax layer to return a n x 3 tensor, where each column represents the probability of the input falling in one of the three classes (0, 1 or 2).

However, I must return a 1 x n tensor, and I want to somehow pick the lowest probability for each input and create a tensor indicating which class had the lowest probability. How can I achieve this using Pytorch?

To illustrate, my Softmax outputs this:

[[0.2, 0.1, 0.7],
 [0.6, 0.3, 0.1],
 [0.15, 0.8, 0.05]]
And I must return this:

[1, 2, 2], which has the type torch.LongTensor


A:

<code>
import numpy as np
import pandas as pd
import torch
softmax_output = load_data()
def solve(softmax_output):
</code>
y = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import torch
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            softmax_output = torch.FloatTensor(
                [[0.2, 0.1, 0.7], [0.6, 0.1, 0.3], [0.4, 0.5, 0.1]]
            )
        elif test_case_id == 2:
            softmax_output = torch.FloatTensor(
                [[0.7, 0.2, 0.1], [0.3, 0.6, 0.1], [0.05, 0.15, 0.8], [0.25, 0.35, 0.4]]
            )
        return softmax_output

    def generate_ans(data):
        softmax_output = data
        y = torch.argmin(softmax_output, dim=1).detach()
        return y

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        assert result.type() == "torch.LongTensor"
        torch.testing.assert_close(result, ans)
        return 1
    except:
        return 0


exec_context = r"""
import numpy as np
import pandas as pd
import torch
softmax_output = test_input
def solve(softmax_output):
[insert]
    return y
y = solve(softmax_output)
result = y
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
