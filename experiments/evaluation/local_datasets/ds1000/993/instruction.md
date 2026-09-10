Problem:

I have a trained PyTorch model and I want to get the confidence score of predictions in range (0-1). The code below is giving me a score but its range is undefined. I want the score in a defined range of (0-1) using softmax. Any idea how to get this?

conf, classes = torch.max(output.reshape(1, 3), 1)
My code:

MyNet.load_state_dict(torch.load("my_model.pt"))
def predict_allCharacters(input):
    output = MyNet(input)
    conf, classes = torch.max(output.reshape(1, 3), 1)
    class_names = '012'
    return conf, class_names[classes.item()]

Model definition:

MyNet = torch.nn.Sequential(torch.nn.Linear(4, 15),
                            torch.nn.Sigmoid(),
                            torch.nn.Linear(15, 3),
                            )

A:

runnable code
<code>
import numpy as np
import pandas as pd
import torch
MyNet = torch.nn.Sequential(torch.nn.Linear(4, 15),
                            torch.nn.Sigmoid(),
                            torch.nn.Linear(15, 3),
                            )
MyNet.load_state_dict(torch.load("my_model.pt"))
input = load_data()
assert type(input) == torch.Tensor
</code>
confidence_score = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import torch
import copy
import sklearn
from sklearn.datasets import load_iris


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            X, y = load_iris(return_X_y=True)
            input = torch.from_numpy(X[42]).float()
            torch.manual_seed(42)
            MyNet = torch.nn.Sequential(
                torch.nn.Linear(4, 15),
                torch.nn.Sigmoid(),
                torch.nn.Linear(15, 3),
            )
            torch.save(MyNet.state_dict(), "my_model.pt")
        return input

    def generate_ans(data):
        input = data
        MyNet = torch.nn.Sequential(
            torch.nn.Linear(4, 15),
            torch.nn.Sigmoid(),
            torch.nn.Linear(15, 3),
        )
        MyNet.load_state_dict(torch.load("my_model.pt"))
        output = MyNet(input)
        probs = torch.nn.functional.softmax(output.reshape(1, 3), dim=1)
        confidence_score, classes = torch.max(probs, 1)
        return confidence_score

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
MyNet = torch.nn.Sequential(torch.nn.Linear(4, 15),
                            torch.nn.Sigmoid(),
                            torch.nn.Linear(15, 3),
                            )
MyNet.load_state_dict(torch.load("my_model.pt"))
input = test_input
[insert]
result = confidence_score
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
