Problem:

How to convert a numpy array of dtype=object to torch Tensor?

array([
   array([0.5, 1.0, 2.0], dtype=float16),
   array([4.0, 6.0, 8.0], dtype=float16)
], dtype=object)


A:

<code>
import pandas as pd
import torch
import numpy as np
x_array = load_data()
def Convert(a):
    # return the solution in this function
    # t = Convert(a)
    ### BEGIN SOLUTION

Execution context (contains [insert] placeholder):
import numpy as np
import torch
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            x = np.array(
                [
                    np.array([0.5, 1.0, 2.0], dtype=np.float16),
                    np.array([4.0, 6.0, 8.0], dtype=np.float16),
                ],
                dtype=object,
            )
        elif test_case_id == 2:
            x = np.array(
                [
                    np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float16),
                    np.array([4.0, 6.0, 8.0, 9.0], dtype=np.float16),
                    np.array([4.0, 6.0, 8.0, 9.0], dtype=np.float16),
                ],
                dtype=object,
            )
        return x

    def generate_ans(data):
        x_array = data
        x_tensor = torch.from_numpy(x_array.astype(float))
        return x_tensor

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        assert torch.is_tensor(result)
        torch.testing.assert_close(result, ans, check_dtype=False)
        return 1
    except:
        return 0


exec_context = r"""
import numpy as np
import pandas as pd
import torch
x_array = test_input
def Convert(a):
[insert]
x_tensor = Convert(x_array)
result = x_tensor
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
