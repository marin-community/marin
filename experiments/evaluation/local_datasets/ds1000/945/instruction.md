Problem:

I'm trying to slice a PyTorch tensor using a logical index on the columns. I want the columns that correspond to a 1 value in the index vector. Both slicing and logical indexing are possible, but are they possible together? If so, how? My attempt keeps throwing the unhelpful error

TypeError: indexing a tensor with an object of type ByteTensor. The only supported types are integers, slices, numpy scalars and torch.LongTensor or torch.ByteTensor as the only argument.

MCVE
Desired Output

import torch

C = torch.LongTensor([[1, 3], [4, 6]])
# 1 3
# 4 6
Logical indexing on the columns only:

A_log = torch.ByteTensor([1, 0, 1]) # the logical index
B = torch.LongTensor([[1, 2, 3], [4, 5, 6]])
C = B[:, A_log] # Throws error
If the vectors are the same size, logical indexing works:

B_truncated = torch.LongTensor([1, 2, 3])
C = B_truncated[A_log]


A:

<code>
import numpy as np
import pandas as pd
import torch
A_log, B = load_data()
def solve(A_log, B):
    # return the solution in this function
    # C = solve(A_log, B)
    ### BEGIN SOLUTION

Execution context (contains [insert] placeholder):
import torch
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            A_log = torch.LongTensor([0, 1, 0])
            B = torch.LongTensor([[1, 2, 3], [4, 5, 6]])
        elif test_case_id == 2:
            A_log = torch.BoolTensor([True, False, True])
            B = torch.LongTensor([[1, 2, 3], [4, 5, 6]])
        elif test_case_id == 3:
            A_log = torch.ByteTensor([1, 1, 0])
            B = torch.LongTensor([[999, 777, 114514], [9999, 7777, 1919810]])
        return A_log, B

    def generate_ans(data):
        A_log, B = data
        C = B[:, A_log.bool()]
        return C

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
A_log, B = test_input
def solve(A_log, B):
[insert]
C = solve(A_log, B)
result = C
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(3):
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
