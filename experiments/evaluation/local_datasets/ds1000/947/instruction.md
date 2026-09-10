Problem:

I'm trying to slice a PyTorch tensor using an index on the columns. The index, contains a list of columns that I want to select in order. You can see the example later.
I know that there is a function index_select. Now if I have the index, which is a LongTensor, how can I apply index_select to get the expected result?

For example:
the expected output:
C = torch.LongTensor([[1, 3], [4, 6]])
# 1 3
# 4 6
the index and the original data should be:
idx = torch.LongTensor([1, 2])
B = torch.LongTensor([[2, 1, 3], [5, 4, 6]])

Thanks.


A:

<code>
import numpy as np
import pandas as pd
import torch
idx, B = load_data()
</code>
C = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import torch
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            idx = torch.LongTensor([1, 2])
            B = torch.LongTensor([[2, 1, 3], [5, 4, 6]])
        elif test_case_id == 2:
            idx = torch.LongTensor([0, 1, 3])
            B = torch.LongTensor([[1, 2, 3, 777], [4, 999, 5, 6]])
        return idx, B

    def generate_ans(data):
        idx, B = data
        C = B.index_select(1, idx)
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
idx, B = test_input
[insert]
result = C
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
    assert "index_select" in tokens

Return only the Python code snippet that replaces [insert]:
- No explanations, Markdown, comments, prints, or extra text; no non-English output
- No imports, function/class definitions, or placeholders; only the missing logic
- Do not reassign/reset provided variables/inputs; use them to compute the result
- The snippet must be syntactically valid and run as-is in this context; deterministic (assume temperature=0)
- Do not read/write files or call external services
- Output only the code lines for [insert]; no Markdown fences, no explanations, no bullets.
