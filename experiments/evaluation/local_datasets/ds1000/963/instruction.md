Problem:

Given a 3d tenzor, say: batch x sentence length x embedding dim

a = torch.rand((10, 1000, 23))
and an array(or tensor) of actual lengths for each sentence

lengths =  torch .randint(1000,(10,))
outputs tensor([ 137., 152., 165., 159., 145., 264., 265., 276.,1000., 203.])

How to fill tensor ‘a’ with 2333 before certain index along dimension 1 (sentence length) according to tensor ‘lengths’ ?

I want smth like that :

a[ : , : lengths , : ]  = 2333


A:

<code>
import numpy as np
import pandas as pd
import torch
a = torch.rand((10, 1000, 23))
lengths = torch.randint(1000, (10,))
</code>
a = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import torch
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            torch.random.manual_seed(42)
            a = torch.rand((10, 1000, 23))
            lengths = torch.randint(1000, (10,))
        return a, lengths

    def generate_ans(data):
        a, lengths = data
        for i_batch in range(10):
            a[i_batch, : lengths[i_batch], :] = 2333
        return a

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
a, lengths = test_input
[insert]
result = a
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
