Problem:

I'd like to convert a torch tensor to pandas dataframe but by using pd.DataFrame I'm getting a dataframe filled with tensors instead of numeric values.

import torch
import pandas as  pd
x = torch.rand(4,4)
px = pd.DataFrame(x)
Here's what I get when clicking on px in the variable explorer:

0   1   2   3
tensor(0.3880)  tensor(0.4598)  tensor(0.4239)  tensor(0.7376)
tensor(0.4174)  tensor(0.9581)  tensor(0.0987)  tensor(0.6359)
tensor(0.6199)  tensor(0.8235)  tensor(0.9947)  tensor(0.9679)
tensor(0.7164)  tensor(0.9270)  tensor(0.7853)  tensor(0.6921)


A:

<code>
import numpy as np
import torch
import pandas as pd
x = load_data()
</code>
px = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import pandas as pd
import torch
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        torch.random.manual_seed(42)
        if test_case_id == 1:
            x = torch.rand(4, 4)
        elif test_case_id == 2:
            x = torch.rand(6, 6)
        return x

    def generate_ans(data):
        x = data
        px = pd.DataFrame(x.numpy())
        return px

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        assert type(result) == pd.DataFrame
        np.testing.assert_allclose(result, ans)
        return 1
    except:
        return 0


exec_context = r"""
import numpy as np
import pandas as pd
import torch
x = test_input
[insert]
result = px
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
