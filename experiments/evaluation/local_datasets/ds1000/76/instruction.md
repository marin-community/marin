Problem:
I have a simple dataframe which I would like to bin for every 3 rows.


It looks like this:


    col1
0      2
1      1
2      3
3      1
4      0
and I would like to turn it into this:


    col1
0      2
1    0.5
I have already posted a similar question here but I have no Idea how to port the solution to my current use case.


Can you help me out?


Many thanks!




A:
<code>
import pandas as pd


df = pd.DataFrame({'col1':[2, 1, 3, 1, 0]})
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        return df.groupby(df.index // 3).mean()

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame({"col1": [2, 1, 3, 1, 0]})
        if test_case_id == 2:
            df = pd.DataFrame({"col1": [1, 9, 2, 6, 8]})
        return df

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        pd.testing.assert_frame_equal(result, ans, check_dtype=False)
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
df = test_input
[insert]
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
