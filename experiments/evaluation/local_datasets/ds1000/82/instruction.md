Problem:
I have the following dataframe:
index = range(14)
data = [1, 0, 0, 2, 0, 4, 6, 8, 0, 0, 0, 0, 2, 1]
df = pd.DataFrame(data=data, index=index, columns = ['A'])


How can I fill the zeros with the previous non-zero value using pandas? Is there a fillna that is not just for "NaN"?.  
The output should look like:
    A
0   1
1   1
2   1
3   2
4   2
5   4
6   6
7   8
8   8
9   8
10  8
11  8
12  2
13  1




A:
<code>
import pandas as pd


index = range(14)
data = [1, 0, 0, 2, 0, 4, 6, 8, 0, 0, 0, 0, 2, 1]
df = pd.DataFrame(data=data, index=index, columns = ['A'])
</code>
df = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        df["A"].replace(to_replace=0, method="ffill", inplace=True)
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            index = range(14)
            data = [1, 0, 0, 2, 0, 4, 6, 8, 0, 0, 0, 0, 2, 1]
            df = pd.DataFrame(data=data, index=index, columns=["A"])
        if test_case_id == 2:
            index = range(14)
            data = [1, 0, 0, 9, 0, 2, 6, 8, 0, 0, 0, 0, 1, 7]
            df = pd.DataFrame(data=data, index=index, columns=["A"])
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
result = df
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
