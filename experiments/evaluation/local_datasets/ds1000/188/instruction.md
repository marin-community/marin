Problem:
I have a dataframe, e.g:
Date             B           C   
20.07.2018      10           8
20.07.2018       1           0
21.07.2018       0           1
21.07.2018       1           0


How can I count the zero and non-zero values for each column for each date?
Using .sum() doesn't help me because it will sum the non-zero values.
e.g: expected output for the zero values:
            B  C
Date            
20.07.2018  0  1
21.07.2018  1  1


non-zero values:
            B  C
Date            
20.07.2018  2  1
21.07.2018  1  1


A:
<code>
import pandas as pd


df = pd.DataFrame({'Date': ['20.07.2018', '20.07.2018', '21.07.2018', '21.07.2018'],
                   'B': [10, 1, 0, 1],
                   'C': [8, 0, 1, 0]})
</code>
result1: zero
result2: non-zero
result1, result2 = ... # put solution in these variables
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        df1 = df.groupby("Date").agg(lambda x: x.eq(0).sum())
        df2 = df.groupby("Date").agg(lambda x: x.ne(0).sum())
        return df1, df2

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "Date": ["20.07.2018", "20.07.2018", "21.07.2018", "21.07.2018"],
                    "B": [10, 1, 0, 1],
                    "C": [8, 0, 1, 0],
                }
            )
        if test_case_id == 2:
            df = pd.DataFrame(
                {
                    "Date": ["20.07.2019", "20.07.2019", "21.07.2019", "21.07.2019"],
                    "B": [10, 1, 0, 1],
                    "C": [8, 0, 1, 0],
                }
            )
        return df

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        pd.testing.assert_frame_equal(ans[0], result[0], check_dtype=False)
        pd.testing.assert_frame_equal(ans[1], result[1], check_dtype=False)
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
df = test_input
[insert]
result = (result1, result2)
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
