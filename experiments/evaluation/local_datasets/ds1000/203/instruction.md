Problem:
I have a Dataframe as below.
Name  2001 2002 2003 2004 2005 2006  
Name1  2    5     0    0    4    6  
Name2  1    4     2    0    4    0  
Name3  0    5     0    0    0    2  


I wanted to calculate the cumulative average for each row from end to head using pandas, But while calculating the Average It has to ignore if the value is zero.
The expected output is as below.
 Name  2001  2002  2003  2004  2005  2006
Name1  3.50   5.0     5     5     5     6
Name2  2.25   3.5     3     4     4     0
Name3  3.50   3.5     2     2     2     2


A:
<code>
import pandas as pd


df = pd.DataFrame({'Name': ['Name1', 'Name2', 'Name3'],
                   '2001': [2, 1, 0],
                   '2002': [5, 4, 5],
                   '2003': [0, 2, 0],
                   '2004': [0, 0, 0],
                   '2005': [4, 4, 0],
                   '2006': [6, 0, 2]})
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
        cols = list(df)[1:]
        cols = cols[::-1]
        for idx in df.index:
            s = 0
            cnt = 0
            for col in cols:
                if df.loc[idx, col] != 0:
                    cnt = min(cnt + 1, 2)
                    s = (s + df.loc[idx, col]) / cnt
                df.loc[idx, col] = s
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "Name": ["Name1", "Name2", "Name3"],
                    "2001": [2, 1, 0],
                    "2002": [5, 4, 5],
                    "2003": [0, 2, 0],
                    "2004": [0, 0, 0],
                    "2005": [4, 4, 0],
                    "2006": [6, 0, 2],
                }
            )
        if test_case_id == 2:
            df = pd.DataFrame(
                {
                    "Name": ["Name1", "Name2", "Name3"],
                    "2011": [2, 1, 0],
                    "2012": [5, 4, 5],
                    "2013": [0, 2, 0],
                    "2014": [0, 0, 0],
                    "2015": [4, 4, 0],
                    "2016": [6, 0, 2],
                }
            )
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
