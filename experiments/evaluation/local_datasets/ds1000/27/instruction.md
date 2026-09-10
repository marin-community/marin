Problem:
So I have a dataframe that looks like this:
                         #1                     #2
1980-01-01               11.6985                126.0
1980-01-02               43.6431                134.0
1980-01-03               54.9089                130.0
1980-01-04               63.1225                126.0
1980-01-05               72.4399                120.0


What I want to do is to shift the last row of the first column (72.4399) up 1 row, and then the first row of the first column (11.6985) would be shifted to the last row, first column, like so:
                 #1     #2
1980-01-01  43.6431  126.0
1980-01-02  54.9089  134.0
1980-01-03  63.1225  130.0
1980-01-04  72.4399  126.0
1980-01-05  11.6985  120.0


The idea is that I want to use these dataframes to find an R^2 value for every shift, so I need to use all the data or it might not work. I have tried to use <a href="https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shift.html" rel="noreferrer">pandas.Dataframe.shift()</a>:
print(data)
#Output
1980-01-01               11.6985                126.0
1980-01-02               43.6431                134.0
1980-01-03               54.9089                130.0
1980-01-04               63.1225                126.0
1980-01-05               72.4399                120.0
print(data.shift(1,axis = 0))
1980-01-01                   NaN                  NaN
1980-01-02               11.6985                126.0
1980-01-03               43.6431                134.0
1980-01-04               54.9089                130.0
1980-01-05               63.1225                126.0


So it just shifts both columns down and gets rid of the last row of data, which is not what I want.
Any advice?


A:
<code>
import pandas as pd


df = pd.DataFrame({'#1': [11.6985, 43.6431, 54.9089, 63.1225, 72.4399],
                   '#2': [126.0, 134.0, 130.0, 126.0, 120.0]},
                  index=['1980-01-01', '1980-01-02', '1980-01-03', '1980-01-04', '1980-01-05'])
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
        df["#1"] = np.roll(df["#1"], shift=-1)
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "#1": [11.6985, 43.6431, 54.9089, 63.1225, 72.4399],
                    "#2": [126.0, 134.0, 130.0, 126.0, 120.0],
                },
                index=[
                    "1980-01-01",
                    "1980-01-02",
                    "1980-01-03",
                    "1980-01-04",
                    "1980-01-05",
                ],
            )
        elif test_case_id == 2:
            df = pd.DataFrame(
                {"#1": [45, 51, 14, 11, 14], "#2": [126.0, 134.0, 130.0, 126.0, 120.0]},
                index=[
                    "1980-01-01",
                    "1980-01-02",
                    "1980-01-03",
                    "1980-01-04",
                    "1980-01-05",
                ],
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
