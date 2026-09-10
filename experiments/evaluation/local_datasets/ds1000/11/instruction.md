Problem:
I have been struggling with removing the time zone info from a column in a pandas dataframe. I have checked the following question, but it does not work for me:


Can I export pandas DataFrame to Excel stripping tzinfo?


I used tz_localize to assign a timezone to a datetime object, because I need to convert to another timezone using tz_convert. This adds an UTC offset, in the way "-06:00". I need to get rid of this offset, because it results in an error when I try to export the dataframe to Excel.


Actual output


2015-12-01 00:00:00-06:00


Desired output
2015-12-01 00:00:00


I have tried to get the characters I want using the str() method, but it seems the result of tz_localize is not a string. My solution so far is to export the dataframe to csv, read the file, and to use the str() method to get the characters I want.
Is there an easier solution?


A:
<code>
import pandas as pd


df = pd.DataFrame({'datetime': ['2015-12-01 00:00:00-06:00', '2015-12-02 00:01:00-06:00', '2015-12-03 00:00:00-06:00']})
df['datetime'] = pd.to_datetime(df['datetime'])
</code>
df = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import pandas as pd
import numpy as np
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        df["datetime"] = df["datetime"].dt.tz_localize(None)
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "datetime": [
                        "2015-12-01 00:00:00-06:00",
                        "2015-12-02 00:01:00-06:00",
                        "2015-12-03 00:00:00-06:00",
                    ]
                }
            )
            df["datetime"] = pd.to_datetime(df["datetime"])
        elif test_case_id == 2:
            df = pd.DataFrame(
                {
                    "datetime": [
                        "2016-12-02 00:01:00-06:00",
                        "2016-12-01 00:00:00-06:00",
                        "2016-12-03 00:00:00-06:00",
                    ]
                }
            )
            df["datetime"] = pd.to_datetime(df["datetime"])
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


def test_string(solution: str):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(solution.encode("utf-8")).readline):
        tokens.append(token.string)
    assert "tz_localize" in tokens

Return only the Python code snippet that replaces [insert]:
- No explanations, Markdown, comments, prints, or extra text; no non-English output
- No imports, function/class definitions, or placeholders; only the missing logic
- Do not reassign/reset provided variables/inputs; use them to compute the result
- The snippet must be syntactically valid and run as-is in this context; deterministic (assume temperature=0)
- Do not read/write files or call external services
- Output only the code lines for [insert]; no Markdown fences, no explanations, no bullets.
