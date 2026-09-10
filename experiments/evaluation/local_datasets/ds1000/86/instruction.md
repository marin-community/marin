Problem:
This is my data frame
  duration
1   year 7
2     day2
3   week 4
4  month 8


I need to separate numbers from time and put them in two new columns. 
I also need to create another column based on the values of time column. So the new dataset is like this:
  duration   time number  time_day
1   year 7   year      7       365
2     day2    day      2         1
3   week 4   week      4         7
4  month 8  month      8        30


df['time_day']= df.time.replace(r'(year|month|week|day)', r'(365|30|7|1)', regex=True, inplace=True)


This is my code:
df ['numer'] = df.duration.replace(r'\d.*' , r'\d', regex=True, inplace = True)
df [ 'time']= df.duration.replace (r'\.w.+',r'\w.+', regex=True, inplace = True )


But it does not work. Any suggestion ?


A:
<code>
import pandas as pd


df = pd.DataFrame({'duration': ['year 7', 'day2', 'week 4', 'month 8']},
                  index=list(range(1,5)))
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
        df[["time", "number"]] = df.duration.str.extract(r"\s*(.*)(\d+)", expand=True)
        for i in df.index:
            df.loc[i, "time"] = df.loc[i, "time"].strip()
        df["time_days"] = df["time"].replace(
            ["year", "month", "week", "day"], [365, 30, 7, 1], regex=True
        )
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {"duration": ["year 7", "day2", "week 4", "month 8"]},
                index=list(range(1, 5)),
            )
        if test_case_id == 2:
            df = pd.DataFrame(
                {"duration": ["year 2", "day6", "week 8", "month 7"]},
                index=list(range(1, 5)),
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
