Problem:
I have multi-index df as follows


                x  y
id  date            
abc 3/1/1994  100  7
    9/1/1994   90  8
    3/1/1995   80  9
Where dates are stored as str.


I want to parse date index. The following statement


df.index.levels[1] = pd.to_datetime(df.index.levels[1])
returns error:


TypeError: 'FrozenList' does not support mutable operations.


A:
<code>
import pandas as pd


index = pd.MultiIndex.from_tuples([('abc', '3/1/1994'), ('abc', '9/1/1994'), ('abc', '3/1/1995')],
                                 names=('id', 'date'))
df = pd.DataFrame({'x': [100, 90, 80], 'y':[7, 8, 9]}, index=index)
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
        df.index = df.index.set_levels(
            [df.index.levels[0], pd.to_datetime(df.index.levels[1])]
        )
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            index = pd.MultiIndex.from_tuples(
                [("abc", "3/1/1994"), ("abc", "9/1/1994"), ("abc", "3/1/1995")],
                names=("id", "date"),
            )
            df = pd.DataFrame({"x": [100, 90, 80], "y": [7, 8, 9]}, index=index)
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
