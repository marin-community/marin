Problem:
I have multi-index df as follows


                        x  y
date        id         
3/1/1994 abc   100  7
9/1/1994 abc   90  8
3/1/1995 abc    80  9
Where dates are stored as str.


I want to parse date index using pd.to_datetime, and swap the two levels.
The final output should be
                x  y
id  date            
abc 1994-03-01  100  7
    1994-09-01   90  8
    1995-03-01   80  9
 Any help would be appreciated.

A:
<code>
import pandas as pd
def f(df):
    # return the solution in this function
    # df = f(df)
    ### BEGIN SOLUTION

Execution context (contains [insert] placeholder):
import pandas as pd
import numpy as np
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        df.index = df.index.from_tuples(
            [(x[1], pd.to_datetime(x[0])) for x in df.index.values],
            names=[df.index.names[1], df.index.names[0]],
        )
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            index = pd.MultiIndex.from_tuples(
                [("3/1/1994", "abc"), ("9/1/1994", "abc"), ("3/1/1995", "abc")],
                names=("date", "id"),
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
def f(df):
[insert]
df = test_input
result = f(df)
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)


def test_string(solution: str):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(solution.encode("utf-8")).readline):
        tokens.append(token.string)
    assert "pd" in tokens and "to_datetime" in tokens

Return only the Python code snippet that replaces [insert]:
- No explanations, Markdown, comments, prints, or extra text; no non-English output
- No imports, function/class definitions, or placeholders; only the missing logic
- Do not reassign/reset provided variables/inputs; use them to compute the result
- The snippet must be syntactically valid and run as-is in this context; deterministic (assume temperature=0)
- Do not read/write files or call external services
- Output only the code lines for [insert]; no Markdown fences, no explanations, no bullets.
