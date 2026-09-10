Problem:
I have this Pandas dataframe (df):
     A    B
0    1    green
1    2    red
2    s    blue
3    3    yellow
4    b    black


A type is object.
I'd select the record where A value are string to have:
   A      B
2  s   blue
4  b  black


Thanks


A:
<code>
import pandas as pd


df = pd.DataFrame({'A': [1, 2, 's', 3, 'b'],
                   'B': ['green', 'red', 'blue', 'yellow', 'black']})
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
        result = []
        for i in range(len(df)):
            if type(df.loc[i, "A"]) == str:
                result.append(i)
        return df.iloc[result]

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "A": [-1, 2, "s", 3, "b"],
                    "B": ["green", "red", "blue", "yellow", "black"],
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
