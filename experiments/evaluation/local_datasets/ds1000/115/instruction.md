Problem:
I have my data in a pandas DataFrame, and it looks like the following:
cat  val1   val2   val3   val4
A    7      10     0      19
B    10     2      1      14
C    5      15     6      16


I'd like to compute the percentage of the category (cat) that each value has. 
For example, for category A, val1 is 7 and the row total is 36. The resulting value would be 7/36, so val1 is 19.4% of category A.
My expected result would look like the following:
cat  val1   val2   val3   val4
A    .194   .278   .0     .528
B    .370   .074   .037   .519
C    .119   .357   .143   .381


Is there an easy way to compute this?


A:
<code>
import pandas as pd


df = pd.DataFrame({'cat': ['A', 'B', 'C'],
                   'val1': [7, 10, 5],
                   'val2': [10, 2, 15],
                   'val3': [0, 1, 6],
                   'val4': [19, 14, 16]})
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
        df = df.set_index("cat")
        res = df.div(df.sum(axis=1), axis=0)
        return res.reset_index()

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "cat": ["A", "B", "C"],
                    "val1": [7, 10, 5],
                    "val2": [10, 2, 15],
                    "val3": [0, 1, 6],
                    "val4": [19, 14, 16],
                }
            )
        if test_case_id == 2:
            df = pd.DataFrame(
                {
                    "cat": ["A", "B", "C"],
                    "val1": [1, 1, 4],
                    "val2": [10, 2, 15],
                    "val3": [0, 1, 6],
                    "val4": [19, 14, 16],
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
