Problem:
I have a script that generates a pandas data frame with a varying number of value columns. As an example, this df might be
import pandas as pd
df = pd.DataFrame({
'group': ['A', 'A', 'A', 'B', 'B'],
'group_color' : ['green', 'green', 'green', 'blue', 'blue'],
'val1': [5, 2, 3, 4, 5], 
'val2' : [4, 2, 8, 5, 7]
})
  group group_color  val1  val2
0     A       green     5     4
1     A       green     2     2
2     A       green     3     8
3     B        blue     4     5
4     B        blue     5     7


My goal is to get the grouped sum for each of the value columns. In this specific case (with 2 value columns), I can use
df.groupby('group').agg({"group_color": "first", "val1": "sum", "val2": "sum"})
      group_color  val1  val2
group                        
A           green    10    14
B            blue     9    12


but that does not work when the data frame in question has more value columns (val3, val4 etc.).
Is there a way to dynamically take the sum of "all the other columns" or "all columns containing val in their names"?


A:
<code>
import pandas as pd


df = pd.DataFrame({ 'group': ['A', 'A', 'A', 'B', 'B'], 'group_color' : ['green', 'green', 'green', 'blue', 'blue'], 'val1': [5, 2, 3, 4, 5], 'val2' : [4, 2, 8, 5, 7],'val3':[1,1,4,5,1] })
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
        return df.groupby("group").agg(
            lambda x: x.head(1) if x.dtype == "object" else x.sum()
        )

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "group": ["A", "A", "A", "B", "B"],
                    "group_color": ["green", "green", "green", "blue", "blue"],
                    "val1": [5, 2, 3, 4, 5],
                    "val2": [4, 2, 8, 5, 7],
                    "val3": [1, 1, 4, 5, 1],
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
