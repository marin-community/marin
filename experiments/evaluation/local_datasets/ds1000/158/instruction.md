Problem:
   Survived  SibSp  Parch
0         0      1      0
1         1      1      0
2         1      0      0
3         1      1      0
4         0      0      1


Given the above dataframe, is there an elegant way to groupby with a condition?
I want to split the data into two groups based on the following conditions:
(df['Survived'] > 0) | (df['Parch'] > 0) =   New Group -"Has Family"
 (df['Survived'] == 0) & (df['Parch'] == 0) = New Group - "No Family"


then take the means of both of these groups and end up with an output like this:


Has Family    0.5
No Family     1.0
Name: SibSp, dtype: float64


Can it be done using groupby or would I have to append a new column using the above conditional statement?


A:
<code>
import pandas as pd


df = pd.DataFrame({'Survived': [0,1,1,1,0],
                   'SibSp': [1,1,0,1,0],
                   'Parch': [0,0,0,0,1]})
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
        family = np.where(
            (df["Survived"] + df["Parch"]) >= 1, "Has Family", "No Family"
        )
        return df.groupby(family)["SibSp"].mean()

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "Survived": [0, 1, 1, 1, 0],
                    "SibSp": [1, 1, 0, 1, 0],
                    "Parch": [0, 0, 0, 0, 1],
                }
            )
        if test_case_id == 2:
            df = pd.DataFrame(
                {
                    "Survived": [1, 0, 0, 0, 1],
                    "SibSp": [0, 0, 1, 0, 1],
                    "Parch": [1, 1, 1, 1, 0],
                }
            )
        return df

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        pd.testing.assert_series_equal(result, ans, check_dtype=False, atol=1e-02)
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
