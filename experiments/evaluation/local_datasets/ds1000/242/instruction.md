Problem:
I have two DataFrames C and D as follows:
C
    A  B
0  AB  1
1  CD  2
2  EF  3
D
    A  B
1  CD  4
2  GH  5


I have to merge both the dataframes but the merge should overwrite the values in the right df. Rest of the rows from the dataframe should not change. I want to add a new column 'dulplicated'. If datafram C and D have the same A in this row, dulplicated = True, else False.


Output
    A  B   dulplicated
0  AB  1   False
1  CD  4   True
2  EF  3   False
3  GH  5   False


The order of the rows of df must not change i.e. CD should remain in index 1. I tried using outer merge which is handling index but duplicating columns instead of overwriting.
>>> pd.merge(c,d, how='outer', on='A')
    A  B_x  B_y
0  AB  1.0  NaN
1  CD  2.0  4.0
2  EF  3.0  NaN
3  GH  NaN  5.0 


Basically B_y should have replaced values in B_x(only where values occur).
I am using Python3.7.


A:
<code>
import pandas as pd


C = pd.DataFrame({"A": ["AB", "CD", "EF"], "B": [1, 2, 3]})
D = pd.DataFrame({"A": ["CD", "GH"], "B": [4, 5]})
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
        data = data
        C, D = data
        df = (
            pd.concat([C, D])
            .drop_duplicates("A", keep="last")
            .sort_values(by=["A"])
            .reset_index(drop=True)
        )
        for i in range(len(C)):
            if df.loc[i, "A"] in D.A.values:
                df.loc[i, "dulplicated"] = True
            else:
                df.loc[i, "dulplicated"] = False
        for i in range(len(C), len(df)):
            df.loc[i, "dulplicated"] = False
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            C = pd.DataFrame({"A": ["AB", "CD", "EF"], "B": [1, 2, 3]})
            D = pd.DataFrame({"A": ["CD", "GH"], "B": [4, 5]})
        if test_case_id == 2:
            D = pd.DataFrame({"A": ["AB", "CD", "EF"], "B": [1, 2, 3]})
            C = pd.DataFrame({"A": ["CD", "GH"], "B": [4, 5]})
        return C, D

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
C, D = test_input
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
