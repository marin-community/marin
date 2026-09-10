Problem:
Let's say I have 5 columns.
pd.DataFrame({
'Column1': [1, 2, 3, 4, 5, 6, 7, 8, 9],
'Column2': [4, 3, 6, 8, 3, 4, 1, 4, 3],
'Column3': [7, 3, 3, 1, 2, 2, 3, 2, 7],
'Column4': [9, 8, 7, 6, 5, 4, 3, 2, 1],
'Column5': [1, 1, 1, 1, 1, 1, 1, 1, 1]})


Is there a function to know the type of relationship each par of columns has? (one-2-one, one-2-many, many-2-one, many-2-many)
An DataFrame output like:
            Column1      Column2      Column3     Column4      Column5
Column1         NaN   one-2-many   one-2-many   one-2-one   one-2-many
Column2  many-2-one          NaN  many-2-many  many-2-one  many-2-many
Column3  many-2-one  many-2-many          NaN  many-2-one  many-2-many
Column4   one-2-one   one-2-many   one-2-many         NaN   one-2-many
Column5  many-2-one  many-2-many  many-2-many  many-2-one          NaN


A:
<code>
import pandas as pd


df = pd.DataFrame({
    'Column1': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'Column2': [4, 3, 6, 8, 3, 4, 1, 4, 3],
    'Column3': [7, 3, 3, 1, 2, 2, 3, 2, 7],
    'Column4': [9, 8, 7, 6, 5, 4, 3, 2, 1],
    'Column5': [1, 1, 1, 1, 1, 1, 1, 1, 1]})
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

        def get_relation(df, col1, col2):
            first_max = df[[col1, col2]].groupby(col1).count().max()[0]
            second_max = df[[col1, col2]].groupby(col2).count().max()[0]
            if first_max == 1:
                if second_max == 1:
                    return "one-2-one"
                else:
                    return "one-2-many"
            else:
                if second_max == 1:
                    return "many-2-one"
                else:
                    return "many-2-many"

        result = pd.DataFrame(index=df.columns, columns=df.columns)
        for col_i in df.columns:
            for col_j in df.columns:
                if col_i == col_j:
                    continue
                result.loc[col_i, col_j] = get_relation(df, col_i, col_j)
        return result

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "Column1": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                    "Column2": [4, 3, 6, 8, 3, 4, 1, 4, 3],
                    "Column3": [7, 3, 3, 1, 2, 2, 3, 2, 7],
                    "Column4": [9, 8, 7, 6, 5, 4, 3, 2, 1],
                    "Column5": [1, 1, 1, 1, 1, 1, 1, 1, 1],
                }
            )
        if test_case_id == 2:
            df = pd.DataFrame(
                {
                    "Column1": [4, 3, 6, 8, 3, 4, 1, 4, 3],
                    "Column2": [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    "Column3": [7, 3, 3, 1, 2, 2, 3, 2, 7],
                    "Column4": [9, 8, 7, 6, 5, 4, 3, 2, 1],
                    "Column5": [1, 2, 3, 4, 5, 6, 7, 8, 9],
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
