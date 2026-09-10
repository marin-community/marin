Problem:
I've seen similar questions but mine is more direct and abstract.

I have a dataframe with "n" rows, being "n" a small number.We can assume the index is just the row number. I would like to convert it to just one row.

So for example if I have

A,B,C,D,E
---------
1,2,3,4,5
6,7,8,9,10
11,12,13,14,5
I want as a result a dataframe with a single row:

A_1,B_1,C_1,D_1,E_1,A_2,B_2_,C_2,D_2,E_2,A_3,B_3,C_3,D_3,E_3
--------------------------
1,2,3,4,5,6,7,8,9,10,11,12,13,14,5
What would be the most idiomatic way to do this in Pandas?

A:
<code>
import pandas as pd

df = pd.DataFrame([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]],columns=['A','B','C','D','E'])
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
        df.index += 1
        df_out = df.stack()
        df.index -= 1
        df_out.index = df_out.index.map("{0[1]}_{0[0]}".format)
        return df_out.to_frame().T

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
                columns=["A", "B", "C", "D", "E"],
            )
        if test_case_id == 2:
            df = pd.DataFrame(
                [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], columns=["A", "B", "C", "D", "E"]
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


def test_string(solution: str):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(solution.encode("utf-8")).readline):
        tokens.append(token.string)
    assert "for" not in tokens and "while" not in tokens

Return only the Python code snippet that replaces [insert]:
- No explanations, Markdown, comments, prints, or extra text; no non-English output
- No imports, function/class definitions, or placeholders; only the missing logic
- Do not reassign/reset provided variables/inputs; use them to compute the result
- The snippet must be syntactically valid and run as-is in this context; deterministic (assume temperature=0)
- Do not read/write files or call external services
- Output only the code lines for [insert]; no Markdown fences, no explanations, no bullets.
