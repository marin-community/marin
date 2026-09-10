Problem:
I have a data frame with one (string) column and I'd like to split it into two (string) columns, with one column header as 'fips' and the other 'row'


My dataframe df looks like this:


row
0 114 AAAAAA
1 514 ENENEN
2 1926 HAHAHA
3 0817 O-O,O-O
4 998244353 TTTTTT
I do not know how to use df.row.str[:] to achieve my goal of splitting the row cell. I can use df['fips'] = hello to add a new column and populate it with hello. Any ideas?


fips row
0 114 AAAAAA
1 514 ENENEN
2 1926 HAHAHA
3 0817 O-O,O-O
4 998244353 TTTTTT






A:
<code>
import pandas as pd


df = pd.DataFrame({'row': ['114 AAAAAA', '514 ENENEN',
                           '1926 HAHAHA', '0817 O-O,O-O',
                           '998244353 TTTTTT']})
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
        return pd.DataFrame(df.row.str.split(" ", 1).tolist(), columns=["fips", "row"])

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "row": [
                        "114 AAAAAA",
                        "514 ENENEN",
                        "1926 HAHAHA",
                        "0817 O-O,O-O",
                        "998244353 TTTTTT",
                    ]
                }
            )
        if test_case_id == 2:
            df = pd.DataFrame(
                {
                    "row": [
                        "00000 UNITED STATES",
                        "01000 ALABAMA",
                        "01001 Autauga County, AL",
                        "01003 Baldwin County, AL",
                        "01005 Barbour County, AL",
                    ]
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
