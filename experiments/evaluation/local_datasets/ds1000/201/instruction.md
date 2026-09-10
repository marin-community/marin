Problem:
I have a data frame with one (string) column and I'd like to split it into three(string) columns, with one column header as 'fips' ,'medi' and 'row'


My dataframe df looks like this:


row
0 00000 UNITED STATES
1 01000 ALAB AMA
2 01001 Autauga County, AL
3 01003 Baldwin County, AL
4 01005 Barbour County, AL
I do not know how to use df.row.str[:] to achieve my goal of splitting the row cell. I can use df['fips'] = hello to add a new column and populate it with hello. Any ideas?


fips medi row
0 00000 UNITED STATES
1 01000 ALAB AMA
2 01001 Autauga County, AL
3 01003 Baldwin County, AL
4 01005 Barbour County, AL






A:
<code>
import pandas as pd


df = pd.DataFrame({'row': ['00000 UNITED STATES', '01000 ALAB AMA',
                           '01001 Autauga County, AL', '01003 Baldwin County, AL',
                           '01005 Barbour County, AL']})
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
        return pd.DataFrame(
            df.row.str.split(" ", 2).tolist(), columns=["fips", "medi", "row"]
        )

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "row": [
                        "00000 UNITED STATES",
                        "01000 ALAB AMA",
                        "01001 Autauga County, AL",
                        "01003 Baldwin County, AL",
                        "01005 Barbour County, AL",
                    ]
                }
            )
        if test_case_id == 2:
            df = pd.DataFrame(
                {
                    "row": [
                        "10000 UNITED STATES",
                        "11000 ALAB AMA",
                        "11001 Autauga County, AL",
                        "11003 Baldwin County, AL",
                        "11005 Barbour County, AL",
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
