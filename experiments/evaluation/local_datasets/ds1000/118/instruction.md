Problem:
I am trying to extract rows from a Pandas dataframe using a list of row names, but it can't be done. Here is an example


# df
    alias  chrome  poston 
rs#
TP3      A/C      0    3   
TP7      A/T      0    7   
TP12     T/A      0   12  
TP15     C/A      0   15 
TP18     C/T      0   18


rows = ['TP3', 'TP18']


df.select(rows)
This is what I was trying to do with just element of the list and I am getting this error TypeError: 'Index' object is not callable. What am I doing wrong?

A:
<code>
import pandas as pd
import io

data = io.StringIO("""
rs    alias  chrome  poston
TP3      A/C      0    3
TP7      A/T      0    7
TP12     T/A      0   12
TP15     C/A      0   15
TP18     C/T      0   18
""")
df = pd.read_csv(data, delim_whitespace=True).set_index('rs')
test = ['TP3', 'TP18']
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import pandas as pd
import numpy as np
import io
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        data = data
        df, test = data
        return df.loc[test]

    def define_test_input(test_case_id):
        if test_case_id == 1:
            data = io.StringIO(
                """
            rs    alias  chrome  poston 
            TP3      A/C      0    3  
            TP7      A/T      0    7
            TP12     T/A      0   12 
            TP15     C/A      0   15
            TP18     C/T      0   18 
            """
            )
            df = pd.read_csv(data, delim_whitespace=True).set_index("rs")
            test = ["TP3", "TP18"]
        return df, test

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
import io
df, test = test_input
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
