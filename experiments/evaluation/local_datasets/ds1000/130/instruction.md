Problem:
I am trying to find duplicates rows in a pandas dataframe.
df=pd.DataFrame(data=[[1,2],[3,4],[1,2],[1,4],[1,2]],columns=['col1','col2'])
df
Out[15]: 
   col1  col2
0     1     2
1     3     4
2     1     2
3     1     4
4     1     2
duplicate_bool = df.duplicated(subset=['col1','col2'], keep='first')
duplicate = df.loc[duplicate_bool == True]
duplicate
Out[16]: 
   col1  col2
2     1     2
4     1     2


Is there a way to add a column referring to the index of the first duplicate (the one kept)
duplicate
Out[16]: 
   col1  col2  index_original
2     1     2               0
4     1     2               0


Note: df could be very very big in my case....


A:
<code>
import pandas as pd


df=pd.DataFrame(data=[[1,2],[3,4],[1,2],[1,4],[1,2]],columns=['col1','col2'])
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
        df["index_original"] = df.groupby(["col1", "col2"]).col1.transform("idxmin")
        return df[df.duplicated(subset=["col1", "col2"], keep="first")]

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                data=[[1, 2], [3, 4], [1, 2], [1, 4], [1, 2]], columns=["col1", "col2"]
            )
        if test_case_id == 2:
            df = pd.DataFrame(
                data=[[1, 1], [3, 1], [1, 4], [1, 9], [1, 6]], columns=["col1", "col2"]
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
