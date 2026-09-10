Problem:
Example
import pandas as pd
import numpy as np
d = {'l':  ['left', 'right', 'left', 'right', 'left', 'right'],
     'r': ['right', 'left', 'right', 'left', 'right', 'left'],
     'v': [-1, 1, -1, 1, -1, np.nan]}
df = pd.DataFrame(d)


Problem
When a grouped dataframe contains a value of np.NaN I want the grouped sum to be NaN as is given by the skipna=False flag for pd.Series.sum and also pd.DataFrame.sum however, this
In [235]: df.v.sum(skipna=False)
Out[235]: nan


However, this behavior is not reflected in the pandas.DataFrame.groupby object
In [237]: df.groupby('r')['v'].sum()['right']
Out[237]: 2.0


and cannot be forced by applying the np.sum method directly
In [238]: df.groupby('r')['v'].apply(np.sum)['right']
Out[238]: 2.0


desired:
r
left     NaN
right   -3.0
Name: v, dtype: float64


A:
<code>
import pandas as pd
import numpy as np


d = {'l':  ['left', 'right', 'left', 'right', 'left', 'right'],
     'r': ['right', 'left', 'right', 'left', 'right', 'left'],
     'v': [-1, 1, -1, 1, -1, np.nan]}
df = pd.DataFrame(d)
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
        return df.groupby("r")["v"].apply(pd.Series.sum, skipna=False)

    def define_test_input(test_case_id):
        if test_case_id == 1:
            d = {
                "l": ["left", "right", "left", "right", "left", "right"],
                "r": ["right", "left", "right", "left", "right", "left"],
                "v": [-1, 1, -1, 1, -1, np.nan],
            }
            df = pd.DataFrame(d)
        if test_case_id == 2:
            d = {
                "l": ["left", "right", "left", "left", "right", "right"],
                "r": ["right", "left", "right", "right", "left", "left"],
                "v": [-1, 1, -1, -1, 1, np.nan],
            }
            df = pd.DataFrame(d)
        return df

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        pd.testing.assert_series_equal(result, ans, check_dtype=False)
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
