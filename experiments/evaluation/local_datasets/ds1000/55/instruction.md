Problem:
The title might not be intuitive--let me provide an example.  Say I have df, created with:
a = np.array([[ 1. ,  0.9,  1. ],
              [ 0.9,  0.9,  1. ],
              [ 0.8,  1. ,  0.5],
              [ 1. ,  0.3,  0.2],
              [ 1. ,  0.2,  0.1],
              [ 0.9,  1. ,  1. ],
              [ 1. ,  0.9,  1. ],
              [ 0.6,  0.9,  0.7],
              [ 1. ,  0.9,  0.8],
              [ 1. ,  0.8,  0.9]])
idx = pd.date_range('2017', periods=a.shape[0])
df = pd.DataFrame(a, index=idx, columns=list('abc'))


I can get the index location of each respective column minimum with
df.idxmin()


Now, how could I get the location of the first occurrence of the column-wise maximum, down to the location of the minimum?


where the max's before the minimum occurrence are ignored.
I can do this with .apply, but can it be done with a mask/advanced indexing
Desired result:
a   2017-01-09
b   2017-01-06
c   2017-01-06
dtype: datetime64[ns]


A:
<code>
import pandas as pd
import numpy as np


a = np.array([[ 1. ,  0.9,  1. ],
              [ 0.9,  0.9,  1. ],
              [ 0.8,  1. ,  0.5],
              [ 1. ,  0.3,  0.2],
              [ 1. ,  0.2,  0.1],
              [ 0.9,  1. ,  1. ],
              [ 1. ,  0.9,  1. ],
              [ 0.6,  0.9,  0.7],
              [ 1. ,  0.9,  0.8],
              [ 1. ,  0.8,  0.9]])


idx = pd.date_range('2017', periods=a.shape[0])
df = pd.DataFrame(a, index=idx, columns=list('abc'))
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
        return df.mask(~(df == df.min()).cumsum().astype(bool)).idxmax()

    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.array(
                [
                    [1.0, 0.9, 1.0],
                    [0.9, 0.9, 1.0],
                    [0.8, 1.0, 0.5],
                    [1.0, 0.3, 0.2],
                    [1.0, 0.2, 0.1],
                    [0.9, 1.0, 1.0],
                    [1.0, 0.9, 1.0],
                    [0.6, 0.9, 0.7],
                    [1.0, 0.9, 0.8],
                    [1.0, 0.8, 0.9],
                ]
            )
            idx = pd.date_range("2017", periods=a.shape[0])
            df = pd.DataFrame(a, index=idx, columns=list("abc"))
        if test_case_id == 2:
            a = np.array(
                [
                    [1.0, 0.9, 1.0],
                    [0.9, 0.9, 1.0],
                    [0.8, 1.0, 0.5],
                    [1.0, 0.3, 0.2],
                    [1.0, 0.2, 0.1],
                    [0.9, 1.0, 1.0],
                    [0.9, 0.9, 1.0],
                    [0.6, 0.9, 0.7],
                    [1.0, 0.9, 0.8],
                    [1.0, 0.8, 0.9],
                ]
            )
            idx = pd.date_range("2022", periods=a.shape[0])
            df = pd.DataFrame(a, index=idx, columns=list("abc"))
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
