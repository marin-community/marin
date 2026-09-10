Problem:


I have a pandas series which values are numpy array. For simplicity, say




    series = pd.Series([np.array([1,2,3,4]), np.array([5,6,7,8]), np.array([9,10,11,12])], index=['file1', 'file2', 'file3'])


file1       [1, 2, 3, 4]
file2       [5, 6, 7, 8]
file3    [9, 10, 11, 12]


How can I expand it to a dataframe of the form df_concatenated:
       0   1   2   3
file1  1   2   3   4
file2  5   6   7   8
file3  9  10  11  12


A:
<code>
import pandas as pd
import numpy as np


series = pd.Series([np.array([1,2,3,4]), np.array([5,6,7,8]), np.array([9,10,11,12])], index=['file1', 'file2', 'file3'])
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
        s = data
        return pd.DataFrame.from_records(s.values, index=s.index)

    def define_test_input(test_case_id):
        if test_case_id == 1:
            series = pd.Series(
                [
                    np.array([1, 2, 3, 4]),
                    np.array([5, 6, 7, 8]),
                    np.array([9, 10, 11, 12]),
                ],
                index=["file1", "file2", "file3"],
            )
        if test_case_id == 2:
            series = pd.Series(
                [
                    np.array([11, 12, 13, 14]),
                    np.array([5, 6, 7, 8]),
                    np.array([9, 10, 11, 12]),
                ],
                index=["file1", "file2", "file3"],
            )
        return series

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
series = test_input
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
