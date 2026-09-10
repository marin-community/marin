Problem:
Say that you have 3 numpy arrays: lat, lon, val:
import numpy as np
lat=np.array([[10, 20, 30],
              [20, 11, 33],
              [21, 20, 10]])
lon=np.array([[100, 102, 103],
              [105, 101, 102],
              [100, 102, 103]])
val=np.array([[17, 2, 11],
              [86, 84, 1],
              [9, 5, 10]])
And say that you want to create a pandas dataframe where df.columns = ['lat', 'lon', 'val'], but since each value in lat is associated with both a long and a val quantity, you want them to appear in the same row.
Also, you want the row-wise order of each column to follow the positions in each array, so to obtain the following dataframe:
      lat   lon   val
0     10    100    17
1     20    102    2
2     30    103    11
3     20    105    86
...   ...   ...    ...
Then I want to add a column to its right, consisting of maximum value of each row.
      lat   lon   val   maximum
0     10    100    17   100
1     20    102    2    102
2     30    103    11   103
3     20    105    86   105
...   ...   ...    ...
So basically the first row in the dataframe stores the "first" quantities of each array, and so forth. How to do this?
I couldn't find a pythonic way of doing this, so any help will be much appreciated.
A:
<code>
import numpy as np
import pandas as pd
lat=np.array([[10, 20, 30],
              [20, 11, 33],
              [21, 20, 10]])

lon=np.array([[100, 102, 103],
              [105, 101, 102],
              [100, 102, 103]])

val=np.array([[17, 2, 11],
              [86, 84, 1],
              [9, 5, 10]])
</code>
df = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import pandas as pd
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            lat = np.array([[10, 20, 30], [20, 11, 33], [21, 20, 10]])
            lon = np.array([[100, 102, 103], [105, 101, 102], [100, 102, 103]])
            val = np.array([[17, 2, 11], [86, 84, 1], [9, 5, 10]])
        elif test_case_id == 2:
            np.random.seed(42)
            lat = np.random.rand(5, 6)
            lon = np.random.rand(5, 6)
            val = np.random.rand(5, 6)
        return lat, lon, val

    def generate_ans(data):
        _a = data
        lat, lon, val = _a
        df = pd.DataFrame({"lat": lat.ravel(), "lon": lon.ravel(), "val": val.ravel()})
        df["maximum"] = df.max(axis=1)
        return df

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    pd.testing.assert_frame_equal(result, ans, check_dtype=False)
    return 1


exec_context = r"""
import numpy as np
import pandas as pd
lat, lon, val = test_input
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
