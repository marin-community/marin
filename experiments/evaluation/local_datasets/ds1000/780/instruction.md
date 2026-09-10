Problem:
I have the following data frame:
import pandas as pd
import io
from scipy import stats
temp=u"""probegenes,sample1,sample2,sample3
1415777_at Pnliprp1,20,0.00,11
1415805_at Clps,17,0.00,55
1415884_at Cela3b,47,0.00,100"""
df = pd.read_csv(io.StringIO(temp),index_col='probegenes')
df
It looks like this
                     sample1  sample2  sample3
probegenes
1415777_at Pnliprp1       20        0       11
1415805_at Clps           17        0       55
1415884_at Cela3b         47        0      100
What I want to do is too perform column-zscore calculation using SCIPY. AND I want to show data and zscore together in a single dataframe. For each element, I want to only keep 3 decimals places. At the end of the day. the result will look like:
                               sample1  sample2  sample3
probegenes
1415777_at Pnliprp1   data     20.000    0.000    11.000
					zscore	   -0.593    NaN    -1.220
1415805_at Clps		  data     17.000	0.000	55.000
					zscore     -0.815    NaN    -0.009
1415884_at Cela3b	  data     47.000	0.000	100.000
					zscore     1.408     NaN     1.229

A:
<code>
import pandas as pd
import io
import numpy as np
from scipy import stats

temp=u"""probegenes,sample1,sample2,sample3
1415777_at Pnliprp1,20,0.00,11
1415805_at Clps,17,0.00,55
1415884_at Cela3b,47,0.00,100"""
df = pd.read_csv(io.StringIO(temp),index_col='probegenes')
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import pandas as pd
import io
import copy
from scipy import stats


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            temp = """probegenes,sample1,sample2,sample3
    1415777_at Pnliprp1,20,0.00,11
    1415805_at Clps,17,0.00,55
    1415884_at Cela3b,47,0.00,100"""
        elif test_case_id == 2:
            temp = """probegenes,sample1,sample2,sample3
    1415777_at Pnliprp1,20,2.00,11
    1415805_at Clps,17,0.30,55
    1415884_at Cela3b,47,1.00,100"""
        df = pd.read_csv(io.StringIO(temp), index_col="probegenes")
        return df

    def generate_ans(data):
        _a = data
        df = _a
        indices = [
            ("1415777_at Pnliprp1", "data"),
            ("1415777_at Pnliprp1", "zscore"),
            ("1415805_at Clps", "data"),
            ("1415805_at Clps", "zscore"),
            ("1415884_at Cela3b", "data"),
            ("1415884_at Cela3b", "zscore"),
        ]
        indices = pd.MultiIndex.from_tuples(indices)
        df2 = pd.DataFrame(
            data=stats.zscore(df, axis=0), index=df.index, columns=df.columns
        )
        df3 = pd.concat([df, df2], axis=1).to_numpy().reshape(-1, 3)
        result = pd.DataFrame(data=np.round(df3, 3), index=indices, columns=df.columns)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    pd.testing.assert_frame_equal(result, ans, atol=1e-3, check_dtype=False)
    return 1


exec_context = r"""
import pandas as pd
import io
import numpy as np
from scipy import stats
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
