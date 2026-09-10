Problem:

I have a pandas DataFrame data
it has about 12k rows and more than 500 columns, each column has its unique name
However, when I used sklearn preprocessing, I found the result lose the information about the columns
Here's the code

from sklearn import preprocessing
preprocessing.scale(data)
outputs a numpy array.

So my question is, how to apply preprocessing.scale to DataFrames, and don't lose the information(index, columns)?


A:

<code>
import numpy as np
import pandas as pd
from sklearn import preprocessing
data = load_data()
</code>
df_out = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import pandas as pd
import copy
from sklearn import preprocessing


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(42)
            data = pd.DataFrame(
                np.random.rand(3, 3),
                index=["first", "second", "third"],
                columns=["c1", "c2", "c3"],
            )
        return data

    def generate_ans(data):
        data = data
        df_out = pd.DataFrame(
            preprocessing.scale(data), index=data.index, columns=data.columns
        )
        return df_out

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        pd.testing.assert_frame_equal(result, ans, check_dtype=False, check_exact=False)
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
data = test_input
[insert]
result = df_out
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
