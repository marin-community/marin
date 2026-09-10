Problem:

How do I convert data from a Scikit-learn Bunch object (from sklearn.datasets) to a Pandas DataFrame?

from sklearn.datasets import fetch_california_housing
import pandas as pd
data = fetch_california_housing()
print(type(data))
data1 = pd. # Is there a Pandas method to accomplish this?

A:

<code>
import numpy as np
from sklearn.datasets import fetch_california_housing
import pandas as pd
data = load_data()
</code>
data1 = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import pandas as pd
import copy
from sklearn.datasets import fetch_california_housing


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            data = fetch_california_housing()
        return data

    def generate_ans(data):
        data = data
        data1 = pd.DataFrame(data.data, columns=data.feature_names)
        data1["target"] = pd.Series(data.target)
        return data1

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        for c in ans.columns:
            pd.testing.assert_series_equal(result[c], ans[c], check_dtype=False)
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
data = test_input
[insert]
result = data1
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
