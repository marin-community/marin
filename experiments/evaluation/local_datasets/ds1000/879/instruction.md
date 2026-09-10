Problem:

Given a list of variant length features, for example:

f = [
    ['t1'],
    ['t2', 't5', 't7'],
    ['t1', 't2', 't3', 't4', 't5'],
    ['t4', 't5', 't6']
]
where each sample has variant number of features and the feature dtype is str and already one hot.

In order to use feature selection utilities of sklearn, I have to convert the features to a 2D-array which looks like:

f
    t1  t2  t3  t4  t5  t6  t7
r1   0   1   1   1   1   1   1
r2   1   0   1   1   0   1   0
r3   0   0   0   0   0   1   1
r4   1   1   1   0   0   0   1
How could I achieve it via sklearn or numpy?

A:

<code>
import pandas as pd
import numpy as np
import sklearn
features = load_data()
</code>
new_features = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
import sklearn
from sklearn.preprocessing import MultiLabelBinarizer


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            features = [["f1", "f2", "f3"], ["f2", "f4", "f5", "f6"], ["f1", "f2"]]
        elif test_case_id == 2:
            features = [
                ["t1"],
                ["t2", "t5", "t7"],
                ["t1", "t2", "t3", "t4", "t5"],
                ["t4", "t5", "t6"],
            ]
        return features

    def generate_ans(data):
        features = data
        new_features = MultiLabelBinarizer().fit_transform(features)
        rows, cols = new_features.shape
        for i in range(rows):
            for j in range(cols):
                if new_features[i, j] == 1:
                    new_features[i, j] = 0
                else:
                    new_features[i, j] = 1
        return new_features

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))

    return test_input, expected_result


def exec_test(result, ans):
    try:
        np.testing.assert_equal(result, ans)
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
import sklearn
features = test_input
[insert]
result = new_features
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
