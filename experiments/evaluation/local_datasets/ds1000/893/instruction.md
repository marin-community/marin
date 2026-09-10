Problem:

I have a csv file without headers which I'm importing into python using pandas. The last column is the target class, while the rest of the columns are pixel values for images. How can I go ahead and split this dataset into a training set and a testing set (80/20)?

Also, once that is done how would I also split each of those sets so that I can define x (all columns except the last one), and y (the last column)?

I've imported my file using:

dataset = pd.read_csv('example.csv', header=None, sep=',')
Thanks

A:

use random_state=42
<code>
import numpy as np
import pandas as pd
dataset = load_data()
</code>
x_train, x_test, y_train, y_test = ... # put solution in these variables
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import pandas as pd
import copy
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            data, target = load_iris(as_frame=True, return_X_y=True)
            dataset = pd.concat([data, target], axis=1)
        return dataset

    def generate_ans(data):
        dataset = data
        x_train, x_test, y_train, y_test = train_test_split(
            dataset.iloc[:, :-1], dataset.iloc[:, -1], test_size=0.2, random_state=42
        )
        return x_train, x_test, y_train, y_test

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))

    return test_input, expected_result


def exec_test(result, ans):
    try:
        pd.testing.assert_frame_equal(result[0], ans[0])
        pd.testing.assert_frame_equal(result[1], ans[1])
        pd.testing.assert_series_equal(result[2], ans[2])
        pd.testing.assert_series_equal(result[3], ans[3])
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
dataset = test_input
[insert]
result = (x_train, x_test, y_train, y_test)
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
