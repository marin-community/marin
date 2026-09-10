Problem:

How can I perform regression in sklearn, using SVM and a polynomial kernel (degree=2)?
Note to use default arguments. Thanks.

A:

<code>
import numpy as np
import pandas as pd
import sklearn
X, y = load_data()
assert type(X) == np.ndarray
assert type(y) == np.ndarray
# fit, then predict X
</code>
predict = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
import sklearn
from sklearn.datasets import make_regression
from sklearn.svm import SVR


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            X, y = make_regression(n_samples=1000, n_features=4, random_state=42)
        return X, y

    def generate_ans(data):
        X, y = data
        svr_poly = SVR(kernel="poly", degree=2)
        svr_poly.fit(X, y)
        predict = svr_poly.predict(X)
        return predict

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))

    return test_input, expected_result


def exec_test(result, ans):
    try:
        np.testing.assert_allclose(result, ans)
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
import sklearn
X, y = test_input
[insert]
result = predict
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
