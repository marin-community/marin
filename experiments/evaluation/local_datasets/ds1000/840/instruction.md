Problem:

I want to get the probability of the Logistic Regression model, while use cross-validation.
But now I'm only able to get the scores of the model, can u help me to get the probabilities?
please save the probabilities into a list or an array. thanks.

A:

<code>
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
X, y = load_data()
assert type(X) == np.ndarray
assert type(y) == np.ndarray
cv = StratifiedKFold(5).split(X, y)
logreg = LogisticRegression()
</code>
proba = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import sklearn
from sklearn import datasets


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            iris = datasets.load_iris()
            X = iris.data
            y = iris.target
        return X, y

    def generate_ans(data):
        def ans1(data):
            X, y = data
            cv = StratifiedKFold(5).split(X, y)
            logreg = LogisticRegression(random_state=42)
            proba = cross_val_predict(logreg, X, y, cv=cv, method="predict_proba")
            return proba

        def ans2(data):
            X, y = data
            cv = StratifiedKFold(5).split(X, y)
            logreg = LogisticRegression(random_state=42)
            proba = []
            for train, test in cv:
                logreg.fit(X[train], y[train])
                proba.append(logreg.predict_proba(X[test]))
            return proba

        return ans1(copy.deepcopy(data)), ans2(copy.deepcopy(data))

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))

    return test_input, expected_result


def exec_test(result, ans):
    ret = 0
    try:
        np.testing.assert_allclose(result, ans[0], rtol=1e-3)
        ret = 1
    except:
        pass
    try:
        np.testing.assert_allclose(result, ans[1], rtol=1e-3)
        ret = 1
    except:
        pass
    return ret


exec_context = r"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
X, y = test_input
cv = StratifiedKFold(5).split(X, y)
logreg = LogisticRegression(random_state=42)
[insert]
result = proba
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
