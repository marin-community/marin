Problem:

I am new to scikit-learn, but it did what I was hoping for. Now, maddeningly, the only remaining issue is that I don't find how I could print the model's coefficients it estimated. Especially when it comes to a pipeline fitted by a GridSearch. Now I have a pipeline including data scaling, centering, and a classifier model. What is the way to get its estimated coefficients?
here is my current code
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", SGDClassifier(random_state=42))
])
grid = GridSearchCV(pipe, param_grid={"model__alpha": [1e-3, 1e-2, 1e-1, 1]}, cv=5)
# where is the coef?

Any advice is appreciated. Thanks in advance.


A:

runnable code
<code>
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
X, y = load_data()
assert type(X) == np.ndarray
assert type(y) == np.ndarray
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", SGDClassifier(random_state=42))
])
grid = GridSearchCV(pipe, param_grid={"model__alpha": [1e-3, 1e-2, 1e-1, 1]}, cv=5)
</code>
coef = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sklearn
from sklearn.datasets import make_classification


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            X, y = make_classification(random_state=42)
        return X, y

    def generate_ans(data):
        X, y = data
        pipe = Pipeline(
            [("scale", StandardScaler()), ("model", SGDClassifier(random_state=42))]
        )
        grid = GridSearchCV(
            pipe, param_grid={"model__alpha": [1e-3, 1e-2, 1e-1, 1]}, cv=5
        )
        grid.fit(X, y)
        coef = grid.best_estimator_.named_steps["model"].coef_
        return coef

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
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", SGDClassifier(random_state=42))
])
grid = GridSearchCV(pipe, param_grid={"model__alpha": [1e-3, 1e-2, 1e-1, 1]}, cv=5)
X, y = test_input
[insert]
result = coef
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
