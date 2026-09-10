Problem:

I'm trying to find the best hyper-parameters using sklearn function GridSearchCV on XGBoost.
However, I'd like it to do early stop when doing gridsearch, since this could reduce a lot of search time and might gain a better result on my tasks.
Actually, I am using XGBoost via its sklearn API.
    model = xgb.XGBRegressor()
    GridSearchCV(model, paramGrid, verbose=1, cv=TimeSeriesSplit(n_splits=3).get_n_splits([trainX, trainY]), n_jobs=n_jobs, iid=iid).fit(trainX, trainY)
I don't know how to add the early stopping parameters with fit_params. I tried, but then it throws this error which is basically because early stopping needs validation set and there is a lack of it:

So how can I apply GridSearch on XGBoost with using early_stopping_rounds?
note that I'd like to use params below
fit_params={"early_stopping_rounds":42,
            "eval_metric" : "mae",
            "eval_set" : [[testX, testY]]}

note: model is working without gridsearch, also GridSearch works without fit_params
How can I do that? Thanks.

A:

<code>
import numpy as np
import pandas as pd
import xgboost.sklearn as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
gridsearch, testX, testY, trainX, trainY = load_data()
assert type(gridsearch) == sklearn.model_selection._search.GridSearchCV
assert type(trainX) == list
assert type(trainY) == list
assert type(testX) == list
assert type(testY) == list
</code>
solve this question with example variable `gridsearch` and put score in `b`, put prediction in `c`
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
import xgboost.sklearn as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            trainX = [[1], [2], [3], [4], [5]]
            trainY = [1, 2, 3, 4, 5]
            testX, testY = trainX, trainY
            paramGrid = {"subsample": [0.5, 0.8]}
            model = xgb.XGBRegressor()
            gridsearch = GridSearchCV(
                model,
                paramGrid,
                cv=TimeSeriesSplit(n_splits=2).get_n_splits([trainX, trainY]),
            )
        return gridsearch, testX, testY, trainX, trainY

    def generate_ans(data):
        gridsearch, testX, testY, trainX, trainY = data
        fit_params = {
            "early_stopping_rounds": 42,
            "eval_metric": "mae",
            "eval_set": [[testX, testY]],
        }
        gridsearch.fit(trainX, trainY, **fit_params)
        b = gridsearch.score(trainX, trainY)
        c = gridsearch.predict(trainX)
        return b, c

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        np.testing.assert_allclose(result[0], ans[0])
        np.testing.assert_allclose(result[1], ans[1])
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
import xgboost.sklearn as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
gridsearch, testX, testY, trainX, trainY = test_input
[insert]
b = gridsearch.score(trainX, trainY)
c = gridsearch.predict(trainX)
result = (b, c)
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
