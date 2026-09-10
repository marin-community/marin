Problem:

Given the following example:

from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pandas as pd

pipe = Pipeline(steps=[
    ('select', SelectKBest(k=2)),
    ('clf', LogisticRegression())]
)

pipe.fit(data, target)
I would like to get intermediate data state in scikit learn pipeline corresponding to 'select' output (after fit_transform on 'select' but not LogisticRegression). Or to say things in another way, it would be the same than to apply

SelectKBest(k=2).fit_transform(data, target)
Any ideas to do that?

A:

<code>
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pandas as pd

data, target = load_data()

pipe = Pipeline(steps=[
    ('select', SelectKBest(k=2)),
    ('clf', LogisticRegression())]
)
</code>
select_out = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
import tokenize, io
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import sklearn
from sklearn.datasets import load_iris


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            iris = load_iris()
        return iris.data, iris.target

    def generate_ans(data):
        data, target = data
        pipe = Pipeline(
            steps=[("select", SelectKBest(k=2)), ("clf", LogisticRegression())]
        )
        select_out = pipe.named_steps["select"].fit_transform(data, target)
        return select_out

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
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
data, target = test_input
pipe = Pipeline(steps=[
    ('select', SelectKBest(k=2)),
    ('clf', LogisticRegression())]
)
[insert]
result = select_out
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)


def test_string(solution: str):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(solution.encode("utf-8")).readline):
        tokens.append(token.string)
    assert "SelectKBest" not in tokens

Return only the Python code snippet that replaces [insert]:
- No explanations, Markdown, comments, prints, or extra text; no non-English output
- No imports, function/class definitions, or placeholders; only the missing logic
- Do not reassign/reset provided variables/inputs; use them to compute the result
- The snippet must be syntactically valid and run as-is in this context; deterministic (assume temperature=0)
- Do not read/write files or call external services
- Output only the code lines for [insert]; no Markdown fences, no explanations, no bullets.
