Problem:

I have used sklearn for Cross-validation and want to do a more visual information with the values of each model.

The problem is, I can't only get the name of the templates.
Instead, the parameters always come altogether. How can I only retrieve the name of the models without its parameters?
Or does it mean that I have to create an external list for the names?

here I have a piece of code:

for model in models:
   scores = cross_val_score(model, X, y, cv=5)
   print(f'Name model: {model} , Mean score: {scores.mean()}')
But I also obtain the parameters:

Name model: LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False), Mean score: 0.8066782865537986
In fact I want to get the information this way:

Name Model: LinearRegression, Mean Score: 0.8066782865537986
Any ideas to do that? Thanks!

A:

<code>
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
model = LinearRegression()
</code>
model_name = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
from sklearn.linear_model import LinearRegression
import sklearn
from sklearn.svm import LinearSVC


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            model = LinearRegression()
        elif test_case_id == 2:
            model = LinearSVC()
        return model

    def generate_ans(data):
        model = data
        model_name = type(model).__name__
        return model_name

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
from sklearn.linear_model import LinearRegression
model = test_input
[insert]
result = model_name
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
