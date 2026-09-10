Problem:

Is there any package in Python that does data transformation like Yeo-Johnson transformation to eliminate skewness of data? In R this could be done using caret package:

set.seed(1)
predictors = data.frame(x1 = rnorm(1000,
                                   mean = 5,
                                   sd = 2),
                        x2 = rexp(1000,
                                  rate=10))

require(caret)

trans = preProcess(predictors,
                   c("BoxCox", "center", "scale"))
predictorsTrans = data.frame(
      trans = predict(trans, predictors))
I know about sklearn, but I was unable to find functions to do Yeo-Johnson transformation.
How can I use sklearn to solve this?

A:

<code>
import numpy as np
import pandas as pd
import sklearn
data = load_data()
assert type(data) == np.ndarray
</code>
yeo_johnson_data = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
import tokenize, io
import sklearn
from sklearn.preprocessing import PowerTransformer


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            data = np.array([[1, 2], [3, 2], [4, 5]])
        return data

    def generate_ans(data):
        def ans1(data):
            pt = PowerTransformer(method="yeo-johnson")
            yeo_johnson_data = pt.fit_transform(data)
            return yeo_johnson_data

        def ans2(data):
            pt = PowerTransformer(method="yeo-johnson", standardize=False)
            yeo_johnson_data = pt.fit_transform(data)
            return yeo_johnson_data

        return ans1(copy.deepcopy(data)), ans2(copy.deepcopy(data))

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))

    return test_input, expected_result


def exec_test(result, ans):
    ret = 0
    try:
        np.testing.assert_allclose(result, ans[0])
        ret = 1
    except:
        pass
    try:
        np.testing.assert_allclose(result, ans[1])
        ret = 1
    except:
        pass
    return ret


exec_context = r"""
import pandas as pd
import numpy as np
import sklearn
data = test_input
[insert]
result = yeo_johnson_data
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
    assert "sklearn" in tokens

Return only the Python code snippet that replaces [insert]:
- No explanations, Markdown, comments, prints, or extra text; no non-English output
- No imports, function/class definitions, or placeholders; only the missing logic
- Do not reassign/reset provided variables/inputs; use them to compute the result
- The snippet must be syntactically valid and run as-is in this context; deterministic (assume temperature=0)
- Do not read/write files or call external services
- Output only the code lines for [insert]; no Markdown fences, no explanations, no bullets.
