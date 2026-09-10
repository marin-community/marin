Problem:

Right now, I have my data in a 2 by 2 numpy array. If I was to use MinMaxScaler fit_transform on the array, it will normalize it column by column, whereas I wish to normalize the entire np array all together. Is there anyway to do that?

A:

<code>
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
np_array = load_data()
def Transform(a):
    # return the solution in this function
    # new_a = Transform(a)
    ### BEGIN SOLUTION

Execution context (contains [insert] placeholder):
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            X = np.array([[-1, 2], [-0.5, 6]])
        return X

    def generate_ans(data):
        X = data
        scaler = MinMaxScaler()
        X_one_column = X.reshape([-1, 1])
        result_one_column = scaler.fit_transform(X_one_column)
        result = result_one_column.reshape(X.shape)
        return result

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
from sklearn.preprocessing import MinMaxScaler
np_array = test_input
def Transform(a):
[insert]
transformed = Transform(np_array)
result = transformed
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
