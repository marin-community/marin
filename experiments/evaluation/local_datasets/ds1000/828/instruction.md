Problem:

I have used the

sklearn.preprocessing.OneHotEncoder
to transform some data the output is scipy.sparse.csr.csr_matrix how can I merge it back into my original dataframe along with the other columns?

I tried to use pd.concat but I get

TypeError: cannot concatenate a non-NDFrame object
Thanks

A:

<code>
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
df_origin, transform_output = load_data()
</code>
df = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import pandas as pd
import copy
from scipy.sparse import csr_matrix


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            df_origin = pd.DataFrame([[1, 1, 4], [0, 3, 0]], columns=["A", "B", "C"])
            transform_output = csr_matrix([[1, 0, 2], [0, 3, 0]])
        elif test_case_id == 2:
            df_origin = pd.DataFrame(
                [[1, 1, 4, 5], [1, 4, 1, 9]], columns=["A", "B", "C", "D"]
            )
            transform_output = csr_matrix([[1, 9, 8, 1, 0], [1, 1, 4, 5, 1]])
        return df_origin, transform_output

    def generate_ans(data):
        df_origin, transform_output = data
        df = pd.concat([df_origin, pd.DataFrame(transform_output.toarray())], axis=1)
        return df

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        pd.testing.assert_frame_equal(result, ans, check_dtype=False, check_names=False)
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
df_origin, transform_output = test_input
[insert]
result = df
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
