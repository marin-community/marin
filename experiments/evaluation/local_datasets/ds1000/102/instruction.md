Problem:
In pandas, how do I replace &AMP; with '&' from all columns where &AMP could be in any position in a string?
For example, in column Title if there is a value 'Good &AMP; bad', how do I replace it with 'Good & bad'?


A:
<code>
import pandas as pd

example_df = pd.DataFrame({'A': ['Good &AMP; bad', 'BB', 'CC', 'DD', 'Good &AMP; bad'], 'B': range(5), 'C': ['Good &AMP; bad'] * 5})
def f(df=example_df):
    # return the solution in this function
    # result = f(df)
    ### BEGIN SOLUTION

Execution context (contains [insert] placeholder):
import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        return df.replace("&AMP;", "&", regex=True)

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "A": ["Good &AMP; bad", "BB", "CC", "DD", "Good &AMP; bad"],
                    "B": range(5),
                    "C": ["Good &AMP; bad"] * 5,
                }
            )
        return df

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        pd.testing.assert_frame_equal(result, ans, check_dtype=False)
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
def f(df):
[insert]
df = test_input
result = f(df)
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
