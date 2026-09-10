Problem:
I need to rename only the first column in my dataframe, the issue is there are many columns with the same name (there is a reason for this), thus I cannot use the code in other examples online. Is there a way to use something specific that just isolates the first column?
I have tried to do something like this
df.rename(columns={df.columns[0]: 'Test'}, inplace=True)
However this then means that all columns with that same header are changed to 'Test', whereas I just want the first one to change.
I kind of need something like df.columns[0] = 'Test'  but this doesn't work.


A:
<code>
import pandas as pd


df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=list('ABA'))
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        return df.set_axis(["Test", *df.columns[1:]], axis=1, inplace=False)

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=list("ABA"))
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
df = test_input
[insert]
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
