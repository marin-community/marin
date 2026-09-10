Problem:
I have the following kind of strings in my column seen below. I would like to parse out everything after the last _ of each string, and if there is no _ then leave the string as-is. (as my below try will just exclude strings with no _)
so far I have tried below, seen here:  Python pandas: remove everything after a delimiter in a string . But it is just parsing out everything after first _
d6['SOURCE_NAME'] = d6['SOURCE_NAME'].str.split('_').str[0]
Here are some example strings in my SOURCE_NAME column.
Stackoverflow_1234
Stack_Over_Flow_1234
Stackoverflow
Stack_Overflow_1234


Expected:
Stackoverflow
Stack_Over_Flow
Stackoverflow
Stack_Overflow


any help would be appreciated.


A:
<code>
import pandas as pd


strs = ['Stackoverflow_1234',
        'Stack_Over_Flow_1234',
        'Stackoverflow',
        'Stack_Overflow_1234']
df = pd.DataFrame(data={'SOURCE_NAME': strs})
</code>
df = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        df["SOURCE_NAME"] = df["SOURCE_NAME"].str.rsplit("_", 1).str.get(0)
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            strs = [
                "Stackoverflow_1234",
                "Stack_Over_Flow_1234",
                "Stackoverflow",
                "Stack_Overflow_1234",
            ]
            df = pd.DataFrame(data={"SOURCE_NAME": strs})
        if test_case_id == 2:
            strs = [
                "Stackoverflow_4321",
                "Stack_Over_Flow_4321",
                "Stackoverflow",
                "Stack_Overflow_4321",
            ]
            df = pd.DataFrame(data={"SOURCE_NAME": strs})
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
