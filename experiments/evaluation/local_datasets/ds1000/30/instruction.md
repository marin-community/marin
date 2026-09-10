Problem:
Considering a simple df:
HeaderA | HeaderB | HeaderC 
    476      4365      457


Is there a way to rename all columns, for example to add to all columns an "X" in the end? 
HeaderAX | HeaderBX | HeaderCX 
    476      4365      457


I am concatenating multiple dataframes and want to easily differentiate the columns dependent on which dataset they came from. 
Or is this the only way?
df.rename(columns={'HeaderA': 'HeaderAX'}, inplace=True)


I have over 50 column headers and ten files; so the above approach will take a long time. 
Thank You


A:
<code>
import pandas as pd


df = pd.DataFrame(
    {'HeaderA': [476],
     'HeaderB': [4365],
     'HeaderC': [457]})
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
        return df.add_suffix("X")

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame({"HeaderA": [476], "HeaderB": [4365], "HeaderC": [457]})
        if test_case_id == 2:
            df = pd.DataFrame({"HeaderD": [114], "HeaderF": [4365], "HeaderG": [514]})
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
