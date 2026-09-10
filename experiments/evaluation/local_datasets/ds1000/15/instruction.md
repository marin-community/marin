Problem:
I have a data set like below:
name    status    number   message
matt    active    12345    [job:  , money: none, wife: none]
james   active    23456    [group: band, wife: yes, money: 10000]
adam    inactive  34567    [job: none, money: none, wife:  , kids: one, group: jail]


How can I extract the key value pairs, and turn them into a dataframe expanded all the way out?

Expected output: 
name    status   number    job    money    wife    group   kids 
matt    active   12345     none   none     none    none    none
james   active   23456     none   10000    none    band    none
adam    inactive 34567     none   none     none    none    one

Notice: 'none' is a string
The message contains multiple different key types. 
Any help would be greatly appreciated. 


A:
<code>
import pandas as pd


df = pd.DataFrame({'name': ['matt', 'james', 'adam'],
                   'status': ['active', 'active', 'inactive'],
                   'number': [12345, 23456, 34567],
                   'message': ['[job:  , money: none, wife: none]',
                               '[group: band, wife: yes, money: 10000]',
                               '[job: none, money: none, wife:  , kids: one, group: jail]']})
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import pandas as pd
import numpy as np
import yaml
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        df.message = df.message.replace(["\[", "\]"], ["{", "}"], regex=True).apply(
            yaml.safe_load
        )
        df1 = pd.DataFrame(df.pop("message").values.tolist(), index=df.index)
        result = pd.concat([df, df1], axis=1)
        result = result.replace("", "none")
        result = result.replace(np.nan, "none")
        return result

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "name": ["matt", "james", "adam"],
                    "status": ["active", "active", "inactive"],
                    "number": [12345, 23456, 34567],
                    "message": [
                        "[job:  , money: none, wife: none]",
                        "[group: band, wife: yes, money: 10000]",
                        "[job: none, money: none, wife:  , kids: one, group: jail]",
                    ],
                }
            )
        if test_case_id == 2:
            df = pd.DataFrame(
                {
                    "name": ["matt", "james", "adam"],
                    "status": ["active", "active", "inactive"],
                    "number": [12345, 23456, 34567],
                    "message": [
                        "[job:  , money: 114514, wife: none, kids: one, group: jail]",
                        "[group: band, wife: yes, money: 10000]",
                        "[job: none, money: none, wife:  , kids: one, group: jail]",
                    ],
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
df = test_input
[insert]
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
