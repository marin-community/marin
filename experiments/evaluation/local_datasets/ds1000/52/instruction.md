Problem:
Sample dataframe:
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 0]})

I'd like to add inverses of each existing column to the dataframe and name them based on existing column names with a prefix, e.g. inv_A is an inverse of column A and so on.
Notice that 0 has no inverse and please keep it in inv_A
The resulting dataframe should look like so:
result = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 0], "inv_A": [1/1, 1/2, 1/3], "inv_B": [1/4, 1/5, 0]})

Obviously there are redundant methods like doing this in a loop, but there should exist much more pythonic ways of doing it and after searching for some time I didn't find anything. I understand that this is most probably a duplicate; if so, please point me to an existing answer.

A:
<code>
import pandas as pd


df = pd.DataFrame({"A": [1, 0, 3], "B": [4, 5, 6]})
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import pandas as pd
import numpy as np
import math
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        return df.join(df.apply(lambda x: 1 / x).add_prefix("inv_")).replace(
            math.inf, 0
        )

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        if test_case_id == 2:
            df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
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


def test_string(solution: str):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(solution.encode("utf-8")).readline):
        tokens.append(token.string)
    assert "while" not in tokens and "for" not in tokens

Return only the Python code snippet that replaces [insert]:
- No explanations, Markdown, comments, prints, or extra text; no non-English output
- No imports, function/class definitions, or placeholders; only the missing logic
- Do not reassign/reset provided variables/inputs; use them to compute the result
- The snippet must be syntactically valid and run as-is in this context; deterministic (assume temperature=0)
- Do not read/write files or call external services
- Output only the code lines for [insert]; no Markdown fences, no explanations, no bullets.
