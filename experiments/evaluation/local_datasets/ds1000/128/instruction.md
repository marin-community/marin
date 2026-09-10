Problem:
I have a pandas Dataframe like below:
UserId    ProductId    Quantity
1         1            6
1         4            1
1         7            3
2         4            2
3         2            7
3         1            2


Now, I want to randomly select the 20% of rows of this DataFrame, using df.sample(n), set random_state=0 and change the value of the ProductId column of these rows to zero. I would also like to keep the indexes of the altered rows. So the resulting DataFrame would be:
UserId    ProductId    Quantity
1         1            6
1         4            1
1         7            3
2         0            2
3         2            7
3         0            2


A:
<code>
import pandas as pd


df = pd.DataFrame({'UserId': [1, 1, 1, 2, 3, 3],
                   'ProductId': [1, 4, 7, 4, 2, 1],
                   'Quantity': [6, 1, 3, 2, 7, 2]})
</code>
df = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import pandas as pd
import numpy as np
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        l = int(0.2 * len(df))
        dfupdate = df.sample(l, random_state=0)
        dfupdate.ProductId = 0
        df.update(dfupdate)
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "UserId": [1, 1, 1, 2, 3, 3],
                    "ProductId": [1, 4, 7, 4, 2, 1],
                    "Quantity": [6, 1, 3, 2, 7, 2],
                }
            )
        if test_case_id == 2:
            df = pd.DataFrame(
                {
                    "UserId": [1, 1, 1, 2, 3, 3, 1, 1, 4],
                    "ProductId": [1, 4, 7, 4, 2, 1, 5, 1, 4],
                    "Quantity": [6, 1, 3, 2, 7, 2, 9, 9, 6],
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
result = df
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
    assert "sample" in tokens

Return only the Python code snippet that replaces [insert]:
- No explanations, Markdown, comments, prints, or extra text; no non-English output
- No imports, function/class definitions, or placeholders; only the missing logic
- Do not reassign/reset provided variables/inputs; use them to compute the result
- The snippet must be syntactically valid and run as-is in this context; deterministic (assume temperature=0)
- Do not read/write files or call external services
- Output only the code lines for [insert]; no Markdown fences, no explanations, no bullets.
