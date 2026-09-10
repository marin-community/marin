Problem:
I have a dataframe with numerous columns (≈30) from an external source (csv file) but several of them have no value or always the same. Thus, I would to see quickly the value_counts for each column. How can i do that?
For example
  id, temp, name
1 34, null, mark
2 22, null, mark
3 34, null, mark


Please return a Series like this:


id    22      1.0
      34      2.0
temp  null    3.0
name  mark    3.0
dtype: float64


So I would know that temp is irrelevant and name is not interesting (always the same)


A:
<code>
import pandas as pd


df = pd.DataFrame(data=[[34, 'null', 'mark'], [22, 'null', 'mark'], [34, 'null', 'mark']], columns=['id', 'temp', 'name'], index=[1, 2, 3])
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
        return df.apply(lambda x: x.value_counts()).T.stack()

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                data=[[34, "null", "mark"], [22, "null", "mark"], [34, "null", "mark"]],
                columns=["id", "temp", "name"],
                index=[1, 2, 3],
            )
        if test_case_id == 2:
            df = pd.DataFrame(
                data=[
                    [34, "null", "mark"],
                    [22, "null", "mark"],
                    [34, "null", "mark"],
                    [21, "null", "mark"],
                ],
                columns=["id", "temp", "name"],
                index=[1, 2, 3, 4],
            )
        return df

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        pd.testing.assert_series_equal(result, ans)
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
