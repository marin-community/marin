Problem:
I do know some posts are quite similar to my question but none of them succeded in giving me the correct answer. I want, for each row of a pandas dataframe, to perform the sum of values taken from several columns. As the number of columns tends to vary, I want this sum to be performed from a list of columns.
At the moment my code looks like this:
df['Sum'] = df['Col A'] + df['Col E'] + df['Col Z']


I want it to be something like :
df['Sum'] = sum(list_of_my_columns)


or
df[list_of_my_columns].sum(axis=1)


But both of them return an error. Might be because my list isn't properly created? This is how I did it:
list_of_my_columns = [df['Col A'], df['Col E'], df['Col Z']]


But this doesn't seem to work... Any ideas ? Thank you !
A:
<code>
import pandas as pd
import numpy as np


np.random.seed(10)
data = {}
for i in [chr(x) for x in range(65,91)]:
    data['Col '+i] = np.random.randint(1,100,10)
df = pd.DataFrame(data)
list_of_my_columns = ['Col A', 'Col E', 'Col Z']
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
        data = data
        df, list_of_my_columns = data
        df["Sum"] = df[list_of_my_columns].sum(axis=1)
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(10)
            data = {}
            for i in [chr(x) for x in range(65, 91)]:
                data["Col " + i] = np.random.randint(1, 100, 10)
            df = pd.DataFrame(data)
            list_of_my_columns = ["Col A", "Col E", "Col Z"]
        return df, list_of_my_columns

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
df, list_of_my_columns = test_input
[insert]
result = df
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
