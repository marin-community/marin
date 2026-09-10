Problem:
I have a dataframe with one of its column having a list at each index. I want to concatenate these lists into one list. I am using 
ids = df.loc[0:index, 'User IDs'].values.tolist()


However, this results in 
['[1,2,3,4......]'] which is a string. Somehow each value in my list column is type str. I have tried converting using list(), literal_eval() but it does not work. The list() converts each element within a list into a string e.g. from [12,13,14...] to ['['1'',','2',','1',',','3'......]'].
How to concatenate pandas column with list values into one list? Kindly help out, I am banging my head on it for several hours. 


A:
<code>
import pandas as pd


df = pd.DataFrame(dict(col1=[[1, 2, 3]] * 2))
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
        return df.col1.sum()

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(dict(col1=[[1, 2, 3]] * 2))
        return df

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        assert result == ans
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
