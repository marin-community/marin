Problem:
I am trying to get count of special chars in column using Pandas.
But not getting desired output.
My .txt file is:
str
Aa
Bb
?? ?
x;
###


My Code is :
import pandas as pd
df=pd.read_csv('inn.txt',sep='\t')
def count_special_char(string):
    special_char = 0
    for i in range(len(string)):
        if(string[i].isalpha()):
            continue
        else:
            special_char = special_char + 1
df["new"]=df.apply(count_special_char, axis = 0)
print(df)


And the output is:
    str  new
0    Aa  NaN
1    Bb  NaN
2  ?? ?  NaN
3   ###  NaN
4   x;      Nan


Desired output is:
    str  new
0    Aa  NaN
1    Bb  NaN
2  ?? ?  4
3   ###  3
4   x;     1


How to go ahead on this ?


A:
<code>
import pandas as pd


df = pd.DataFrame({'str': ['Aa', 'Bb', '?? ?', '###', '{}xxa;']})
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
        df["new"] = df.apply(lambda p: sum(not q.isalpha() for q in p["str"]), axis=1)
        df["new"] = df["new"].replace(0, np.NAN)
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame({"str": ["Aa", "Bb", "?? ?", "###", "{}xxa;"]})
        if test_case_id == 2:
            df = pd.DataFrame({"str": ["Cc", "Dd", "!! ", "###%", "{}xxa;"]})
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
