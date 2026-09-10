Problem:

I would like to break down a pandas column, which is the last column, consisting of a list of elements into as many columns as there are unique elements i.e. one-hot-encode them (with value 1 representing a given element existing in a row and 0 in the case of absence).

For example, taking dataframe df

Col1   Col2    Col3          Col4
 C      33      11       [Apple, Orange, Banana]
 A      2.5     4.5      [Apple, Grape]
 B      42      14       [Banana]
 D      666     1919810  [Suica, Orange]
I would like to convert this to:

df

Col1 Col2     Col3  Apple  Banana  Grape  Orange  Suica
C   33       11      1       1      0       1      0
A  2.5      4.5      1       0      1       0      0
B   42       14      0       1      0       0      0
D  666  1919810      0       0      0       1      1
How can I use pandas/sklearn to achieve this?

A:

<code>
import pandas as pd
import numpy as np
import sklearn
df = load_data()
</code>
df_out = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import pandas as pd
import copy
import sklearn
from sklearn.preprocessing import MultiLabelBinarizer


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "Col1": ["C", "A", "B", "D"],
                    "Col2": ["33", "2.5", "42", "666"],
                    "Col3": ["11", "4.5", "14", "1919810"],
                    "Col4": [
                        ["Apple", "Orange", "Banana"],
                        ["Apple", "Grape"],
                        ["Banana"],
                        ["Suica", "Orange"],
                    ],
                }
            )
        elif test_case_id == 2:
            df = pd.DataFrame(
                {
                    "Col1": ["c", "a", "b", "d"],
                    "Col2": ["33", "2.5", "42", "666"],
                    "Col3": ["11", "4.5", "14", "1919810"],
                    "Col4": [
                        ["Apple", "Orange", "Banana"],
                        ["Apple", "Grape"],
                        ["Banana"],
                        ["Suica", "Orange"],
                    ],
                }
            )
        elif test_case_id == 3:
            df = pd.DataFrame(
                {
                    "Col1": ["C", "A", "B", "D"],
                    "Col2": ["33", "2.5", "42", "666"],
                    "Col3": ["11", "4.5", "14", "1919810"],
                    "Col4": [
                        ["Apple", "Orange", "Banana", "Watermelon"],
                        ["Apple", "Grape"],
                        ["Banana"],
                        ["Suica", "Orange"],
                    ],
                }
            )
        return df

    def generate_ans(data):
        df = data
        mlb = MultiLabelBinarizer()
        df_out = df.join(
            pd.DataFrame(
                mlb.fit_transform(df.pop("Col4")), index=df.index, columns=mlb.classes_
            )
        )
        return df_out

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))

    return test_input, expected_result


def exec_test(result, ans):
    try:
        for i in range(2):
            pd.testing.assert_series_equal(
                result.iloc[:, i], ans.iloc[:, i], check_dtype=False
            )
    except:
        return 0
    try:
        for c in ans.columns:
            pd.testing.assert_series_equal(result[c], ans[c], check_dtype=False)
        return 1
    except:
        pass
    try:
        for c in ans.columns:
            ans[c] = ans[c].replace(1, 2).replace(0, 1).replace(2, 0)
            pd.testing.assert_series_equal(result[c], ans[c], check_dtype=False)
        return 1
    except:
        pass
    return 0


exec_context = r"""
import pandas as pd
import numpy as np
import sklearn
df = test_input
[insert]
result = df_out
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(3):
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
