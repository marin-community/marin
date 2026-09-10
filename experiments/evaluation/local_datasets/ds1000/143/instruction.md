Problem:
I have

df = pd.DataFrame.from_dict({'id': ['A', 'B', 'A', 'C', 'D', 'B', 'C'], 'val': [1,2,-3,1,5,6,-2], 'stuff':['12','23232','13','1234','3235','3236','732323']})

  id   stuff  val
0  A      12    1
1  B   23232    2
2  A      13   -3
3  C    1234    1
4  D    3235    5
5  B    3236    6
6  C  732323   -2
I'd like to get a running sum of val for each id, so the desired output looks like this:

  id   stuff  val  cumsum
0  A      12    1   1
1  B   23232    2   2
2  A      13   -3   -2
3  C    1234    1   1
4  D    3235    5   5
5  B    3236    6   8
6  C  732323   -2  -1
This is what I tried:

df['cumsum'] = df.groupby('id').cumsum(['val'])
and

df['cumsum'] = df.groupby('id').cumsum(['val'])
This is the error I get:

ValueError: Wrong number of items passed 0, placement implies 1

A:
<code>
import pandas as pd

df = pd.DataFrame.from_dict({'id': ['A', 'B', 'A', 'C', 'D', 'B', 'C'],
                             'val': [1,2,-3,1,5,6,-2],
                             'stuff':['12','23232','13','1234','3235','3236','732323']})
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
        df["cumsum"] = df.groupby("id")["val"].transform(pd.Series.cumsum)
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame.from_dict(
                {
                    "id": ["A", "B", "A", "C", "D", "B", "C"],
                    "val": [1, 2, -3, 1, 5, 6, -2],
                    "stuff": ["12", "23232", "13", "1234", "3235", "3236", "732323"],
                }
            )
        elif test_case_id == 2:
            np.random.seed(19260817)
            df = pd.DataFrame(
                {
                    "id": ["A", "B"] * 10 + ["C"] * 10,
                    "val": np.random.randint(0, 100, 30),
                }
            )
        elif test_case_id == 3:
            np.random.seed(19260817)
            df = pd.DataFrame(
                {
                    "id": np.random.choice(list("ABCDE"), 1000),
                    "val": np.random.randint(-1000, 1000, 1000),
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
result=df
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
