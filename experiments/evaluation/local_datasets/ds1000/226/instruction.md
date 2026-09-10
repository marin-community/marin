Problem:
i need to create a dataframe containing tuples from a series of dataframes arrays. What I need is the following:
I have dataframes a and b:
a = pd.DataFrame(np.array([[1, 2],[3, 4]]), columns=['one', 'two'])
b = pd.DataFrame(np.array([[5, 6],[7, 8]]), columns=['one', 'two'])
a:
   one  two
0    1    2
1    3    4
b: 
   one  two
0    5    6
1    7    8


I want to create a dataframe a_b in which each element is a tuple formed from the corresponding elements in a and b, i.e.
a_b = pd.DataFrame([[(1, 5), (2, 6)],[(3, 7), (4, 8)]], columns=['one', 'two'])
a_b: 
      one     two
0  (1, 5)  (2, 6)
1  (3, 7)  (4, 8)


Ideally i would like to do this with an arbitrary number of dataframes. 
I was hoping there was a more elegant way than using a for cycle
I'm using python 3


A:
<code>
import pandas as pd
import numpy as np

a = pd.DataFrame(np.array([[1, 2],[3, 4]]), columns=['one', 'two'])
b = pd.DataFrame(np.array([[5, 6],[7, 8]]), columns=['one', 'two'])
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import pandas as pd
import numpy as np
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def generate_ans(data):
        data = data
        a, b = data
        return pd.DataFrame(
            np.rec.fromarrays((a.values, b.values)).tolist(),
            columns=a.columns,
            index=a.index,
        )

    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = pd.DataFrame(np.array([[1, 2], [3, 4]]), columns=["one", "two"])
            b = pd.DataFrame(np.array([[5, 6], [7, 8]]), columns=["one", "two"])
        if test_case_id == 2:
            b = pd.DataFrame(np.array([[1, 2], [3, 4]]), columns=["one", "two"])
            a = pd.DataFrame(np.array([[5, 6], [7, 8]]), columns=["one", "two"])
        return a, b

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
a,b = test_input
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
    assert "for" not in tokens and "while" not in tokens

Return only the Python code snippet that replaces [insert]:
- No explanations, Markdown, comments, prints, or extra text; no non-English output
- No imports, function/class definitions, or placeholders; only the missing logic
- Do not reassign/reset provided variables/inputs; use them to compute the result
- The snippet must be syntactically valid and run as-is in this context; deterministic (assume temperature=0)
- Do not read/write files or call external services
- Output only the code lines for [insert]; no Markdown fences, no explanations, no bullets.
