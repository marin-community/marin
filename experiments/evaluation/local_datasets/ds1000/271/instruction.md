Problem:
pandas version: 1.2
I have a dataframe that columns as 'float64' with null values represented as pd.NAN. Is there way to round without converting to string then decimal:
df = pd.DataFrame([(.21, .3212), (.01, .61237), (.66123, .03), (.21, .18),(pd.NA, .18)],
                  columns=['dogs', 'cats'])
df
      dogs     cats
0     0.21  0.32120
1     0.01  0.61237
2  0.66123  0.03000
3     0.21  0.18000
4     <NA>  0.18000


Here is what I wanted to do, but it is erroring:
df['dogs'] = df['dogs'].round(2)


TypeError: float() argument must be a string or a number, not 'NAType'


Here is another way I tried but this silently fails and no conversion occurs:
tn.round({'dogs': 1})
      dogs     cats
0     0.21  0.32120
1     0.01  0.61237
2  0.66123  0.03000
3     0.21  0.18000
4     <NA>  0.18000


A:
<code>
import pandas as pd


df = pd.DataFrame([(.21, .3212), (.01, .61237), (.66123, .03), (.21, .18),(pd.NA, .18)],
                  columns=['dogs', 'cats'])
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
        df["dogs"] = df["dogs"].apply(lambda x: round(x, 2) if str(x) != "<NA>" else x)
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                [
                    (0.21, 0.3212),
                    (0.01, 0.61237),
                    (0.66123, 0.03),
                    (0.21, 0.18),
                    (pd.NA, 0.18),
                ],
                columns=["dogs", "cats"],
            )
        if test_case_id == 2:
            df = pd.DataFrame(
                [
                    (0.215, 0.3212),
                    (0.01, 0.11237),
                    (0.36123, 0.03),
                    (0.21, 0.18),
                    (pd.NA, 0.18),
                ],
                columns=["dogs", "cats"],
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

Return only the Python code snippet that replaces [insert]:
- No explanations, Markdown, comments, prints, or extra text; no non-English output
- No imports, function/class definitions, or placeholders; only the missing logic
- Do not reassign/reset provided variables/inputs; use them to compute the result
- The snippet must be syntactically valid and run as-is in this context; deterministic (assume temperature=0)
- Do not read/write files or call external services
- Output only the code lines for [insert]; no Markdown fences, no explanations, no bullets.
