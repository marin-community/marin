Problem:
I have a dataset :
id    url     keep_if_dup
1     A.com   Yes
2     A.com   Yes
3     B.com   No
4     B.com   No
5     C.com   No


I want to remove duplicates, i.e. keep last occurence of "url" field, BUT keep duplicates if the field "keep_if_dup" is YES.
Expected output :
id    url     keep_if_dup
1     A.com   Yes
2     A.com   Yes
4     B.com   No
5     C.com   No


What I tried :
Dataframe=Dataframe.drop_duplicates(subset='url', keep='first')


which of course does not take into account "keep_if_dup" field. Output is :
id    url     keep_if_dup
1     A.com   Yes
3     B.com   No
5     C.com   No


A:
<code>
import pandas as pd


df = pd.DataFrame({'url': ['A.com', 'A.com', 'A.com', 'B.com', 'B.com', 'C.com', 'B.com'],
                   'keep_if_dup': ['Yes', 'Yes', 'No', 'No', 'No', 'No', 'Yes']})
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
        return df.loc[(df["keep_if_dup"] == "Yes") | ~df["url"].duplicated(keep="last")]

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "url": [
                        "A.com",
                        "A.com",
                        "A.com",
                        "B.com",
                        "B.com",
                        "C.com",
                        "B.com",
                    ],
                    "keep_if_dup": ["Yes", "Yes", "No", "No", "No", "No", "Yes"],
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
