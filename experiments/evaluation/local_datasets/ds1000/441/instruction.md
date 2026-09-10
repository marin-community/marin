Problem:
I just want to check if a numpy array contains a single number quickly similar to contains for a list. Is there a concise way to do this?
a = np.array(9,2,7,0)
a.contains(0)  == true
A:
<code>
import numpy as np
a = np.array([9, 2, 7, 0])
number = 0
</code>
is_contained = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import pandas as pd
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.array([9, 2, 7, 0])
            number = 0
        elif test_case_id == 2:
            a = np.array([1, 2, 3, 5])
            number = 4
        elif test_case_id == 3:
            a = np.array([1, 1, 1, 1])
            number = 1
        return a, number

    def generate_ans(data):
        _a = data
        a, number = _a
        is_contained = number in a
        return is_contained

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert result == ans
    return 1


exec_context = r"""
import numpy as np
a, number = test_input
[insert]
result = is_contained
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
