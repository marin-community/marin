Problem:
I want to generate a random array of size N which only contains 0 and 1, I want my array to have some ratio between 0 and 1. For example, 90% of the array be 1 and the remaining 10% be 0 (I want this 90% to be random along with the whole array).
right now I have:
randomLabel = np.random.randint(2, size=numbers)
But I can't control the ratio between 0 and 1.
A:
<code>
import numpy as np
one_ratio = 0.9
size = 1000
</code>
nums = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import pandas as pd
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            one_ratio = 0.9
            size = 1000
        elif test_case_id == 2:
            size = 100
            one_ratio = 0.8
        return size, one_ratio

    def generate_ans(data):
        _a = data
        return _a

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    size, one_ratio = ans
    assert result.shape == (size,)
    assert abs(len(np.where(result == 1)[0]) - size * one_ratio) / size <= 0.05
    assert abs(len(np.where(result == 0)[0]) - size * (1 - one_ratio)) / size <= 0.05
    return 1


exec_context = r"""
import numpy as np
size, one_ratio = test_input
[insert]
result = nums
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
    assert "random" in tokens

Return only the Python code snippet that replaces [insert]:
- No explanations, Markdown, comments, prints, or extra text; no non-English output
- No imports, function/class definitions, or placeholders; only the missing logic
- Do not reassign/reset provided variables/inputs; use them to compute the result
- The snippet must be syntactically valid and run as-is in this context; deterministic (assume temperature=0)
- Do not read/write files or call external services
- Output only the code lines for [insert]; no Markdown fences, no explanations, no bullets.
