Problem:
What's the more pythonic way to pad an array with zeros at the end?
def pad(A, length):
    ...
A = np.array([1,2,3,4,5])
pad(A, 8)    # expected : [1,2,3,4,5,0,0,0]

In my real use case, in fact I want to pad an array to the closest multiple of 1024. Ex: 1342 => 2048, 3000 => 3072, so I want non-loop solution.
A:
<code>
import numpy as np
A = np.array([1,2,3,4,5])
length = 8
</code>
result = ... # put solution in this variable
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
            A = np.array([1, 2, 3, 4, 5])
            length = 8
        elif test_case_id == 2:
            np.random.seed(42)
            A = np.random.rand(10)
            length = np.random.randint(12, 18)
        return A, length

    def generate_ans(data):
        _a = data
        A, length = _a
        result = np.pad(A, (0, length - A.shape[0]), "constant")
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
A, length = test_input
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
    assert "while" not in tokens and "for" not in tokens

Return only the Python code snippet that replaces [insert]:
- No explanations, Markdown, comments, prints, or extra text; no non-English output
- No imports, function/class definitions, or placeholders; only the missing logic
- Do not reassign/reset provided variables/inputs; use them to compute the result
- The snippet must be syntactically valid and run as-is in this context; deterministic (assume temperature=0)
- Do not read/write files or call external services
- Output only the code lines for [insert]; no Markdown fences, no explanations, no bullets.
