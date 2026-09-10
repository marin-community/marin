Problem:
I have two arrays A (len of 3.8million) and B (len of 20k). For the minimal example, lets take this case:
A = np.array([1,1,2,3,3,3,4,5,6,7,8,8])
B = np.array([1,2,8])
Now I want the resulting array to be:
C = np.array([3,3,3,4,5,6,7])
i.e. if any value in B is found in A, remove it from A, if not keep it.
I would like to know if there is any way to do it without a for loop because it is a lengthy array and so it takes long time to loop.
A:
<code>
import numpy as np
A = np.array([1,1,2,3,3,3,4,5,6,7,8,8])
B = np.array([1,2,8])
</code>
C = ... # put solution in this variable
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
            A = np.array([1, 1, 2, 3, 3, 3, 4, 5, 6, 7, 8, 8])
            B = np.array([1, 2, 8])
        elif test_case_id == 2:
            np.random.seed(42)
            A = np.random.randint(0, 10, (20,))
            B = np.random.randint(0, 10, (3,))
        return A, B

    def generate_ans(data):
        _a = data
        A, B = _a
        C = A[~np.in1d(A, B)]
        return C

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
A, B = test_input
[insert]
result = C
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
