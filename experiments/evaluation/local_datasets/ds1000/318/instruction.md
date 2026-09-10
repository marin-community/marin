Problem:
Is there a way to change the order of the columns in a numpy 2D array to a new and arbitrary order? For example, I have an array `a`:
array([[10, 20, 30, 40, 50],
       [ 6,  7,  8,  9, 10]])
and I want to change it into, say
array([[10, 30, 50, 40, 20],
       [ 6,  8, 10,  9,  7]])
by applying the permutation
0 -> 0
1 -> 4
2 -> 1
3 -> 3
4 -> 2
on the columns. In the new matrix, I therefore want the first column of the original to stay in place, the second to move to the last column and so on.
Is there a numpy function to do it? I have a fairly large matrix and expect to get even larger ones, so I need a solution that does this quickly and in place if possible (permutation matrices are a no-go)
Thank you.
A:
<code>
import numpy as np
a = np.array([[10, 20, 30, 40, 50],
       [ 6,  7,  8,  9, 10]])
permutation = [0, 4, 1, 3, 2]
</code>
a = ... # put solution in this variable
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
            a = np.array([[10, 20, 30, 40, 50], [6, 7, 8, 9, 10]])
            permutation = [0, 4, 1, 3, 2]
        elif test_case_id == 2:
            np.random.seed(42)
            a = np.random.rand(np.random.randint(5, 10), np.random.randint(6, 10))
            permutation = np.arange(a.shape[1])
            np.random.shuffle(permutation)
        return a, permutation

    def generate_ans(data):
        _a = data
        a, permutation = _a
        c = np.empty_like(permutation)
        c[permutation] = np.arange(len(permutation))
        a = a[:, c]
        return a

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
a, permutation = test_input
[insert]
result = a
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
