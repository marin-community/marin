Problem:

>>> arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
>>> del_col = [1, 2, 4, 5]
>>> arr
array([[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12]])
I am deleting some columns(in this example, 1st, 2nd and 4th)
def_col = np.array([1, 2, 4, 5])
array([[ 3],
       [ 7],
       [ 11]])
Note that del_col might contain out-of-bound indices, so we should ignore them.
Are there any good way ? Please consider this to be a novice question.
A:
<code>
import numpy as np
a = np.arange(12).reshape(3, 4)
del_col = np.array([1, 2, 4, 5])
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.arange(12).reshape(3, 4)
            del_col = np.array([1, 2, 4, 5])
        elif test_case_id == 2:
            np.random.seed(42)
            a = np.random.rand(np.random.randint(5, 10), np.random.randint(6, 10))
            del_col = np.random.randint(0, 15, (4,))
        return a, del_col

    def generate_ans(data):
        _a = data
        a, del_col = _a
        mask = del_col <= a.shape[1]
        del_col = del_col[mask] - 1
        result = np.delete(a, del_col, axis=1)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
a, del_col = test_input
[insert]
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
