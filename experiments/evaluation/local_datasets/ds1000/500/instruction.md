Problem:
In order to get a numpy array from a list I make the following:
Suppose n = 12
np.array([i for i in range(0, n)])
And get:
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
Then I would like to make a (4,3) matrix from this array:
np.array([i for i in range(0, 12)]).reshape(4, 3)
and I get the following matrix:
array([[ 0,  1,  2],
       [ 3,  4,  5],
       [ 6,  7,  8],
       [ 9, 10, 11]])
But if I know that I will have 3 * n elements in the initial list how can I reshape my numpy array, because the following code
np.array([i for i in range(0,12)]).reshape(a.shape[0]/3,3)
Results in the error
TypeError: 'float' object cannot be interpreted as an integer
A:
<code>
import numpy as np
a = np.arange(12)
</code>
a = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.arange(12)
        elif test_case_id == 2:
            np.random.seed(42)
            n = np.random.randint(15, 20)
            a = np.random.rand(3 * n)
        return a

    def generate_ans(data):
        _a = data
        a = _a
        a = a.reshape(-1, 3)
        return a

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
a = test_input
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

Return only the Python code snippet that replaces [insert]:
- No explanations, Markdown, comments, prints, or extra text; no non-English output
- No imports, function/class definitions, or placeholders; only the missing logic
- Do not reassign/reset provided variables/inputs; use them to compute the result
- The snippet must be syntactically valid and run as-is in this context; deterministic (assume temperature=0)
- Do not read/write files or call external services
- Output only the code lines for [insert]; no Markdown fences, no explanations, no bullets.
