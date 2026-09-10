Problem:
I need to do some analysis on a large dataset from a hydrolgeology field work. I am using NumPy. I want to know how I can:
1.	multiply e.g. the col-th column of my array by a number (e.g. 5.2). And then
2.	calculate the cumulative sum of the numbers in that column.
As I mentioned I only want to work on a specific column and not the whole array.The result should be an 1-d array --- the cumulative sum.
A:
<code>
import numpy as np
a = np.random.rand(8, 5)
col = 2
multiply_number = 5.2
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
            np.random.seed(42)
            a = np.random.rand(8, 5)
            col = 2
            const = 5.2
        elif test_case_id == 2:
            np.random.seed(42)
            a = np.random.rand(np.random.randint(5, 10), np.random.randint(6, 10))
            col = 4
            const = np.random.rand()
        return a, col, const

    def generate_ans(data):
        _a = data
        a, col, multiply_number = _a
        a[:, col - 1] *= multiply_number
        result = np.cumsum(a[:, col - 1])
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
a, col, multiply_number = test_input
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
