Problem:
I'm sorry in advance if this is a duplicated question, I looked for this information but still couldn't find it.
Is it possible to get a numpy array (or python list) filled with the indexes of the elements in increasing order?
For instance, the array:
a = array([4, 1, 0, 8, 5, 2])
The indexes of the elements in increasing order would give :
0 --> 2
1 --> 1
2 --> 5
4 --> 0
5 --> 4
8 --> 3
result = [2,1,5,0,4,3]
Thanks in advance!
A:
<code>
import numpy as np
a = np.array([4, 1, 0, 8, 5, 2])
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
            a = np.array([4, 1, 0, 8, 5, 2])
        elif test_case_id == 2:
            np.random.seed(42)
            a = (np.random.rand(100) - 0.5) * 100
        return a

    def generate_ans(data):
        _a = data
        a = _a
        result = np.argsort(a)
        return result

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
