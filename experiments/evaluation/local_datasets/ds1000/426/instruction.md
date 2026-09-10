Problem:
I have integers and I would like to convert them to binary numpy arrays of length m. For example, say m = 4. Now 15 = 1111 in binary and so the output should be (1,1,1,1).  2 = 10 in binary and so the output should be (0,0,1,0). If m were 3 then 2 should be converted to (0,1,0).
I tried np.unpackbits(np.uint8(num)) but that doesn't give an array of the right length. For example,
np.unpackbits(np.uint8(15))
Out[5]: array([0, 0, 0, 0, 1, 1, 1, 1], dtype=uint8)
Pay attention that the integers might overflow, and they might be negative. For m = 4:
63 = 0b00111111, output should be (1,1,1,1)
-2 = 0b11111110, output should be (1,1,1,0)
I would like a method that worked for whatever m I have in the code. Given an n-element integer array, I want to process it as above to generate a (n, m) matrix.
A:
<code>
import numpy as np
a = np.array([1, 2, 3, 4, 5])
m = 6
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import pandas as pd
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.array([1, 2, 3, 4, 5])
            m = 6
        elif test_case_id == 2:
            np.random.seed(42)
            a = np.random.randint(-100, 100, (20,))
            m = np.random.randint(4, 6)
        elif test_case_id == 3:
            np.random.seed(20)
            a = np.random.randint(-1000, 1000, (20,))
            m = 15
        return a, m

    def generate_ans(data):
        _a = data
        a, m = _a
        result = (((a[:, None] & (1 << np.arange(m))[::-1])) > 0).astype(int)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
a, m = test_input
[insert]
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
