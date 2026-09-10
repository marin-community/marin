Problem:
Lists have a very simple method to insert elements:
a = [1,2,3,4]
a.insert(2,66)
print a
[1, 2, 66, 3, 4]
For a numpy array I could do:
a = np.asarray([1,2,3,4])
a_l = a.tolist()
a_l.insert(2,66)
a = np.asarray(a_l)
print a
[1 2 66 3 4]
but this is very convoluted.
Is there an insert equivalent for numpy arrays?
A:
<code>
import numpy as np
a = np.asarray([1,2,3,4])
pos = 2
element = 66
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
            a = np.asarray([1, 2, 3, 4])
            pos = 2
            element = 66
        elif test_case_id == 2:
            np.random.seed(42)
            a = np.random.rand(100)
            pos = np.random.randint(0, 99)
            element = 66
        return a, pos, element

    def generate_ans(data):
        _a = data
        a, pos, element = _a
        a = np.insert(a, pos, element)
        return a

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
a, pos, element = test_input
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
    assert "insert" in tokens

Return only the Python code snippet that replaces [insert]:
- No explanations, Markdown, comments, prints, or extra text; no non-English output
- No imports, function/class definitions, or placeholders; only the missing logic
- Do not reassign/reset provided variables/inputs; use them to compute the result
- The snippet must be syntactically valid and run as-is in this context; deterministic (assume temperature=0)
- Do not read/write files or call external services
- Output only the code lines for [insert]; no Markdown fences, no explanations, no bullets.
