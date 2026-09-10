Problem:
What is the most efficient way to remove real numbers in a complex array? I have tried numpy.delete and Remove all specific value from array and code of the form x[x != i].
For:
import numpy as np
x = np.array([-2+1j, -1.4, -1.1, 0, 1.2, 2.2+2j, 3.1, 4.4, 8.3, 9.9, 10+0j, 14, 16.2])
I want to end up with an array:
[-2+1j, 2.2+2j]
A:
<code>
import numpy as np
x = np.array([-2+1j, -1.4, -1.1, 0, 1.2, 2.2+2j, 3.1, 4.4, 8.3, 9.9, 10+0j, 14, 16.2])
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
            x = np.array(
                [
                    -2 + 1j,
                    -1.4,
                    -1.1,
                    0,
                    1.2,
                    2.2 + 2j,
                    3.1,
                    4.4,
                    8.3,
                    9.9,
                    10 + 0j,
                    14,
                    16.2,
                ]
            )
        elif test_case_id == 2:
            np.random.seed(42)
            x = np.random.rand(10) - 0.5
            x = x.astype(np.complex128)
            x[[2, 5]] = -1.1 + 2j
        return x

    def generate_ans(data):
        _a = data
        x = _a
        result = x[x.imag != 0]
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
x = test_input
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
