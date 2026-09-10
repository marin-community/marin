Problem:
Say I have a 3 dimensional numpy array:
np.random.seed(1145)
A = np.random.random((5,5,5))
and I have two lists of indices corresponding to the 2nd and 3rd dimensions:
second = [1,2]
third = [3,4]
and I want to select the elements in the numpy array corresponding to
A[:][second][third]
so the shape of the sliced array would be (5,2,2) and
A[:][second][third].flatten()
would be equivalent to to:
In [226]:
for i in range(5):
    for j in second:
        for k in third:
            print A[i][j][k]
0.556091074129
0.622016249651
0.622530505868
0.914954716368
0.729005532319
0.253214472335
0.892869371179
0.98279375528
0.814240066639
0.986060321906
0.829987410941
0.776715489939
0.404772469431
0.204696635072
0.190891168574
0.869554447412
0.364076117846
0.04760811817
0.440210532601
0.981601369658
Is there a way to slice a numpy array in this way? So far when I try A[:][second][third] I get IndexError: index 3 is out of bounds for axis 0 with size 2 because the [:] for the first dimension seems to be ignored.
A:
<code>
import numpy as np
a = np.random.rand(5, 5, 5)
second = [1, 2]
third = [3, 4]
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
            np.random.seed(42)
            a = np.random.rand(5, 5, 5)
            second = [1, 2]
            third = [3, 4]
        elif test_case_id == 2:
            np.random.seed(45)
            a = np.random.rand(7, 8, 9)
            second = [0, 4]
            third = [6, 7]
        return a, second, third

    def generate_ans(data):
        _a = data
        a, second, third = _a
        result = a[:, np.array(second).reshape(-1, 1), third]
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
a, second, third = test_input
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
