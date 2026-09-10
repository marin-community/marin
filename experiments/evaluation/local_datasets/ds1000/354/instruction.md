Problem:
Say I have these 2D arrays A and B.
How can I get elements from A that are not in B, and those from B that are not in A? (Symmetric difference in set theory: A△B)
Example:
A=np.asarray([[1,1,1], [1,1,2], [1,1,3], [1,1,4]])
B=np.asarray([[0,0,0], [1,0,2], [1,0,3], [1,0,4], [1,1,0], [1,1,1], [1,1,4]])
#elements in A first, elements in B then. in original order.
#output = array([[1,1,2], [1,1,3], [0,0,0], [1,0,2], [1,0,3], [1,0,4], [1,1,0]])

A:
<code>
import numpy as np
A=np.asarray([[1,1,1], [1,1,2], [1,1,3], [1,1,4]])
B=np.asarray([[0,0,0], [1,0,2], [1,0,3], [1,0,4], [1,1,0], [1,1,1], [1,1,4]])
</code>
output = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import pandas as pd
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            A = np.asarray([[1, 1, 1], [1, 1, 2], [1, 1, 3], [1, 1, 4]])
            B = np.asarray(
                [
                    [0, 0, 0],
                    [1, 0, 2],
                    [1, 0, 3],
                    [1, 0, 4],
                    [1, 1, 0],
                    [1, 1, 1],
                    [1, 1, 4],
                ]
            )
        elif test_case_id == 2:
            np.random.seed(42)
            A = np.random.randint(0, 2, (10, 3))
            B = np.random.randint(0, 2, (20, 3))
        elif test_case_id == 3:
            A = np.asarray([[1, 1, 1], [1, 1, 4]])
            B = np.asarray(
                [
                    [0, 0, 0],
                    [1, 0, 2],
                    [1, 0, 3],
                    [1, 0, 4],
                    [1, 1, 0],
                    [1, 1, 1],
                    [1, 1, 4],
                ]
            )
        return A, B

    def generate_ans(data):
        _a = data
        A, B = _a
        dims = np.maximum(B.max(0), A.max(0)) + 1
        result = A[
            ~np.in1d(np.ravel_multi_index(A.T, dims), np.ravel_multi_index(B.T, dims))
        ]
        output = np.append(
            result,
            B[
                ~np.in1d(
                    np.ravel_multi_index(B.T, dims), np.ravel_multi_index(A.T, dims)
                )
            ],
            axis=0,
        )
        return output

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    if ans.shape[0]:
        np.testing.assert_array_equal(result, ans)
    else:
        result = result.reshape(0)
        ans = ans.reshape(0)
        np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
A, B = test_input
[insert]
result = output
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
