Problem:
How to get one maximal set of linearly independent vectors of a given matrix `a`?
For example, [[0 1 0 0], [0 0 1 0], [1 0 0 1]] in [[0 1 0 0], [0 0 1 0], [0 1 1 0], [1 0 0 1]]
A:
<code>
import numpy as np
a = np.array([[0,1,0,0], [0,0,1,0], [0,1,1,0], [1,0,0,1]])
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
            a = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1]])
        elif test_case_id == 2:
            a = np.array(
                [
                    [0, 1, 0, 0, 1, 1],
                    [0, 0, 1, 0, 0, 1],
                    [0, 1, 1, 0, 0, 0],
                    [1, 0, 1, 0, 0, 1],
                    [1, 1, 0, 1, 0, 0],
                ]
            )
        elif test_case_id == 3:
            a = np.array(
                [
                    [0, 1, 0, 0, 1, 1],
                    [0, 1, 0, 0, 1, 1],
                    [0, 1, 0, 0, 1, 1],
                    [1, 0, 1, 0, 0, 1],
                    [1, 1, 0, 1, 0, 0],
                ]
            )
        return a

    def generate_ans(data):
        _a = data
        a = _a
        return a

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    result = np.array(result)
    if result.shape[1] == ans.shape[1]:
        assert np.linalg.matrix_rank(ans) == np.linalg.matrix_rank(
            result
        ) and np.linalg.matrix_rank(result) == len(result)
        assert len(np.unique(result, axis=0)) == len(result)
        for arr in result:
            assert np.any(np.all(ans == arr, axis=1))
    else:
        assert (
            np.linalg.matrix_rank(ans) == np.linalg.matrix_rank(result)
            and np.linalg.matrix_rank(result) == result.shape[1]
        )
        assert np.unique(result, axis=1).shape[1] == result.shape[1]
        for i in range(result.shape[1]):
            assert np.any(np.all(ans == result[:, i].reshape(-1, 1), axis=0))
    return 1


exec_context = r"""
import numpy as np
a = test_input
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
