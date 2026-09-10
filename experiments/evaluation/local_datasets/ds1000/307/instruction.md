Problem:
Following-up from this question years ago, is there a "shift" function in numpy? Ideally it can be applied to 2-dimensional arrays, and the numbers of shift are different among rows.
Example:
In [76]: xs
Out[76]: array([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],
		 [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.]])
In [77]: shift(xs, [1,3])
Out[77]: array([[nan,   0.,   1.,   2.,   3.,   4.,   5.,   6.,	7.,	8.], [nan, nan, nan, 1.,  2.,  3.,  4.,  5.,  6.,  7.])
In [78]: shift(xs, [-2,-3])
Out[78]: array([[2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  nan,  nan], [4.,  5.,  6.,  7.,  8.,  9., 10., nan, nan, nan]])
Any help would be appreciated.
A:
<code>
import numpy as np
a = np.array([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],
		[1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.]])
shift = [-2, 3]
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
            a = np.array(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                ]
            )
            shift = [-2, 3]
        elif test_case_id == 2:
            np.random.seed(42)
            a = np.random.rand(10, 100)
            shift = np.random.randint(-99, 99, (10,))
        return a, shift

    def generate_ans(data):
        _a = data
        a, shift = _a

        def solution(xs, shift):
            e = np.empty_like(xs)
            for i, n in enumerate(shift):
                if n >= 0:
                    e[i, :n] = np.nan
                    e[i, n:] = xs[i, :-n]
                else:
                    e[i, n:] = np.nan
                    e[i, :n] = xs[i, -n:]
            return e

        result = solution(a, shift)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
a, shift = test_input
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
