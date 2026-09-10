Problem:
When testing if a numpy array c is member of a list of numpy arrays CNTS:
import numpy as np
c = np.array([[[ NaN, 763]],
              [[ 57, 763]],
              [[ 57, 749]],
              [[ 75, 749]]])
CNTS = [np.array([[[  78, 1202]],
                  [[  63, 1202]],
                  [[  63, 1187]],
                  [[  78, 1187]]]),
        np.array([[[ NaN, 763]],
                  [[ 57, 763]],
                  [[ 57, 749]],
                  [[ 75, 749]]]),
        np.array([[[ 72, 742]],
                  [[ 58, 742]],
                  [[ 57, 741]],
                  [[ 57, NaN]],
                  [[ 58, 726]],
                  [[ 72, 726]]]),
        np.array([[[ 66, 194]],
                  [[ 51, 194]],
                  [[ 51, 179]],
                  [[ 66, 179]]])]
print(c in CNTS)
I get:
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
However, the answer is rather clear: c is exactly CNTS[1], so c in CNTS should return True!
How to correctly test if a numpy array is member of a list of numpy arrays? Additionally, arrays might contain NaN!
The same problem happens when removing:
CNTS.remove(c)
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
Application: test if an opencv contour (numpy array) is member of a list of contours, see for example Remove an opencv contour from a list of contours.
A:
<code>
import numpy as np
c = np.array([[[ 75, 763]],
              [[ 57, 763]],
              [[ np.nan, 749]],
              [[ 75, 749]]])
CNTS = [np.array([[[  np.nan, 1202]],
                  [[  63, 1202]],
                  [[  63, 1187]],
                  [[  78, 1187]]]),
        np.array([[[ 75, 763]],
                  [[ 57, 763]],
                  [[ np.nan, 749]],
                  [[ 75, 749]]]),
        np.array([[[ 72, 742]],
                  [[ 58, 742]],
                  [[ 57, 741]],
                  [[ 57, np.nan]],
                  [[ 58, 726]],
                  [[ 72, 726]]]),
        np.array([[[ np.nan, 194]],
                  [[ 51, 194]],
                  [[ 51, 179]],
                  [[ 66, 179]]])]
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
            c = np.array([[[75, 763]], [[57, 763]], [[np.nan, 749]], [[75, 749]]])
            CNTS = [
                np.array([[[np.nan, 1202]], [[63, 1202]], [[63, 1187]], [[78, 1187]]]),
                np.array([[[75, 763]], [[57, 763]], [[np.nan, 749]], [[75, 749]]]),
                np.array(
                    [
                        [[72, 742]],
                        [[58, 742]],
                        [[57, 741]],
                        [[57, np.nan]],
                        [[58, 726]],
                        [[72, 726]],
                    ]
                ),
                np.array([[[np.nan, 194]], [[51, 194]], [[51, 179]], [[66, 179]]]),
            ]
        elif test_case_id == 2:
            np.random.seed(42)
            c = np.random.rand(3, 4)
            CNTS = [np.random.rand(x, x + 2) for x in range(3, 7)]
        elif test_case_id == 3:
            c = np.array([[[75, 763]], [[57, 763]], [[np.nan, 749]], [[75, 749]]])
            CNTS = [
                np.array([[[np.nan, 1202]], [[63, 1202]], [[63, 1187]], [[78, 1187]]]),
                np.array([[[np.nan, 763]], [[57, 763]], [[20, 749]], [[75, 749]]]),
                np.array(
                    [
                        [[72, 742]],
                        [[58, 742]],
                        [[57, 741]],
                        [[57, np.nan]],
                        [[58, 726]],
                        [[72, 726]],
                    ]
                ),
                np.array([[[np.nan, 194]], [[51, 194]], [[51, 179]], [[66, 179]]]),
            ]
        return c, CNTS

    def generate_ans(data):
        _a = data
        c, CNTS = _a
        temp_c = c.copy()
        temp_c[np.isnan(temp_c)] = 0
        result = False
        for arr in CNTS:
            temp = arr.copy()
            temp[np.isnan(temp)] = 0
            result |= (
                np.array_equal(temp_c, temp) and (np.isnan(c) == np.isnan(arr)).all()
            )
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert result == ans
    return 1


exec_context = r"""
import numpy as np
c, CNTS = test_input
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
