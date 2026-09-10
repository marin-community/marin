Problem:
I have a numpy array of different numpy arrays and I want to make a deep copy of the arrays. I found out the following:
import numpy as np
pairs = [(2, 3), (3, 4), (4, 5)]
array_of_arrays = np.array([np.arange(a*b).reshape(a,b) for (a, b) in pairs])
a = array_of_arrays[:] # Does not work
b = array_of_arrays[:][:] # Does not work
c = np.array(array_of_arrays, copy=True) # Does not work
Is for-loop the best way to do this? Is there a deep copy function I missed? And what is the best way to interact with each element in this array of different sized arrays?
A:
<code>
import numpy as np
pairs = [(2, 3), (3, 4), (4, 5)]
array_of_arrays = np.array([np.arange(a*b).reshape(a,b) for (a, b) in pairs])
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
            pairs = [(2, 3), (3, 4), (4, 5)]
            array_of_arrays = np.array(
                [np.arange(a * b).reshape(a, b) for (a, b) in pairs], dtype=object
            )
        return pairs, array_of_arrays

    def generate_ans(data):
        _a = data
        pairs, array_of_arrays = _a
        return array_of_arrays

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert id(result) != id(ans)
    for arr1, arr2 in zip(result, ans):
        assert id(arr1) != id(arr2)
        np.testing.assert_array_equal(arr1, arr2)
    return 1


exec_context = r"""
import numpy as np
pairs, array_of_arrays = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
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
