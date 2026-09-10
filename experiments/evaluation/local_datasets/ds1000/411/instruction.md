Problem:
In numpy, is there a way to zero pad entries if I'm slicing past the end of the array, such that I get something that is the size of the desired slice?
For example,
>>> a = np.ones((3,3,))
>>> a
array([[ 1.,  1.,  1.],
       [ 1.,  1.,  1.],
       [ 1.,  1.,  1.]])
>>> a[1:4, 1:4] # would behave as a[1:3, 1:3] by default
array([[ 1.,  1.,  0.],
       [ 1.,  1.,  0.],
       [ 0.,  0.,  0.]])
>>> a[-1:2, -1:2]
 array([[ 0.,  0.,  0.],
       [ 0.,  1.,  1.],
       [ 0.,  1.,  1.]])
I'm dealing with images and would like to zero pad to signify moving off the image for my application.
My current plan is to use np.pad to make the entire array larger prior to slicing, but indexing seems to be a bit tricky. Is there a potentially easier way?
A:
<code>
import numpy as np
a = np.ones((3, 3))
low_index = -1
high_index = 2
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
            a = np.ones((3, 3))
            low_index = -1
            high_index = 2
        elif test_case_id == 2:
            a = np.ones((5, 5)) * 2
            low_index = 1
            high_index = 6
        elif test_case_id == 3:
            a = np.ones((5, 5))
            low_index = 2
            high_index = 7
        return a, low_index, high_index

    def generate_ans(data):
        _a = data
        a, low_index, high_index = _a

        def fill_crop(img, pos, crop):
            img_shape, pos, crop_shape = (
                np.array(img.shape),
                np.array(pos),
                np.array(crop.shape),
            )
            end = pos + crop_shape
            crop_low = np.clip(0 - pos, a_min=0, a_max=crop_shape)
            crop_high = crop_shape - np.clip(end - img_shape, a_min=0, a_max=crop_shape)
            crop_slices = (slice(low, high) for low, high in zip(crop_low, crop_high))
            pos = np.clip(pos, a_min=0, a_max=img_shape)
            end = np.clip(end, a_min=0, a_max=img_shape)
            img_slices = (slice(low, high) for low, high in zip(pos, end))
            crop[tuple(crop_slices)] = img[tuple(img_slices)]
            return crop

        result = fill_crop(
            a,
            [low_index, low_index],
            np.zeros((high_index - low_index, high_index - low_index)),
        )
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
a, low_index, high_index = test_input
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
