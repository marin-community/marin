Problem:
I have a raster with a set of unique ID patches/regions which I've converted into a two-dimensional Python numpy array. I would like to calculate pairwise Euclidean distances between all regions to obtain the minimum distance separating the nearest edges of each raster patch. As the array was originally a raster, a solution needs to account for diagonal distances across cells (I can always convert any distances measured in cells back to metres by multiplying by the raster resolution).
I've experimented with the cdist function from scipy.spatial.distance as suggested in this answer to a related question, but so far I've been unable to solve my problem using the available documentation. As an end result I would ideally have a N*N array in the form of "from ID, to ID, distance", including distances between all possible combinations of regions.
Here's a sample dataset resembling my input data:
import numpy as np
import matplotlib.pyplot as plt
# Sample study area array
example_array = np.array([[0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 2, 0, 2, 2, 0, 6, 0, 3, 3, 3],
                          [0, 0, 0, 0, 2, 2, 0, 0, 0, 3, 3, 3],
                          [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
                          [1, 1, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
                          [1, 1, 1, 0, 0, 0, 3, 3, 3, 0, 0, 3],
                          [1, 1, 1, 0, 0, 0, 3, 3, 3, 0, 0, 0],
                          [1, 1, 1, 0, 0, 0, 3, 3, 3, 0, 0, 0],
                          [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 1, 0, 0, 0, 0, 5, 5, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4]])
# Plot array
plt.imshow(example_array, cmap="spectral", interpolation='nearest')
A:
<code>
import numpy as np
import scipy.spatial.distance
example_arr = np.array([[0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 2, 0, 2, 2, 0, 6, 0, 3, 3, 3],
                          [0, 0, 0, 0, 2, 2, 0, 0, 0, 3, 3, 3],
                          [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
                          [1, 1, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
                          [1, 1, 1, 0, 0, 0, 3, 3, 3, 0, 0, 3],
                          [1, 1, 1, 0, 0, 0, 3, 3, 3, 0, 0, 0],
                          [1, 1, 1, 0, 0, 0, 3, 3, 3, 0, 0, 0],
                          [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 1, 0, 0, 0, 0, 5, 5, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4]])
def f(example_array = example_arr):
    # return the solution in this function
    # result = f(example_array)
    ### BEGIN SOLUTION

Execution context (contains [insert] placeholder):
import numpy as np
import itertools
import copy
import scipy.spatial.distance


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            example_array = np.array(
                [
                    [0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 2, 0, 2, 2, 0, 6, 0, 3, 3, 3],
                    [0, 0, 0, 0, 2, 2, 0, 0, 0, 3, 3, 3],
                    [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
                    [1, 1, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
                    [1, 1, 1, 0, 0, 0, 3, 3, 3, 0, 0, 3],
                    [1, 1, 1, 0, 0, 0, 3, 3, 3, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 3, 3, 3, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0, 0, 5, 5, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
                ]
            )
        return example_array

    def generate_ans(data):
        _a = data
        example_array = _a
        n = example_array.max() + 1
        indexes = []
        for k in range(1, n):
            tmp = np.nonzero(example_array == k)
            tmp = np.asarray(tmp).T
            indexes.append(tmp)
        result = np.zeros((n - 1, n - 1))
        for i, j in itertools.combinations(range(n - 1), 2):
            d2 = scipy.spatial.distance.cdist(
                indexes[i], indexes[j], metric="sqeuclidean"
            )
            result[i, j] = result[j, i] = d2.min() ** 0.5
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert np.allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
import scipy.spatial.distance
example_array = test_input
def f(example_array):
[insert]
result = f(example_array)
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
