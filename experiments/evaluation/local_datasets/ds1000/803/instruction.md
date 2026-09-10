Problem:
After clustering a distance matrix with scipy.cluster.hierarchy.linkage, and assigning each sample to a cluster using scipy.cluster.hierarchy.cut_tree, I would like to extract one element out of each cluster, which is the closest to that cluster's centroid.
•	I would be the happiest if an off-the-shelf function existed for this, but in the lack thereof:
•	some suggestions were already proposed here for extracting the centroids themselves, but not the closest-to-centroid elements.
•	Note that this is not to be confused with the centroid linkage rule in scipy.cluster.hierarchy.linkage. I have already carried out the clustering itself, just want to access the closest-to-centroid elements.
What I want is the index of the closest element in original data for each cluster, i.e., result[0] is the index of the closest element to cluster 0.
A:
<code>
import numpy as np
import scipy.spatial
centroids = np.random.rand(5, 3)
data = np.random.rand(100, 3)
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
import scipy.spatial


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(42)
            centroids = np.random.rand(5, 3)
            data = np.random.rand(100, 3)
        return centroids, data

    def generate_ans(data):
        _a = data
        centroids, data = _a

        def find_k_closest(centroids, data, k=1, distance_norm=2):
            kdtree = scipy.spatial.cKDTree(data)
            distances, indices = kdtree.query(centroids, k, p=distance_norm)
            if k > 1:
                indices = indices[:, -1]
            values = data[indices]
            return indices, values

        result, _ = find_k_closest(centroids, data)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import scipy.spatial
import numpy as np
centroids, data = test_input
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
