Problem:

I have fitted a k-means algorithm on 5000+ samples using the python scikit-learn library. I want to have the 50 samples closest (data, not just index) to a cluster center "p" (e.g. p=2) as an output, here "p" means the p^th center. How do I perform this task?

A:

<code>
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
p, X = load_data()
assert type(X) == np.ndarray
km = KMeans()
</code>
closest_50_samples = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
from sklearn.cluster import KMeans
import sklearn
from sklearn.datasets import make_blobs


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            X, y = make_blobs(n_samples=200, n_features=3, centers=8, random_state=42)
            p = 2
        elif test_case_id == 2:
            X, y = make_blobs(n_samples=200, n_features=3, centers=8, random_state=42)
            p = 3
        return p, X

    def generate_ans(data):
        p, X = data
        km = KMeans(n_clusters=8, random_state=42)
        km.fit(X)
        d = km.transform(X)[:, p]
        indexes = np.argsort(d)[::][:50]
        closest_50_samples = X[indexes]
        return closest_50_samples

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))

    return test_input, expected_result


def exec_test(result, ans):
    try:
        np.testing.assert_allclose(result, ans)
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
p, X = test_input
km = KMeans(n_clusters=8, random_state=42)
[insert]
result = closest_50_samples
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
