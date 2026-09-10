Problem:

I need to perform hierarchical clustering by a distance matrix describing their similarities, which is between different professors, like:

              prof1     prof2     prof3
       prof1     0        0.8     0.9
       prof2     0.8      0       0.2
       prof3     0.9      0.2     0

       data_matrix=[[0,0.8,0.9],[0.8,0,0.2],[0.9,0.2,0]]
The expected number of clusters is 2. Can it be done using sklearn.cluster.AgglomerativeClustering? I tried to do that but failed. Anyone can give me some advice? prefer answer in a list like [label1, label2, ...]

A:

<code>
import numpy as np
import pandas as pd
import sklearn.cluster
data_matrix = load_data()
</code>
cluster_labels = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
import tokenize, io
from sklearn.cluster import AgglomerativeClustering


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            data_matrix = [[0, 0.8, 0.9], [0.8, 0, 0.2], [0.9, 0.2, 0]]
        elif test_case_id == 2:
            data_matrix = [[0, 0.2, 0.9], [0.2, 0, 0.8], [0.9, 0.8, 0]]
        return data_matrix

    def generate_ans(data):
        data_matrix = data
        model = AgglomerativeClustering(
            metric="precomputed", n_clusters=2, linkage="complete"
        ).fit(data_matrix)
        cluster_labels = model.labels_
        return cluster_labels

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    ret = 0
    try:
        np.testing.assert_equal(result, ans)
        ret = 1
    except:
        pass
    try:
        np.testing.assert_equal(result, 1 - ans)
        ret = 1
    except:
        pass
    return ret


exec_context = r"""
import pandas as pd
import numpy as np
import sklearn.cluster
data_matrix = test_input
[insert]
result = cluster_labels
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(2):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)


def test_string(solution: str):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(solution.encode("utf-8")).readline):
        tokens.append(token.string)
    assert "AgglomerativeClustering" in tokens

Return only the Python code snippet that replaces [insert]:
- No explanations, Markdown, comments, prints, or extra text; no non-English output
- No imports, function/class definitions, or placeholders; only the missing logic
- Do not reassign/reset provided variables/inputs; use them to compute the result
- The snippet must be syntactically valid and run as-is in this context; deterministic (assume temperature=0)
- Do not read/write files or call external services
- Output only the code lines for [insert]; no Markdown fences, no explanations, no bullets.
