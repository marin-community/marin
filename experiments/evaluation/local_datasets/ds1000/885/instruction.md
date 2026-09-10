Problem:

Given a distance matrix, with similarity between various fruits :

              fruit1     fruit2     fruit3
       fruit1     0        0.6     0.8
       fruit2     0.6      0       0.111
       fruit3     0.8      0.111     0
I need to perform hierarchical clustering on this data (into 2 clusters), where the above data is in the form of 2-d matrix

       simM=[[0,0.6,0.8],[0.6,0,0.111],[0.8,0.111,0]]
The expected number of clusters is 2. Can it be done using scipy.cluster.hierarchy? prefer answer in a list like [label1, label2, ...]

A:

<code>
import numpy as np
import pandas as pd
import scipy.cluster
simM = load_data()
</code>
cluster_labels = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
import tokenize, io
from scipy.cluster.hierarchy import linkage, cut_tree


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            data_matrix = [[0, 0.8, 0.9], [0.8, 0, 0.2], [0.9, 0.2, 0]]
        elif test_case_id == 2:
            data_matrix = [[0, 0.2, 0.9], [0.2, 0, 0.8], [0.9, 0.8, 0]]
        return data_matrix

    def generate_ans(data):
        simM = data
        Z = linkage(np.array(simM), "ward")
        cluster_labels = (
            cut_tree(Z, n_clusters=2)
            .reshape(
                -1,
            )
            .tolist()
        )
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
        np.testing.assert_equal(result, 1 - np.array(ans))
        ret = 1
    except:
        pass
    return ret


exec_context = r"""
import pandas as pd
import numpy as np
import scipy.cluster
simM = test_input
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
    assert "hierarchy" in tokens

Return only the Python code snippet that replaces [insert]:
- No explanations, Markdown, comments, prints, or extra text; no non-English output
- No imports, function/class definitions, or placeholders; only the missing logic
- Do not reassign/reset provided variables/inputs; use them to compute the result
- The snippet must be syntactically valid and run as-is in this context; deterministic (assume temperature=0)
- Do not read/write files or call external services
- Output only the code lines for [insert]; no Markdown fences, no explanations, no bullets.
