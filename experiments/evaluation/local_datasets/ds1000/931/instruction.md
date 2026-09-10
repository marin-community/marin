Problem:

I am using python and scikit-learn to find cosine similarity between item descriptions.

A have a df, for example:

items    description

1fgg     abcd ty
2hhj     abc r
3jkl     r df
I did following procedures:

1) tokenizing each description

2) transform the corpus into vector space using tf-idf

3) calculated cosine distance between each description text as a measure of similarity. distance = 1 - cosinesimilarity(tfidf_matrix)

My goal is to have a similarity matrix of items like this and answer the question like: "What is the similarity between the items 1ffg and 2hhj :

        1fgg    2hhj    3jkl
1ffg    1.0     0.8     0.1
2hhj    0.8     1.0     0.0
3jkl    0.1     0.0     1.0
How to get this result? Thank you for your time.

A:

<code>
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
df = load_data()
tfidf = TfidfVectorizer()
</code>
cosine_similarity_matrix = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import pandas as pd
import copy
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn
from sklearn.metrics.pairwise import cosine_similarity


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "items": ["1fgg", "2hhj", "3jkl"],
                    "description": ["abcd ty", "abc r", "r df"],
                }
            )
        elif test_case_id == 2:
            df = pd.DataFrame(
                {
                    "items": ["1fgg", "2hhj", "3jkl", "4dsd"],
                    "description": [
                        "Chinese Beijing Chinese",
                        "Chinese Chinese Shanghai",
                        "Chinese Macao",
                        "Tokyo Japan Chinese",
                    ],
                }
            )
        return df

    def generate_ans(data):
        df = data
        tfidf = TfidfVectorizer()
        response = tfidf.fit_transform(df["description"]).toarray()
        tf_idf = response
        cosine_similarity_matrix = np.zeros((len(df), len(df)))
        for i in range(len(df)):
            for j in range(len(df)):
                cosine_similarity_matrix[i, j] = cosine_similarity(
                    [tf_idf[i, :]], [tf_idf[j, :]]
                )
        return cosine_similarity_matrix

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
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
df = test_input
tfidf = TfidfVectorizer()
[insert]
result = cosine_similarity_matrix
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
