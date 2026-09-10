Problem:

My goal is to input 3 queries and find out which query is most similar to a set of 5 documents.

So far I have calculated the tf-idf of the documents doing the following:

from sklearn.feature_extraction.text import TfidfVectorizer

def get_term_frequency_inverse_data_frequency(documents):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(documents)
    return matrix

def get_tf_idf_query_similarity(documents, query):
    tfidf = get_term_frequency_inverse_data_frequency(documents)
The problem I am having is now that I have tf-idf of the documents what operations do I perform on the query so I can find the cosine similarity to the documents? The answer should be like a 3*5 matrix of the similarities.

A:

<code>
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
queries, documents = load_data()
assert type(queries) == list
assert type(documents) == list
def solve(queries, documents):
    tfidf = TfidfVectorizer()
    tfidf.fit_transform(documents)
    # return the solution in this function
    # cosine_similarities_of_queries = solve(queries, documents)
    ### BEGIN SOLUTION

Execution context (contains [insert] placeholder):
import numpy as np
import copy
import scipy.sparse
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def generate_test_case(test_case_id):

    def define_test_input(test_case_id):
        if test_case_id == 1:
            document1 = [
                "Education is the process of learning and acquiring knowledge at an educational institution. It can be divided into three levels: primary, secondary, and tertiary."
                "Primary education generally refers to the first stage of schooling, which is typically compulsory and free in most countries. It usually lasts for six or seven years, from ages five or six to eleven or twelve."
                "Secondary education usually lasts for three or four years, from ages eleven or twelve to fifteen or sixteen. In some countries, it is compulsory, while in others it is not."
                "Tertiary education, also known as post-secondary education, is the stage of education that comes after secondary school. It can be divided into two types: university education and vocational education."
                "University education typically lasts for four years, from ages eighteen to twenty-two. It is usually divided into two parts: undergraduate and graduate."
                "Vocational education is training for a specific trade or profession. It can be either formal or informal. Formal vocational education is typically provided by vocational schools, while informal vocational education is provided by apprenticeships and on-the-job training."
            ]
            document2 = [
                "The purpose of education is to prepare individuals for successful futures. Whether that means getting a good job, being a responsible citizen, or becoming a lifelong learner, education is the key to a bright future."
                "There are many different types of educational institutions, each with its own unique purpose. Primary and secondary schools are the most common, but there are also institutions for special needs education, higher education, and adult education."
                "All educational institutions share the common goal of providing quality education to their students. However, the methods and curriculum used to achieve this goal can vary greatly."
                "Some educational institutions focus on academic knowledge, while others place more emphasis on practical skills. Some use traditional teaching methods, while others use more innovative approaches."
                "The type of education an individual receives should be based on their needs and goals. There is no one-size-fits-all approach to education, and what works for one person may not work for another."
            ]
            document3 = [
                "Education is a fundamental human right. It is essential for the development of individuals and societies. "
                "Access to education should be universal and inclusive. "
                "All individuals, regardless of race, gender, or social status, should have the opportunity to receive an education. Education is a tool that can be used to empower individuals and improve the quality of their lives. "
                "It can help people to develop new skills and knowledge, and to achieve their full potential. Education is also a key factor in the development of societies. "
                "It can help to create more stable and prosperous societies."
            ]
            document4 = [
                "The quality of education is a key factor in the development of individuals and societies."
                "A high-quality education can help individuals to develop their skills and knowledge, and to achieve their full potential. It can also help to create more stable and prosperous societies."
                "There are many factors that contribute to the quality of education. These include the qualifications of teachers, the resources available, and the curriculum being taught."
                "Improving the quality of education is an important goal for all educational institutions. By providing quality education, we can empower individuals and help to create a better world for everyone."
            ]
            document5 = [
                "Education is an important investment in the future.\n\n"
                "By providing quality education, we can empower individuals and help to create a better world for everyone.\n\nA high-quality education can help individuals to develop their skills and knowledge, and to achieve their full potential. It can also help to create more stable and prosperous societies.\n\n"
                "There are many factors that contribute to the quality of education. These include the qualifications of teachers, the resources available, and the curriculum being taught.\n\n"
                "Improving the quality of education is an important goal for all educational institutions. By investing in education, we can make a difference in the world."
            ]
            documents = []
            documents.extend(document1)
            documents.extend(document2)
            documents.extend(document3)
            documents.extend(document4)
            documents.extend(document5)
            query1 = ["What are the benefits of education?"]
            query2 = ["What are the different branches of the military?"]
            query3 = ["What is the definition of science?"]
            queries = []
            queries.extend(query1)
            queries.extend(query2)
            queries.extend(query3)
        return queries, documents

    def generate_ans(data):
        queries, documents = data
        tfidf = TfidfVectorizer()
        tfidf.fit_transform(documents)
        cosine_similarities_of_queries = []
        for query in queries:
            query_tfidf = tfidf.transform([query])
            cosine_similarities_of_queries.append(
                cosine_similarity(query_tfidf, tfidf.transform(documents)).flatten()
            )
        return cosine_similarities_of_queries

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))

    return test_input, expected_result


def exec_test(result, ans):
    try:
        if type(result) == scipy.sparse.csr.csr_matrix:
            result = result.toarray()
        np.testing.assert_allclose(result, ans)
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
queries, documents = test_input
def solve(queries, documents):
    tfidf = TfidfVectorizer()
    tfidf.fit_transform(documents)
[insert]
cosine_similarities_of_queries = solve(queries, documents)
result = cosine_similarities_of_queries
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
