Problem:

I am trying to vectorize some data using

sklearn.feature_extraction.text.CountVectorizer.
This is the data that I am trying to vectorize:

corpus = [
 'We are looking for Java developer',
 'Frontend developer with knowledge in SQL and Jscript',
 'And this is the third one.',
 'Is this the first document?',
]
Properties of the vectorizer are defined by the code below:

vectorizer = CountVectorizer(stop_words="english",binary=True,lowercase=False,vocabulary={'Jscript','.Net','TypeScript','NodeJS','Angular','Mongo','CSS','Python','PHP','Photoshop','Oracle','Linux','C++',"Java",'TeamCity','Frontend','Backend','Full stack', 'UI Design', 'Web','Integration','Database design','UX'})
After I run:

X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(X.toarray())
I get desired results but keywords from vocabulary are ordered alphabetically. The output looks like this:

['.Net', 'Angular', 'Backend', 'C++', 'CSS', 'Database design',
'Frontend', 'Full stack', 'Integration', 'Java', 'Jscript', 'Linux',
'Mongo', 'NodeJS', 'Oracle', 'PHP', 'Photoshop', 'Python',
'TeamCity', 'TypeScript', 'UI Design', 'UX', 'Web']

[
[0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0]
[0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0]
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
]
As you can see, the vocabulary is not in the same order as I set it above. Is there a way to change this? Thanks

A:

<code>
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'We are looking for Java developer',
    'Frontend developer with knowledge in SQL and Jscript',
    'And this is the third one.',
    'Is this the first document?',
]
</code>
feature_names, X = ... # put solution in these variables
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
from sklearn.feature_extraction.text import CountVectorizer


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            corpus = [
                "We are looking for Java developer",
                "Frontend developer with knowledge in SQL and Jscript",
                "And this is the third one.",
                "Is this the first document?",
            ]
        return corpus

    def generate_ans(data):
        corpus = data
        vectorizer = CountVectorizer(
            stop_words="english",
            binary=True,
            lowercase=False,
            vocabulary=[
                "Jscript",
                ".Net",
                "TypeScript",
                "NodeJS",
                "Angular",
                "Mongo",
                "CSS",
                "Python",
                "PHP",
                "Photoshop",
                "Oracle",
                "Linux",
                "C++",
                "Java",
                "TeamCity",
                "Frontend",
                "Backend",
                "Full stack",
                "UI Design",
                "Web",
                "Integration",
                "Database design",
                "UX",
            ],
        )
        X = vectorizer.fit_transform(corpus).toarray()
        feature_names = vectorizer.get_feature_names_out()
        return feature_names, X

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        np.testing.assert_equal(result[0], ans[0])
        np.testing.assert_equal(result[1], ans[1])
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
corpus = test_input
[insert]
result = (feature_names, X)
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
