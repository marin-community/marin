Problem:
I have the following code to run Wilcoxon rank-sum test 
print stats.ranksums(pre_course_scores, during_course_scores)
RanksumsResult(statistic=8.1341352369246582, pvalue=4.1488919597127145e-16)

However, I am interested in extracting the pvalue from the result. I could not find a tutorial about this. i.e.Given two ndarrays, pre_course_scores, during_course_scores, I want to know the pvalue of ranksum. Can someone help?

A:
<code>
import numpy as np
from scipy import stats
example_pre_course_scores = np.random.randn(10)
example_during_course_scores = np.random.randn(10)
def f(pre_course_scores = example_pre_course_scores, during_course_scores = example_during_course_scores):
    # return the solution in this function
    # p_value = f(pre_course_scores, during_course_scores)
    ### BEGIN SOLUTION

Execution context (contains [insert] placeholder):
import numpy as np
import copy
from scipy import stats


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(10)
            A = np.random.randn(10)
            B = np.random.randn(10)
        return A, B

    def generate_ans(data):
        _a = data
        pre_course_scores, during_course_scores = _a
        p_value = stats.ranksums(pre_course_scores, during_course_scores).pvalue
        return p_value

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert abs(result - ans) <= 1e-5
    return 1


exec_context = r"""
import numpy as np
from scipy import stats
def f(pre_course_scores, during_course_scores):
[insert]
pre_course_scores, during_course_scores = test_input
result = f(pre_course_scores, during_course_scores)
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
