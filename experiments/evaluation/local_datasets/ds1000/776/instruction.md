Problem:
Give the N and P, I want to get a 2D binomial distribution probability matrix M,
for i in range(N+1):
   for j in range(i+1):
      M[i,j] = choose(i, j) * p**j * (1-p)**(i-j)
other value = 0

I want to know is there any fast way to get this matrix, instead of the for loop. the N may be bigger than 100,000

A:
<code>
import numpy as np
import scipy.stats
N = 3
p = 0.5
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
import tokenize, io
import scipy.stats


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            N = 3
            p = 0.5
        elif test_case_id == 2:
            np.random.seed(234)
            N = np.random.randint(6, 10)
            p = 0.3
        return N, p

    def generate_ans(data):
        _a = data
        N, p = _a
        n = np.arange(N + 1, dtype=np.int64)
        dist = scipy.stats.binom(p=p, n=n)
        result = dist.pmf(k=np.arange(N + 1, dtype=np.int64)[:, None]).T
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert np.allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
import scipy.stats
N, p = test_input
[insert]
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
    assert "while" not in tokens and "for" not in tokens

Return only the Python code snippet that replaces [insert]:
- No explanations, Markdown, comments, prints, or extra text; no non-English output
- No imports, function/class definitions, or placeholders; only the missing logic
- Do not reassign/reset provided variables/inputs; use them to compute the result
- The snippet must be syntactically valid and run as-is in this context; deterministic (assume temperature=0)
- Do not read/write files or call external services
- Output only the code lines for [insert]; no Markdown fences, no explanations, no bullets.
