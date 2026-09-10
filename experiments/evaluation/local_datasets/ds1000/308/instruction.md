Problem:
I am waiting for another developer to finish a piece of code that will return an np array of shape (100,2000) with values of either -1,0, or 1.
In the meantime, I want to randomly create an array of the same characteristics so I can get a head start on my development and testing. The thing is that I want this randomly created array to be the same each time, so that I'm not testing against an array that keeps changing its value each time I re-run my process.
I can create my array like this, but is there a way to create it so that it's the same each time. I can pickle the object and unpickle it, but wondering if there's another way.
r = np.random.randint(3, size=(100, 2000)) - 1
Specifically, I want r_old, r_new to be generated in the same way as r, but their result should be the same.
A:
<code>
import numpy as np
</code>
r_old, r_new = ... # put solution in these variables
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import pandas as pd
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            return None

    def generate_ans(data):
        none_input = data
        np.random.seed(0)
        r_old = np.random.randint(3, size=(100, 2000)) - 1
        np.random.seed(0)
        r_new = np.random.randint(3, size=(100, 2000)) - 1
        return r_old, r_new

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    r_old, r_new = result
    assert id(r_old) != id(r_new)
    np.testing.assert_array_equal(r_old, r_new)
    return 1


exec_context = r"""
import numpy as np
[insert]
result = [r_old, r_new]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)


def test_string(solution: str):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(solution.encode("utf-8")).readline):
        tokens.append(token.string)
    assert "randint" in tokens

Return only the Python code snippet that replaces [insert]:
- No explanations, Markdown, comments, prints, or extra text; no non-English output
- No imports, function/class definitions, or placeholders; only the missing logic
- Do not reassign/reset provided variables/inputs; use them to compute the result
- The snippet must be syntactically valid and run as-is in this context; deterministic (assume temperature=0)
- Do not read/write files or call external services
- Output only the code lines for [insert]; no Markdown fences, no explanations, no bullets.
