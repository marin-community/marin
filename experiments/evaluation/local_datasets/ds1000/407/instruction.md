Problem:
I'm looking for a fast solution to MATLAB's accumarray in numpy. The accumarray accumulates the elements of an array which belong to the same index.
Note that there might be negative indices in accmap, and we treat them like list indices in Python.
 An example:
a = np.arange(1,11)
# array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
accmap = np.array([0,1,0,0,0,-1,-1,2,2,1])
Result should be
array([13, 12, 30])
Is there a built-in numpy function that can do accumulation like this? Using for-loop is not what I want. Or any other recommendations?
A:
<code>
import numpy as np
a = np.arange(1,11)
accmap = np.array([0,1,0,0,0,-1,-1,2,2,1])
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = np.arange(1, 11)
            accmap = np.array([0, 1, 0, 0, 0, -1, -1, 2, 2, 1])
        elif test_case_id == 2:
            np.random.seed(42)
            accmap = np.random.randint(-2, 5, (100,))
            a = np.random.randint(-100, 100, (100,))
        return a, accmap

    def generate_ans(data):
        _a = data
        a, accmap = _a
        add = np.max(accmap)
        mask = accmap < 0
        accmap[mask] += add + 1
        result = np.bincount(accmap, weights=a)
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
a, accmap = test_input
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
