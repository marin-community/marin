Problem:
How can I read a Numpy array from a string? Take a string like:
"[[ 0.5544  0.4456], [ 0.8811  0.1189]]"
and convert it to an array:
a = from_string("[[ 0.5544  0.4456], [ 0.8811  0.1189]]")
where a becomes the object: np.array([[0.5544, 0.4456], [0.8811, 0.1189]]).
There's nothing I can find in the NumPy docs that does this. 
A:
<code>
import numpy as np
string = "[[ 0.5544  0.4456], [ 0.8811  0.1189]]"
</code>
a = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            string = "[[ 0.5544  0.4456], [ 0.8811  0.1189]]"
        elif test_case_id == 2:
            np.random.seed(42)
            a = np.random.rand(5, 6)
            string = str(a).replace("\n", ",")
        return string

    def generate_ans(data):
        _a = data
        string = _a
        a = np.array(np.matrix(string.replace(",", ";")))
        return a

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1


exec_context = r"""
import numpy as np
string = test_input
[insert]
result = a
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
