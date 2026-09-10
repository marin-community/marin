Problem:

Using scipy, is there an easy way to emulate the behaviour of MATLAB's dctmtx function which returns a NxN (ortho-mode normed) DCT matrix for some given N? There's scipy.fftpack.dctn but that only applies the DCT. Do I have to implement this from scratch if I don't want use another dependency besides scipy?
A:
<code>
import numpy as np
import scipy.fft as sf
N = 8
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import copy
import scipy.fft as sf


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            N = 8
        elif test_case_id == 2:
            np.random.seed(42)
            N = np.random.randint(4, 15)
        return N

    def generate_ans(data):
        _a = data
        N = _a
        result = sf.dct(np.eye(N), axis=0, norm="ortho")
        return result

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert np.allclose(result, ans)
    return 1


exec_context = r"""
import numpy as np
import scipy.fft as sf
N = test_input
[insert]
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
