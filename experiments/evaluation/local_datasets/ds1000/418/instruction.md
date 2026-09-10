Problem:
I have a 2-dimensional numpy array which contains time series data. I want to bin that array into equal partitions of a given length (it is fine to drop the last partition if it is not the same size) and then calculate the mean of each of those bins. Due to some reason, I want the binning starts from the end of the array.
I suspect there is numpy, scipy, or pandas functionality to do this.
example:
data = [[4,2,5,6,7],
	[5,4,3,5,7]]
for a bin size of 2:
bin_data = [[(6,7),(2,5)],
	     [(5,7),(4,3)]]
bin_data_mean = [[6.5,3.5],
		  [6,3.5]]
for a bin size of 3:
bin_data = [[(5,6,7)],
	     [(3,5,7)]]
bin_data_mean = [[6],
		  [5]]
A:
<code>
import numpy as np
data = np.array([[4, 2, 5, 6, 7],
[ 5, 4, 3, 5, 7]])
bin_size = 3
</code>
bin_data_mean = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import numpy as np
import pandas as pd
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            data = np.array([[4, 2, 5, 6, 7], [5, 4, 3, 5, 7]])
            width = 3
        elif test_case_id == 2:
            np.random.seed(42)
            data = np.random.rand(np.random.randint(5, 10), np.random.randint(6, 10))
            width = np.random.randint(2, 4)
        return data, width

    def generate_ans(data):
        _a = data
        data, bin_size = _a
        new_data = data[:, ::-1]
        bin_data_mean = (
            new_data[:, : (data.shape[1] // bin_size) * bin_size]
            .reshape(data.shape[0], -1, bin_size)
            .mean(axis=-1)
        )
        return bin_data_mean

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    np.testing.assert_allclose(result, ans, atol=1e-2)
    return 1


exec_context = r"""
import numpy as np
data, bin_size = test_input
[insert]
result = bin_data_mean
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
