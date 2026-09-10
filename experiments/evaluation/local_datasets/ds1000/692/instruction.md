Problem:
I'm using tensorflow 2.10.0.

import tensorflow as tf
x = [[1,2,3],[4,5,6]]
row = [0,1]
col = [0,2]
x = tf.constant(x)
row = tf.constant(row)
col = tf.constant(col)
m = x[[row,col]]

What I expect is m = [1,6]
I can get the result by theano or numpy. How I get the result using tensorflow?


A:
<code>
import tensorflow as tf

x = [[1,2,3],[4,5,6]]
row = [0,0]
col = [1,2]
x = tf.constant(x)
row = tf.constant(row)
col = tf.constant(col)
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import tensorflow as tf
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        x, row, col = data
        index = [[row[i], col[i]] for i in range(len(col))]
        return tf.gather_nd(x, index)

    def define_test_input(test_case_id):
        if test_case_id == 1:
            x = [[1, 2, 3], [4, 5, 6]]
            row = [0, 0]
            col = [1, 2]
            x = tf.constant(x)
            row = tf.constant(row)
            col = tf.constant(col)
        if test_case_id == 2:
            x = [[1, 2, 3], [4, 5, 6]]
            row = [1, 0]
            col = [1, 2]
            x = tf.constant(x)
            row = tf.constant(row)
            col = tf.constant(col)
        return x, row, col

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    def tensor_equal(a, b):
        if type(a) != type(b):
            return False
        if isinstance(a, type(tf.constant([]))) is not True:
            if isinstance(a, type(tf.Variable([]))) is not True:
                return False
        if a.shape != b.shape:
            return False
        if a.dtype != tf.float32:
            a = tf.cast(a, tf.float32)
        if b.dtype != tf.float32:
            b = tf.cast(b, tf.float32)
        if not tf.reduce_min(tf.cast(a == b, dtype=tf.int32)):
            return False
        return True

    try:
        assert tensor_equal(result, ans)
        return 1
    except:
        return 0


exec_context = r"""
import tensorflow as tf
x,row,col = test_input
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
