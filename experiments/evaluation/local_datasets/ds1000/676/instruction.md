Problem:
I'm using tensorflow 2.10.0.
I have a tensor of lengths in tensorflow, let's say it looks like this:
[4, 3, 5, 2]


I wish to create a mask of 1s and 0s whose number of 0s correspond to the entries to this tensor, padded by 1s to a total length of 8. I.e. I want to create this tensor:
[[0,0,0,0,1,1,1,1],
 [0,0,0,1,1,1,1,1],
 [0,0,0,0,0,1,1,1],
 [0,0,1,1,1,1,1,1]
]


How might I do this?


A:
<code>
import tensorflow as tf


lengths = [4, 3, 5, 2]
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import tensorflow as tf
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        lengths = data
        lengths_transposed = tf.expand_dims(lengths, 1)
        range = tf.range(0, 8, 1)
        range_row = tf.expand_dims(range, 0)
        mask = tf.less(range_row, lengths_transposed)
        result = tf.where(~mask, tf.ones([4, 8]), tf.zeros([4, 8]))
        return result

    def define_test_input(test_case_id):
        if test_case_id == 1:
            lengths = [4, 3, 5, 2]
        if test_case_id == 2:
            lengths = [2, 3, 4, 5]
        return lengths

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
lengths = test_input
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
