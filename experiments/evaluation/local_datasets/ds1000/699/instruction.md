Problem:
I'm using tensorflow 2.10.0.
I've come across a case in which the averaging includes padded values. Given a tensor X of some shape (batch_size, ..., features), there could be zero padded features to get the same shape.
How can I variance the second to last dimension of X (the features) but only the non-zero entries? Example input:
x = [[[[1,2,3], [2,3,4], [0,0,0]],
       [[1,2,3], [2,0,4], [3,4,5]],
       [[1,2,3], [0,0,0], [0,0,0]],
       [[1,2,3], [1,2,3], [0,0,0]]],
      [[[1,2,3], [0,1,0], [0,0,0]],
       [[1,2,3], [2,3,4], [0,0,0]],                                                         
       [[1,2,3], [0,0,0], [0,0,0]],                                                         
       [[1,2,3], [1,2,3], [1,2,3]]]]
# Desired output
y = [[[0.25       0.25       0.25      ]
  [0.6666665  1.         0.66666603]
  [0.         0.         0.        ]
  [0.         0.         0.        ]]

 [[0.         0.25       0.        ]
  [0.25       0.25       0.25      ]
  [0.         0.         0.        ]
  [0.         0.         0.        ]]]

A:
<code>
import tensorflow as tf

x = [[[[1, 2, 3], [2, 3, 4], [0, 0, 0]],
      [[1, 2, 3], [2, 0, 4], [3, 4, 5]],
      [[1, 2, 3], [0, 0, 0], [0, 0, 0]],
      [[1, 2, 3], [1, 2, 3], [0, 0, 0]]],
     [[[1, 2, 3], [0, 1, 0], [0, 0, 0]],
      [[1, 2, 3], [2, 3, 4], [0, 0, 0]],
      [[1, 2, 3], [0, 0, 0], [0, 0, 0]],
      [[1, 2, 3], [1, 2, 3], [1, 2, 3]]]]
x = tf.convert_to_tensor(x, dtype=tf.float32)
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import tensorflow as tf
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        x = data
        non_zero = tf.cast(x != 0, tf.float32)
        y = tf.reduce_sum(x, axis=-2) / tf.reduce_sum(non_zero, axis=-2)
        y = y * y
        z = tf.reduce_sum(x * x, axis=-2) / tf.reduce_sum(non_zero, axis=-2)
        return z - y

    def define_test_input(test_case_id):
        if test_case_id == 1:
            x = [
                [
                    [[1, 2, 3], [2, 3, 4], [0, 0, 0]],
                    [[1, 2, 3], [2, 0, 4], [3, 4, 5]],
                    [[1, 2, 3], [0, 0, 0], [0, 0, 0]],
                    [[1, 2, 3], [1, 2, 3], [0, 0, 0]],
                ],
                [
                    [[1, 2, 3], [0, 1, 0], [0, 0, 0]],
                    [[1, 2, 3], [2, 3, 4], [0, 0, 0]],
                    [[1, 2, 3], [0, 0, 0], [0, 0, 0]],
                    [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
                ],
            ]
            x = tf.convert_to_tensor(x, dtype=tf.float32)
        if test_case_id == 2:
            x = [
                [
                    [[1, 2, 3], [2, 3, 4], [0, 0, 0]],
                    [[1, 2, 3], [2, 0, 4], [3, 4, 5]],
                    [[1, 2, 3], [0, 0, 0], [0, 0, 0]],
                    [[1, 2, 3], [1, 2, 3], [0, 0, 0]],
                ],
                [
                    [[1, 2, 3], [0, 1, 0], [0, 0, 0]],
                    [[1, 2, 3], [2, 3, 4], [0, 0, 0]],
                    [[1, 2, 3], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [1, 2, 3], [1, 2, 3]],
                ],
            ]
            x = tf.convert_to_tensor(x, dtype=tf.float32)
        return x

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
x = test_input
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
