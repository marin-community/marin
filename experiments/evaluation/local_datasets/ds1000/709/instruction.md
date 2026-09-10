Problem:
I'm using tensorflow 2.10.0.
I would like to generate 10 random integers as a tensor in TensorFlow but I don't which command I should use. In particular, I would like to generate from a uniform random variable which takes values in {1, 2, 3, 4}. I have tried to look among the distributions included in tensorflow_probability but I didn't find it.
Please set the random seed to 10 with tf.random.ser_seed().
Thanks in advance for your help.

A:
<code>
import tensorflow as tf

def f(seed_x=10):
    # return the solution in this function
    # result = f(seed_x)
    ### BEGIN SOLUTION

Execution context (contains [insert] placeholder):
import tensorflow as tf
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        seed_x = data
        tf.random.set_seed(seed_x)
        return tf.random.uniform(shape=(10,), minval=1, maxval=5, dtype=tf.int32)

    def define_test_input(test_case_id):
        if test_case_id == 1:
            seed_x = 10
        return seed_x

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
seed_x = test_input
def f(seed_x):
[insert]
result = f(seed_x)
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
