Problem:
I'm using tensorflow 2.10.0.
I have two embeddings tensor A and B, which looks like
[
  [1,1,1],
  [1,1,1]
]


and 
[
  [0,0,0],
  [1,1,1]
]


what I want to do is calculate the L2 distance d(A,B) element-wise. 
First I did a tf.square(tf.sub(lhs, rhs)) to get
[
  [1,1,1],
  [0,0,0]
]


and then I want to do an element-wise reduce which returns 
[
  3,
  0
]


but tf.reduce_sum does not allow my to reduce by row. Any inputs would be appreciated. Thanks.


A:
<code>
import tensorflow as tf

example_a = tf.constant([
  [1,1,1],
  [1,1,1]
])
example_b = tf.constant([
  [0,0,0],
  [1,1,1]
])
def f(A=example_a,B=example_b):
    # return the solution in this function
    # result = f(A,B)
    ### BEGIN SOLUTION

Execution context (contains [insert] placeholder):
import tensorflow as tf
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        a, b = data
        return tf.reduce_sum(tf.square(tf.subtract(a, b)), 1)

    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = tf.constant([[1, 1, 1], [1, 1, 1]])
            b = tf.constant([[0, 0, 0], [1, 1, 1]])
        if test_case_id == 2:
            a = tf.constant([[0, 1, 1], [1, 0, 1]])
            b = tf.constant([[0, 0, 0], [1, 1, 1]])
        return a, b

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
A,B = test_input
def f(A,B):
[insert]
result = f(A,B)
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
