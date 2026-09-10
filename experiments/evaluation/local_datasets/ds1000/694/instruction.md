Problem:
I'm using tensorflow 2.10.0.
I have two 3D tensors, tensor A which has shape [B,N,S] and tensor B which also has shape [B,N,S]. What I want to get is a third tensor C, which I expect to have [B,B,N] shape, where the element C[i,j,k] = np.dot(A[i,k,:], B[j,k,:]. I also want to achieve this is a vectorized way.
Some further info: The two tensors A and B have shape [Batch_size, Num_vectors, Vector_size]. The tensor C, is supposed to represent the dot product between each element in the batch from A and each element in the batch from B, between all of the different vectors.
Hope that it is clear enough and looking forward to you answers!


A:
<code>
import tensorflow as tf
import numpy as np


np.random.seed(10)
A = tf.constant(np.random.randint(low=0, high=5, size=(10, 20, 30)))
B = tf.constant(np.random.randint(low=0, high=5, size=(10, 20, 30)))
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import tensorflow as tf
import numpy as np
import copy
import tokenize, io


def generate_test_case(test_case_id):
    def generate_ans(data):
        A, B = data
        return tf.constant(np.einsum("ikm, jkm-> ijk", A, B))

    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(10)
            A = tf.constant(np.random.randint(low=0, high=5, size=(10, 20, 30)))
            B = tf.constant(np.random.randint(low=0, high=5, size=(10, 20, 30)))
        return A, B

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
import numpy as np
A,B = test_input
[insert]
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
    assert "for" not in tokens and "while" not in tokens

Return only the Python code snippet that replaces [insert]:
- No explanations, Markdown, comments, prints, or extra text; no non-English output
- No imports, function/class definitions, or placeholders; only the missing logic
- Do not reassign/reset provided variables/inputs; use them to compute the result
- The snippet must be syntactically valid and run as-is in this context; deterministic (assume temperature=0)
- Do not read/write files or call external services
- Output only the code lines for [insert]; no Markdown fences, no explanations, no bullets.
