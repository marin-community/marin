Problem:
I'm using tensorflow 2.10.0.
So I'm creating a tensorflow model and for the forward pass, I'm applying my forward pass method to get the scores tensor which contains the prediction scores for each class. The shape of this tensor is [100, 10]. Now, I want to get the accuracy by comparing it to y which contains the actual scores. This tensor has the shape [10]. To compare the two I'll be using torch.mean(scores == y) and I'll count how many are the same. 
The problem is that I need to convert the scores tensor so that each row simply contains the index of the highest value in each column. For example if the tensor looked like this,
tf.Tensor(
    [[0.3232, -0.2321, 0.2332, -0.1231, 0.2435, 0.6728],
    [0.2323, -0.1231, -0.5321, -0.1452, 0.5435, 0.1722],
    [0.9823, -0.1321, -0.6433, 0.1231, 0.023, 0.0711]]
)


Then I'd want it to be converted so that it looks like this. 
tf.Tensor([2 1 0 2 1 0])


How could I do that? 


A:
<code>
import tensorflow as tf


a = tf.constant(
    [[0.3232, -0.2321, 0.2332, -0.1231, 0.2435, 0.6728],
     [0.2323, -0.1231, -0.5321, -0.1452, 0.5435, 0.1722],
     [0.9823, -0.1321, -0.6433, 0.1231, 0.023, 0.0711]]
)
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import tensorflow as tf
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        a = data
        return tf.argmax(a, axis=0)

    def define_test_input(test_case_id):
        if test_case_id == 1:
            a = tf.constant(
                [
                    [0.3232, -0.2321, 0.2332, -0.1231, 0.2435, 0.6728],
                    [0.2323, -0.1231, -0.5321, -0.1452, 0.5435, 0.1722],
                    [0.9823, -0.1321, -0.6433, 0.1231, 0.023, 0.0711],
                ]
            )
        if test_case_id == 2:
            a = tf.constant(
                [
                    [0.3232, -0.2321, 0.2332, -0.1231, 0.2435],
                    [0.2323, -0.1231, -0.5321, -0.1452, 0.5435],
                    [0.9823, -0.1321, -0.6433, 0.1231, 0.023],
                ]
            )
        return a

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
a = test_input
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
