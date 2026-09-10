Problem:
I am struggling with the basic task of constructing a DataFrame of counts by value from a tuple produced by np.unique(arr, return_counts=True), such as:
import numpy as np
import pandas as pd
np.random.seed(123)  
birds=np.random.choice(['African Swallow','Dead Parrot','Exploding Penguin'], size=int(5e4))
someTuple=np.unique(birds, return_counts = True)
someTuple
#(array(['African Swallow', 'Dead Parrot', 'Exploding Penguin'], 
#       dtype='<U17'), array([16510, 16570, 16920], dtype=int64))

First I tried
pd.DataFrame(list(someTuple))
# Returns this:
#                  0            1                  2
# 0  African Swallow  Dead Parrot  Exploding Penguin
# 1            16510        16570              16920

I also tried pd.DataFrame.from_records(someTuple), which returns the same thing.
But what I'm looking for is this:
#              birdType      birdCount
# 0     African Swallow          16510  
# 1         Dead Parrot          16570  
# 2   Exploding Penguin          16920

What's the right syntax?

A:
<code>
import numpy as np
import pandas as pd

np.random.seed(123)
birds = np.random.choice(['African Swallow', 'Dead Parrot', 'Exploding Penguin'], size=int(5e4))
someTuple = np.unique(birds, return_counts=True)
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        someTuple = data
        return pd.DataFrame(
            np.column_stack(someTuple), columns=["birdType", "birdCount"]
        )

    def define_test_input(test_case_id):
        if test_case_id == 1:
            np.random.seed(123)
            birds = np.random.choice(
                ["African Swallow", "Dead Parrot", "Exploding Penguin"], size=int(5e4)
            )
            someTuple = np.unique(birds, return_counts=True)
        return someTuple

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        pd.testing.assert_frame_equal(result, ans, check_dtype=False)
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
someTuple = test_input
[insert]
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
