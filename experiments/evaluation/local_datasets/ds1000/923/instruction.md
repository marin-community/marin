Problem:

I have a data which include dates in sorted order.

I would like to split the given data to train and test set. However, I must to split the data in a way that the test have to be newer than the train set.

Please look at the given example:

Let's assume that we have data by dates:

1, 2, 3, ..., n.

The numbers from 1 to n represents the days.

I would like to split it to 20% from the data to be train set and 80% of the data to be test set.

Good results:

1) train set = 1, 2, 3, ..., 20

   test set = 21, ..., 100


2) train set = 101, 102, ... 120

    test set = 121, ... 200
My code:

train_size = 0.2
train_dataframe, test_dataframe = cross_validation.train_test_split(features_dataframe, train_size=train_size)

train_dataframe = train_dataframe.sort(["date"])
test_dataframe = test_dataframe.sort(["date"])
Does not work for me!

Any suggestions?

A:

<code>
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
features_dataframe = load_data()
def solve(features_dataframe):
    # return the solution in this function
    # train_dataframe, test_dataframe = solve(features_dataframe)
    ### BEGIN SOLUTION

Execution context (contains [insert] placeholder):
import pandas as pd
import datetime
import copy


def generate_test_case(test_case_id):
    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {
                    "date": [
                        "2017-03-01",
                        "2017-03-02",
                        "2017-03-03",
                        "2017-03-04",
                        "2017-03-05",
                        "2017-03-06",
                        "2017-03-07",
                        "2017-03-08",
                        "2017-03-09",
                        "2017-03-10",
                    ],
                    "sales": [
                        12000,
                        8000,
                        25000,
                        15000,
                        10000,
                        15000,
                        10000,
                        25000,
                        12000,
                        15000,
                    ],
                    "profit": [
                        18000,
                        12000,
                        30000,
                        20000,
                        15000,
                        20000,
                        15000,
                        30000,
                        18000,
                        20000,
                    ],
                }
            )
        elif test_case_id == 2:
            df = pd.DataFrame(
                {
                    "date": [
                        datetime.datetime(2020, 7, 1),
                        datetime.datetime(2020, 7, 2),
                        datetime.datetime(2020, 7, 3),
                        datetime.datetime(2020, 7, 4),
                        datetime.datetime(2020, 7, 5),
                        datetime.datetime(2020, 7, 6),
                        datetime.datetime(2020, 7, 7),
                        datetime.datetime(2020, 7, 8),
                        datetime.datetime(2020, 7, 9),
                        datetime.datetime(2020, 7, 10),
                        datetime.datetime(2020, 7, 11),
                        datetime.datetime(2020, 7, 12),
                        datetime.datetime(2020, 7, 13),
                        datetime.datetime(2020, 7, 14),
                        datetime.datetime(2020, 7, 15),
                        datetime.datetime(2020, 7, 16),
                        datetime.datetime(2020, 7, 17),
                        datetime.datetime(2020, 7, 18),
                        datetime.datetime(2020, 7, 19),
                        datetime.datetime(2020, 7, 20),
                        datetime.datetime(2020, 7, 21),
                        datetime.datetime(2020, 7, 22),
                        datetime.datetime(2020, 7, 23),
                        datetime.datetime(2020, 7, 24),
                        datetime.datetime(2020, 7, 25),
                        datetime.datetime(2020, 7, 26),
                        datetime.datetime(2020, 7, 27),
                        datetime.datetime(2020, 7, 28),
                        datetime.datetime(2020, 7, 29),
                        datetime.datetime(2020, 7, 30),
                        datetime.datetime(2020, 7, 31),
                    ],
                    "counts": [
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                        11,
                        12,
                        13,
                        14,
                        15,
                        16,
                        17,
                        18,
                        19,
                        20,
                        21,
                        22,
                        23,
                        24,
                        25,
                        26,
                        27,
                        28,
                        29,
                        30,
                        31,
                    ],
                }
            )
        return df

    def generate_ans(data):
        features_dataframe = data
        n = features_dataframe.shape[0]
        train_size = 0.2
        train_dataframe = features_dataframe.iloc[: int(n * train_size)]
        test_dataframe = features_dataframe.iloc[int(n * train_size) :]
        return train_dataframe, test_dataframe

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        pd.testing.assert_frame_equal(result[0], ans[0], check_dtype=False)
        pd.testing.assert_frame_equal(result[1], ans[1], check_dtype=False)
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
features_dataframe = test_input
def solve(features_dataframe):
[insert]
result = solve(features_dataframe)
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
