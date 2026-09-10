Problem:
I have a set of objects and their positions over time. I would like to get the distance between each car and their nearest neighbour, and calculate an average of this for each time point. An example dataframe is as follows:
 time = [0, 0, 0, 1, 1, 2, 2]
 x = [216, 218, 217, 280, 290, 130, 132]
 y = [13, 12, 12, 110, 109, 3, 56]
 car = [1, 2, 3, 1, 3, 4, 5]
 df = pd.DataFrame({'time': time, 'x': x, 'y': y, 'car': car})
 df
         x       y      car
 time
  0     216     13       1
  0     218     12       2
  0     217     12       3
  1     280     110      1
  1     290     109      3
  2     130     3        4
  2     132     56       5


For each time point, I would like to know the nearest car neighbour for each car. Example:
df2
          car    nearest_neighbour    euclidean_distance  
 time
  0       1            3                    1.41
  0       2            3                    1.00
  0       3            2                    1.00
  1       1            3                    10.05
  1       3            1                    10.05
  2       4            5                    53.04
  2       5            4                    53.04


I know I can calculate the pairwise distances between cars from How to apply euclidean distance function to a groupby object in pandas dataframe? but how do I get the nearest neighbour for each car? 
After that it seems simple enough to get an average of the distances for each frame using groupby, but it's the second step that really throws me off. 
Help appreciated!


A:
<code>
import pandas as pd


time = [0, 0, 0, 1, 1, 2, 2]
x = [216, 218, 217, 280, 290, 130, 132]
y = [13, 12, 12, 110, 109, 3, 56]
car = [1, 2, 3, 1, 3, 4, 5]
df = pd.DataFrame({'time': time, 'x': x, 'y': y, 'car': car})
</code>
df = ... # put solution in this variable
BEGIN SOLUTION
<code>

Execution context (contains [insert] placeholder):
import pandas as pd
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        time = df.time.tolist()
        car = df.car.tolist()
        nearest_neighbour = []
        euclidean_distance = []
        for i in range(len(df)):
            n = 0
            d = np.inf
            for j in range(len(df)):
                if (
                    df.loc[i, "time"] == df.loc[j, "time"]
                    and df.loc[i, "car"] != df.loc[j, "car"]
                ):
                    t = np.sqrt(
                        ((df.loc[i, "x"] - df.loc[j, "x"]) ** 2)
                        + ((df.loc[i, "y"] - df.loc[j, "y"]) ** 2)
                    )
                    if t < d:
                        d = t
                        n = df.loc[j, "car"]
            nearest_neighbour.append(n)
            euclidean_distance.append(d)
        return pd.DataFrame(
            {
                "time": time,
                "car": car,
                "nearest_neighbour": nearest_neighbour,
                "euclidean_distance": euclidean_distance,
            }
        )

    def define_test_input(test_case_id):
        if test_case_id == 1:
            time = [0, 0, 0, 1, 1, 2, 2]
            x = [216, 218, 217, 280, 290, 130, 132]
            y = [13, 12, 12, 110, 109, 3, 56]
            car = [1, 2, 3, 1, 3, 4, 5]
            df = pd.DataFrame({"time": time, "x": x, "y": y, "car": car})
        if test_case_id == 2:
            time = [0, 0, 0, 1, 1, 2, 2]
            x = [219, 219, 216, 280, 290, 130, 132]
            y = [15, 11, 14, 110, 109, 3, 56]
            car = [1, 2, 3, 1, 3, 4, 5]
            df = pd.DataFrame({"time": time, "x": x, "y": y, "car": car})
        return df

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        result.euclidean_distance = np.round(result.euclidean_distance, 2)
        ans.euclidean_distance = np.round(ans.euclidean_distance, 2)
        pd.testing.assert_frame_equal(result, ans, check_dtype=False)
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
df = test_input
[insert]
result = df
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
