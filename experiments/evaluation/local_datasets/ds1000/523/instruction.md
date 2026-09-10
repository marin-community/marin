import numpy
import pandas
import matplotlib.pyplot as plt
import seaborn

seaborn.set(style="ticks")

numpy.random.seed(0)
N = 37
_genders = ["Female", "Male", "Non-binary", "No Response"]
df = pandas.DataFrame(
    {
        "Height (cm)": numpy.random.uniform(low=130, high=200, size=N),
        "Weight (kg)": numpy.random.uniform(low=30, high=100, size=N),
        "Gender": numpy.random.choice(_genders, size=N),
    }
)

# make seaborn relation plot and color by the gender field of the dataframe df
# SOLUTION START

Execution context (contains [insert] placeholder):
import numpy
import pandas
import matplotlib.pyplot as plt
import seaborn
from PIL import Image
import numpy as np


def skip_plt_cmds(l):
    return all(
        p not in l for p in ["plt.show()", "plt.clf()", "plt.close()", "savefig"]
    )


def generate_test_case(test_case_id):
    seaborn.set(style="ticks")
    numpy.random.seed(0)
    N = 37
    _genders = ["Female", "Male", "Non-binary", "No Response"]
    df = pandas.DataFrame(
        {
            "Height (cm)": numpy.random.uniform(low=130, high=200, size=N),
            "Weight (kg)": numpy.random.uniform(low=30, high=100, size=N),
            "Gender": numpy.random.choice(_genders, size=N),
        }
    )
    seaborn.relplot(
        data=df, x="Weight (kg)", y="Height (cm)", hue="Gender", hue_order=_genders
    )
    plt.savefig("ans.png", bbox_inches="tight")
    plt.close()
    return None, None


def exec_test(result, ans):
    code_img = np.array(Image.open("output.png"))
    oracle_img = np.array(Image.open("ans.png"))
    sample_image_stat = code_img.shape == oracle_img.shape and np.allclose(
        code_img, oracle_img
    )
    if not sample_image_stat:
        ax = plt.gca()
        all_colors = set()
        for c in ax.collections:
            colors = c.get_facecolor()
            for i in range(colors.shape[0]):
                all_colors.add(tuple(colors[i]))
        assert len(all_colors) == 4
        assert ax.get_xlabel() == "Weight (kg)"
        assert ax.get_ylabel() == "Height (cm)"
    return 1


exec_context = r"""
import numpy
import pandas
import matplotlib.pyplot as plt
import seaborn
seaborn.set(style="ticks")
numpy.random.seed(0)
N = 37
_genders = ["Female", "Male", "Non-binary", "No Response"]
df = pandas.DataFrame(
    {
        "Height (cm)": numpy.random.uniform(low=130, high=200, size=N),
        "Weight (kg)": numpy.random.uniform(low=30, high=100, size=N),
        "Gender": numpy.random.choice(_genders, size=N),
    }
)
[insert]
plt.savefig('output.png', bbox_inches ='tight')
result = None
"""


def test_execution(solution: str):
    solution = "\n".join(filter(skip_plt_cmds, solution.split("\n")))
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
