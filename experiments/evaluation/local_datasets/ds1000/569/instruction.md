import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(
    {
        "celltype": ["foo", "bar", "qux", "woz"],
        "s1": [5, 9, 1, 7],
        "s2": [12, 90, 13, 87],
    }
)

# For data in df, make a bar plot of s1 and s1 and use celltype as the xlabel
# Make the x-axis tick labels rotate 45 degrees
# SOLUTION START

Execution context (contains [insert] placeholder):
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np


def skip_plt_cmds(l):
    return all(
        p not in l for p in ["plt.show()", "plt.clf()", "plt.close()", "savefig"]
    )


def generate_test_case(test_case_id):
    df = pd.DataFrame(
        {
            "celltype": ["foo", "bar", "qux", "woz"],
            "s1": [5, 9, 1, 7],
            "s2": [12, 90, 13, 87],
        }
    )
    df = df[["celltype", "s1", "s2"]]
    df.set_index(["celltype"], inplace=True)
    df.plot(kind="bar", alpha=0.75, rot=45)
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
        plt.show()
        assert len(ax.patches) > 0
        assert len(ax.xaxis.get_ticklabels()) > 0
        for t in ax.xaxis.get_ticklabels():
            assert t._rotation == 45
        all_ticklabels = [t.get_text() for t in ax.xaxis.get_ticklabels()]
        for cell in ["foo", "bar", "qux", "woz"]:
            assert cell in all_ticklabels
    return 1


exec_context = r"""
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
df = pd.DataFrame(
    {
        "celltype": ["foo", "bar", "qux", "woz"],
        "s1": [5, 9, 1, 7],
        "s2": [12, 90, 13, 87],
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
