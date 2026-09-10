import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    "reports": [4, 24, 31, 2, 3],
    "coverage": [35050800, 54899767, 57890789, 62890798, 70897871],
}
df = pd.DataFrame(data)
sns.catplot(y="coverage", x="reports", kind="bar", data=df, label="Total")

# do not use scientific notation in the y axis ticks labels
# SOLUTION START

Execution context (contains [insert] placeholder):
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


def skip_plt_cmds(l):
    return all(
        p not in l for p in ["plt.show()", "plt.clf()", "plt.close()", "savefig"]
    )


def generate_test_case(test_case_id):
    data = {
        "reports": [4, 24, 31, 2, 3],
        "coverage": [35050800, 54899767, 57890789, 62890798, 70897871],
    }
    df = pd.DataFrame(data)
    sns.catplot(y="coverage", x="reports", kind="bar", data=df, label="Total")
    plt.ticklabel_format(style="plain", axis="y")
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
        assert len(ax.get_yticklabels()) > 0
        for l in ax.get_yticklabels():
            if int(l.get_text()) > 0:
                assert int(l.get_text()) > 1000
            assert "e" not in l.get_text()
    return 1


exec_context = r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = {
    "reports": [4, 24, 31, 2, 3],
    "coverage": [35050800, 54899767, 57890789, 62890798, 70897871],
}
df = pd.DataFrame(data)
sns.catplot(y="coverage", x="reports", kind="bar", data=df, label="Total")
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
