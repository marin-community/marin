import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

x = 10 * np.random.randn(10)

plt.plot(x)

# highlight in red the x range 2 to 4
# SOLUTION START

Execution context (contains [insert] placeholder):
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import matplotlib


def skip_plt_cmds(l):
    return all(
        p not in l for p in ["plt.show()", "plt.clf()", "plt.close()", "savefig"]
    )


def generate_test_case(test_case_id):
    x = 10 * np.random.randn(10)
    plt.plot(x)
    plt.axvspan(2, 4, color="red", alpha=1)
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
        assert len(ax.patches) == 1
        assert isinstance(ax.patches[0], matplotlib.patches.Polygon)
        assert ax.patches[0].get_xy().min(axis=0)[0] == 2
        assert ax.patches[0].get_xy().max(axis=0)[0] == 4
        assert ax.patches[0].get_facecolor()[0] > 0
        assert ax.patches[0].get_facecolor()[1] < 0.1
        assert ax.patches[0].get_facecolor()[2] < 0.1
    return 1


exec_context = r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
x = 10 * np.random.randn(10)
plt.plot(x)
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
