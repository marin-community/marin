import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.arange(10)
y = np.random.rand(10)
z = np.random.rand(10)
a = np.arange(10)

# Make two subplots
# Plot y over x in the first subplot and plot z over a in the second subplot
# Label each line chart and put them into a single legend on the first subplot
# SOLUTION START

Execution context (contains [insert] placeholder):
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def skip_plt_cmds(l):
    return all(
        p not in l for p in ["plt.show()", "plt.clf()", "plt.close()", "savefig"]
    )


def generate_test_case(test_case_id):
    x = np.arange(10)
    y = np.random.rand(10)
    z = np.random.rand(10)
    a = np.arange(10)
    fig, ax = plt.subplots(2, 1)
    (l1,) = ax[0].plot(x, y, color="red", label="y")
    (l2,) = ax[1].plot(a, z, color="blue", label="z")
    ax[0].legend([l1, l2], ["z", "y"])
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
        f = plt.gcf()
        axes = np.array(f.get_axes())
        axes = axes.reshape(-1)
        assert len(axes) == 2
        l = axes[0].get_legend()
        assert l is not None
        assert len(l.get_texts()) == 2
        assert len(axes[0].get_lines()) == 1
        assert len(axes[1].get_lines()) == 1
    return 1


exec_context = r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
x = np.arange(10)
y = np.random.rand(10)
z = np.random.rand(10)
a = np.arange(10)
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
