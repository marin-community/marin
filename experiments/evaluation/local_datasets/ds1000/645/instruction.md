import matplotlib.pyplot as plt
import numpy as np

xvec = np.linspace(-5.0, 5.0, 100)
x, y = np.meshgrid(xvec, xvec)
z = -np.hypot(x, y)
plt.contourf(x, y, z)

# draw x=0 and y=0 axis in my contour plot with white color
# SOLUTION START

Execution context (contains [insert] placeholder):
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def skip_plt_cmds(l):
    return all(
        p not in l for p in ["plt.show()", "plt.clf()", "plt.close()", "savefig"]
    )


def generate_test_case(test_case_id):
    xvec = np.linspace(-5.0, 5.0, 100)
    x, y = np.meshgrid(xvec, xvec)
    z = -np.hypot(x, y)
    plt.contourf(x, y, z)
    plt.axhline(0, color="white")
    plt.axvline(0, color="white")
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
        assert len(ax.lines) == 2
        for l in ax.lines:
            assert l._color == "white" or tuple(l._color) == (1, 1, 1, 1)
        horizontal = False
        vertical = False
        for l in ax.lines:
            if tuple(l.get_ydata()) == (0, 0):
                horizontal = True
        for l in ax.lines:
            if tuple(l.get_xdata()) == (0, 0):
                vertical = True
        assert horizontal and vertical
    return 1


exec_context = r"""
import matplotlib.pyplot as plt
import numpy as np
xvec = np.linspace(-5.0, 5.0, 100)
x, y = np.meshgrid(xvec, xvec)
z = -np.hypot(x, y)
plt.contourf(x, y, z)
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
