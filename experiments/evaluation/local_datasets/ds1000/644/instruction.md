from matplotlib import pyplot as plt
import numpy as np

x = np.arange(10)
y = np.arange(1, 11)
error = np.random.random(y.shape)

# Plot y over x and show the error according to `error`
# Plot the error as a shaded region rather than error bars
# SOLUTION START

Execution context (contains [insert] placeholder):
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import matplotlib


def skip_plt_cmds(l):
    return all(
        p not in l for p in ["plt.show()", "plt.clf()", "plt.close()", "savefig"]
    )


def generate_test_case(test_case_id):
    x = np.arange(10)
    y = np.arange(1, 11)
    error = np.random.random(y.shape)
    plt.plot(x, y, "k-")
    plt.fill_between(x, y - error, y + error)
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
        assert len(ax.lines) == 1
        assert len(ax.collections) == 1
        assert isinstance(ax.collections[0], matplotlib.collections.PolyCollection)
    return 1


exec_context = r"""
from matplotlib import pyplot as plt
import numpy as np
x = np.arange(10)
y = np.arange(1, 11)
error = np.random.random(y.shape)
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
