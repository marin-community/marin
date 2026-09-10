import matplotlib.pyplot as plt
import numpy as np

box_position, box_height, box_errors = np.arange(4), np.ones(4), np.arange(1, 5)
c = ["r", "r", "b", "b"]
fig, ax = plt.subplots()
ax.bar(box_position, box_height, color="yellow")

# Plot error bars with errors specified in box_errors. Use colors in c to color the error bars
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
    box_position, box_height, box_errors = np.arange(4), np.ones(4), np.arange(1, 5)
    c = ["r", "r", "b", "b"]
    fig, ax = plt.subplots()
    ax.bar(box_position, box_height, color="yellow")
    for pos, y, err, color in zip(box_position, box_height, box_errors, c):
        ax.errorbar(pos, y, err, color=color)
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
        assert len(ax.get_lines()) == 4
        line_colors = []
        for line in ax.get_lines():
            line_colors.append(line._color)
        assert set(line_colors) == set(c)
    return 1


exec_context = r"""
import matplotlib.pyplot as plt
import numpy as np
box_position, box_height, box_errors = np.arange(4), np.ones(4), np.arange(1, 5)
c = ["r", "r", "b", "b"]
fig, ax = plt.subplots()
ax.bar(box_position, box_height, color="yellow")
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
