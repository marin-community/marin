import matplotlib.pyplot as plt
import numpy as np

# Specify the values of blue bars (height)
blue_bar = (23, 25, 17)
# Specify the values of orange bars (height)
orange_bar = (19, 18, 14)

# Plot the blue bar and the orange bar side-by-side in the same bar plot.
# Make  sure the bars don't overlap with each other.
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
    blue_bar = (23, 25, 17)
    orange_bar = (19, 18, 14)
    ind = np.arange(len(blue_bar))
    plt.figure(figsize=(10, 5))
    width = 0.3
    plt.bar(ind, blue_bar, width, label="Blue bar label")
    plt.bar(ind + width, orange_bar, width, label="Orange bar label")
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
        assert len(ax.patches) == 6
        x_positions = [rec.get_x() for rec in ax.patches]
        assert len(x_positions) == len(set(x_positions))
    return 1


exec_context = r"""
import matplotlib.pyplot as plt
import numpy as np
blue_bar = (23, 25, 17)
orange_bar = (19, 18, 14)
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
