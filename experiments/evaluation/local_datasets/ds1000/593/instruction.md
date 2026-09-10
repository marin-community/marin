import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.arange(0, 1000, 50)
y = np.arange(0, 1000, 50)

# plot y over x on a log-log plot
# mark the axes with numbers like 1, 10, 100. do not use scientific notation
# SOLUTION START

Execution context (contains [insert] placeholder):
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.ticker import ScalarFormatter


def skip_plt_cmds(l):
    return all(
        p not in l for p in ["plt.show()", "plt.clf()", "plt.close()", "savefig"]
    )


def generate_test_case(test_case_id):
    x = np.arange(0, 1000, 50)
    y = np.arange(0, 1000, 50)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.axis([1, 1000, 1, 1000])
    ax.loglog()
    for axis in [ax.xaxis, ax.yaxis]:
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)
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
        assert ax.get_yaxis().get_scale() == "log"
        assert ax.get_xaxis().get_scale() == "log"
        all_ticklabels = [l.get_text() for l in ax.get_xaxis().get_ticklabels()]
        for t in all_ticklabels:
            assert "$\mathdefault" not in t
        for l in ["1", "10", "100"]:
            assert l in all_ticklabels
        all_ticklabels = [l.get_text() for l in ax.get_yaxis().get_ticklabels()]
        for t in all_ticklabels:
            assert "$\mathdefault" not in t
        for l in ["1", "10", "100"]:
            assert l in all_ticklabels
    return 1


exec_context = r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
x = np.arange(0, 1000, 50)
y = np.arange(0, 1000, 50)
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
