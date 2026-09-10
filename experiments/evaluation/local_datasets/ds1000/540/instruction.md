import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

y = 2 * np.random.rand(10)
x = np.arange(10)
plt.plot(x, y)
myTitle = "Some really really long long long title I really really need - and just can't - just can't - make it any - simply any - shorter - at all."

# fit a very long title myTitle into multiple lines
# SOLUTION START

Execution context (contains [insert] placeholder):
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import wrap
from PIL import Image


def skip_plt_cmds(l):
    return all(
        p not in l for p in ["plt.show()", "plt.clf()", "plt.close()", "savefig"]
    )


def generate_test_case(test_case_id):
    y = 2 * np.random.rand(10)
    x = np.arange(10)
    plt.plot(x, y)
    myTitle = (
        "Some really really long long long title I really really need - and just can't - just can't - make it "
        "any - simply any - shorter - at all."
    )
    ax = plt.gca()
    ax.set_title("\n".join(wrap(myTitle, 60)), loc="center", wrap=True)
    plt.savefig("ans.png", bbox_inches="tight")
    plt.close()
    return None, None


def exec_test(result, ans):
    myTitle = (
        "Some really really long long long title I really really need - and just can't - just can't - make it "
        "any - simply any - shorter - at all."
    )
    code_img = np.array(Image.open("output.png"))
    oracle_img = np.array(Image.open("ans.png"))
    sample_image_stat = code_img.shape == oracle_img.shape and np.allclose(
        code_img, oracle_img
    )
    if not sample_image_stat:
        fg = plt.gcf()
        assert fg.get_size_inches()[0] < 8
        ax = plt.gca()
        assert ax.get_title().startswith(myTitle[:10])
        assert "\n" in ax.get_title()
        assert len(ax.get_title()) >= len(myTitle)
    return 1


exec_context = r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
y = 2 * np.random.rand(10)
x = np.arange(10)
plt.plot(x, y)
myTitle = "Some really really long long long title I really really need - and just can't - just can't - make it any - simply any - shorter - at all."
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
