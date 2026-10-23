"""Executable Python extracted from the Plus code benchmarks."""

import asyncio
import pytest

from eval.chat_benchmarks.HumanEvalPlus.utils.utils import extract_generation_code
from eval.contracts.eval_spec import EvalSpec, GraderSpec, SourceRef
from eval.graders.program_suites import program_suite
from eval.graders.specs import GRADERS
from eval.execution.grading import Graders


@pytest.mark.parametrize("suite", ["HumanEval", "MBPP"])
@pytest.mark.parametrize("answer,score", [(42, 1.0), (41, 0.0)])
def test_fork_suites_execute_through_shared_grading(suite, answer, score):
    problem = {
        "task_id": "python/0", "prompt": "def answer():\n",
        "test": ["assert answer() == 42"] if suite == "MBPP" else "assert answer() == 42",
    }
    spec = EvalSpec(
        task=suite, source=SourceRef("fixture"), source_id="0", ordinal=0, requests=(),
        grader=GraderSpec("program_suite", {
            "suite": suite, "language": "python", "problem": problem,
            "timeout": 1.0, "mbpp": suite == "MBPP",
        }), doc=problem,
    )
    graders = Graders(GRADERS, workers=2)
    try:
        result = asyncio.run(graders(spec, [f"```python\ndef answer():\n    return {answer}\n``` "]))
    finally:
        graders.close()
    assert result.metrics["accuracy"] == score


@pytest.mark.parametrize(
    "completion,function,args,expected",
    [
        ("```python\n    def add(a, b):\n        return a + b\n```", "add", (2, 3), 5),
        ("    def add(a, b):\n        return a + b\n", "add", (2, 3), 5),
        (
            "```python\n    def similar_elements(list1, list2):\n        return set(list1) & set(list2)\n    ```",
            "similar_elements",
            ((3, 4, 5), (4, 5, 7)),
            {4, 5},
        ),
        (
            "```python\n   import math\n   def is_not_prime(n):\n       if n == 1:\n           return True\n"
            "       for i in range(2, int(math.sqrt(n))+1):\n           if n % i == 0:\n"
            "               return True\n       return False\n   ```",
            "is_not_prime",
            (35,),
            True,
        ),
    ],
)
def test_mbppplus_indented_answers_execute(completion, function, args, expected):
    arguments = ", ".join(repr(value) for value in args)
    problem = {
        "task_id": "MBPP/test",
        "prompt": f"def {function}(*args):\n",
        "test": f"assert {function}({arguments}) == {expected!r}",
    }
    spec = EvalSpec(
        task="MBPPPlus",
        source=SourceRef("fixture"),
        source_id="test",
        ordinal=0,
        requests=(),
        grader=GraderSpec("program_suite", {"suite": "MBPPPlus", "language": "python", "problem": problem, "timeout": 3.0}),
        doc=problem,
    )

    assert program_suite(spec, [completion]).metrics["accuracy"] == 1.0


@pytest.mark.parametrize(
    "completion,prompt",
    [
        (
            "```python\n    import math\n    def ceiling(x):\n        return math.ceil(x)\n```",
            "def ceiling(x):\n    \"\"\"Round x up.\"\"\"\n",
        ),
        (
            "```python\n    return math.ceil(x)\n```",
            "import math\n\ndef ceiling(x):\n    \"\"\"Round x up.\"\"\"\n",
        ),
        (
            "    def ceiling(x):\n        return math.ceil(x)\n",
            "import math\n\ndef ceiling(x):\n    \"\"\"Round x up.\"\"\"\n",
        ),
    ],
)
def test_humanevalplus_indented_python_solution_executes(completion, prompt):
    example = {
        "task_id": "HumanEval/test",
        "prompt": prompt,
        "output": completion,
    }

    namespace = {}
    exec(extract_generation_code(example, lang_code="python")["generation"], namespace)

    assert namespace["ceiling"](1.2) == 2
