"""Exercise native graders after the same imports as the evaluation CLI."""

import asyncio
import json
from pathlib import Path
import sys

import eval.eval  # noqa: F401  # Spawn must tolerate the real CLI's imports.
from eval.contracts.eval_spec import EvalSpec, GraderSpec, SourceRef
from eval.execution.grading import Graders
from eval.graders.specs import GRADERS

PAYLOAD = "x" * (8 * 1024 ** 2)
ALLOCATION = "import numpy as np\nassert np.add(1, 2) == 3\nbuffer = bytearray(64 * 1024 * 1024)\n"


def lcb_spec(mode):
    test = {"input": PAYLOAD + "\n", "output": str(len(PAYLOAD)), "testtype": "stdin"}
    if mode == "functional":
        test = {"input": [PAYLOAD], "output": len(PAYLOAD), "testtype": "functional"}
    return EvalSpec(
        "LiveCodeBench", SourceRef("native-pool-control"), mode, 0, (),
        GraderSpec("live_code", {"timeout": 6}), doc={"test": [test]}, target=[test],
    )


def completion(correct, mode="stdin"):
    if mode == "functional":
        code = "def answer(value):\n" + "".join("    " + line + "\n" for line in ALLOCATION.splitlines())
        code += "    return " + ("len(value)" if correct else "0")
    else:
        code = ALLOCATION + "print(" + ("len(input())" if correct else "0") + ")"
    return "```python\n" + code + "\n```"


async def run():
    graders = Graders(GRADERS, workers=1)
    rows = []

    async def grade(name, spec, output):
        result = await graders(spec, [output])
        rows.append((name, result.metrics["accuracy"]))

    try:
        for correct in (True, False):
            await grade(f"lcb_initial_{correct}", lcb_spec("stdin"), completion(correct))
        problem = {"task_id": "python/0", "prompt": "def answer():\n", "test": "assert answer() == 42"}
        human = EvalSpec(
            "HumanEval", SourceRef("native-pool-control"), "answer", 0, (),
            GraderSpec("program_suite", {"suite": "HumanEval", "language": "python", "problem": problem, "timeout": 5.0}),
            doc=problem,
        )
        for value in (42, 41):
            await grade(f"humaneval_{value}", human, f"```python\ndef answer(): return {value}\n```")
        question = {
            "question_id": "pool-amps", "category": "math", "task": "amps_hard", "turns": ["Compute 6*7"],
            "ground_truth": "42", "livebench_release_date": "2024-07-26",
        }
        amps = EvalSpec("LiveBench", SourceRef("native-pool-control"), "pool-amps", 0, (), GraderSpec("livebench"), doc=question)
        for value in (42, 41):
            await grade(f"livebench_amps_{value}", amps, f"\\boxed{{{value}}}")
        for mode in ("stdin", "functional"):
            for correct in (True, False):
                await grade(f"lcb_after_imports_{mode}_{correct}", lcb_spec(mode), completion(correct, mode))
        question = {
            "question_id": "pool-coding", "category": "coding", "task": "LCB_generation", "turns": ["Print the input length"],
            "public_test_cases": json.dumps([{"input": PAYLOAD + "\n", "output": str(len(PAYLOAD)), "testtype": "stdin"}]),
            "private_test_cases": "[]", "original_json": {"metadata": "{}"}, "livebench_release_date": "2024-07-26",
        }
        coding = EvalSpec("LiveBench", SourceRef("native-pool-control"), "pool-coding", 0, (), GraderSpec("livebench"), doc=question)
        for correct in (True, False):
            await grade(f"livebench_coding_{correct}", coding, completion(correct))
    finally:
        graders.close()
    return rows


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(json.dumps(asyncio.run(run())))
