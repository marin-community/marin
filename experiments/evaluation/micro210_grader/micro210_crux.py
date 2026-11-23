# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical CRUXEval grading for the embedded Micro210 grader.

Adapted from taskcompendium's ``code_protocol.py`` asset at the #11928 fix commit
(assertion extraction splits at the first ``==`` so tuple arguments keep their closing
parentheses, and decoded turn markers are stripped before extraction), with the fork-based
candidate execution that replaces the runtime-v4 wheel's spawn-based harness (#11922).
"""

import ast
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eval.contracts.eval_spec import EvalSpec, Grade

CRUX_ANSWER_SPAN = re.compile(r"\[ANSWER\](.*?)\[/ANSWER\]", flags=re.DOTALL)
CRUX_TIMEOUT_SECONDS = 10

# From verifyit.adapters.completion at the #11928 fix commit; the first five markers are
# what the canonical CRUXEval extraction strips.
END_OF_TURN = (
    "<|im_end|>",
    "<|eot_id|>",
    "<|end_of_text|>",
    "",
    "</s>",
    "\nYou are an AI assistant",
    "\nQuestion:",
    "\nQ:",
    "\n[Question]",
    "\nUser:",
    "\nuser\n",
    "\nAssistant:",
)


def crux_answer(response: str, direction: str) -> str:
    """Extract the first answer block holding a CruxEval candidate."""
    response = response.rstrip()
    for marker in END_OF_TURN[:5]:
        if response.endswith(marker):
            response = response.removesuffix(marker).rstrip()
            break
    spans = [span.strip() for span in CRUX_ANSWER_SPAN.findall(response)]
    answer = spans[0] if spans else response
    for span in spans:
        candidate = _crux_expression(span, direction)
        if direction == "input":
            if "f(" in candidate:
                answer = span
                break
        else:
            try:
                ast.parse(candidate, mode="eval")
            except SyntaxError:
                continue
            answer = span
            break
    return _crux_expression(answer, direction)


def _crux_expression(answer: str, direction: str) -> str:
    """Read the call or literal on either side of the first assertion equality, as CRUXEval's harness does."""
    call, separator, literal = answer.partition("==")
    if direction == "input":
        call = call.strip()
        if "assert f" in call:
            # CRUXEval's harness (inference/tasks/input_prediction.py) keeps the call that follows "assert f".
            return "f" + call.split("assert f", 1)[1].strip()
        return call
    return (literal if separator else answer).strip()


def test_program(problem: dict[str, Any], response: str, direction: str) -> str:
    """Compile the private test program for execution only inside a grading sandbox."""
    answer = crux_answer(response, direction)
    # Like CRUXEval's harness (evaluation/utils_general.py), fail an output answer containing f(<input>) and an
    # input answer without "f(" without running them.
    if (direction == "output" and f"f({problem['input']})" in answer) or (direction == "input" and "f(" not in answer):
        return "raise AssertionError"
    # The legacy output task compares literals; it does not recompute f(input).
    return f"{problem['code']}\nassert {problem['output']} == {answer}"


def execution_program(program: str, timeout: int) -> str:
    """Return the grader command's script: the test program in a forked child, with an alarm after ``timeout`` seconds.

    The child has one more second before it is killed, and the reward goes to reward.json beside the script.
    The receipt and exception handler reject early exits, including os._exit(0).
    """
    return (
        "import json\n"
        "import multiprocessing\n"
        "import signal\n"
        "from pathlib import Path\n"
        "import numpy\n"
        "\n"
        "def timeout_handler(signum, frame):\n"
        "    raise TimeoutError('Candidate timed out')\n"
        "\n"
        "def execute(connection):\n"
        "    signal.signal(signal.SIGALRM, timeout_handler)\n"
        f"    signal.setitimer(signal.ITIMER_REAL, {timeout})\n"
        "    try:\n"
        f"        exec(compile({program!r}, '<candidate>', 'exec'), {{}})\n"
        "    except BaseException:\n"
        "        passed = False\n"
        "    else:\n"
        "        passed = True\n"
        "    finally:\n"
        "        signal.setitimer(signal.ITIMER_REAL, 0)\n"
        "    connection.send(passed)\n"
        "\n"
        "context = multiprocessing.get_context('fork')\n"
        "receiver, sender = context.Pipe(duplex=False)\n"
        "process = context.Process(target=execute, args=(sender,))\n"
        "process.start()\n"
        "sender.close()\n"
        f"process.join({timeout + 1})\n"
        "if process.is_alive():\n"
        "    process.kill()\n"
        "    process.join()\n"
        "passed = False\n"
        "if receiver.poll():\n"
        "    try:\n"
        "        passed = receiver.recv() is True and process.exitcode == 0\n"
        "    except EOFError:\n"
        "        pass\n"
        "Path(__file__).with_name('reward.json').write_text(json.dumps({'reward': float(passed)}))\n"
    )


def run_candidate_program(program: str) -> bool:
    """Run the forked-child grader script and read back its reward receipt."""
    with tempfile.TemporaryDirectory(prefix="micro210-crux-") as directory:
        script = Path(directory) / "candidate_grader.py"
        script.write_text(execution_program(program, CRUX_TIMEOUT_SECONDS))
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=directory,
            capture_output=True,
            timeout=CRUX_TIMEOUT_SECONDS + 5,
            check=False,
        )
        receipt = script.with_name("reward.json")
        if completed.returncode != 0 or not receipt.exists():
            raise ValueError(f"CRUXEval candidate grader failed: {completed.stderr.decode()[-2000:]}")
        reward = float(json.loads(receipt.read_text())["reward"])
        if reward not in (0.0, 1.0):
            raise ValueError(f"CRUXEval candidate grader wrote a non-binary reward: {reward}")
        return reward == 1.0


def crux(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    """Grade one CRUXEval response with the canonical extraction and forked-child execution."""
    answer_text = str(responses[-1])
    direction = spec.grader.reference["direction"]
    program = test_program(dict(spec.doc), answer_text, direction)
    passed = run_candidate_program(program)
    return Grade({"accuracy": float(passed)}, filtered_resps=[crux_answer(answer_text, direction)])
