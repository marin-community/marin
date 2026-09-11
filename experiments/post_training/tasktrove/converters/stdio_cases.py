# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Case files for the ``stdio`` mode: ``tests/<cases>/input_<n>.txt`` and ``output_<n>.txt`` pairs."""


from tasktrove_verify.modes.extract import collapse_whitespace

from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus, Rejected
from experiments.post_training.tasktrove.taskbinary import TaskFiles

SOLUTION_COMMAND = "python3 /app/solution.py"
"""How every stdio converter runs the agent's program, once per case."""
CASES_DIR = "tests/cases"


def case_files(inputs: list[str], outputs: list[str]) -> dict[str, bytes]:
    """Case files from parallel input and output lists, numbered from 0."""
    if len(inputs) != len(outputs):
        raise ValueError(f"{len(inputs)} inputs but {len(outputs)} outputs")
    files: dict[str, bytes] = {}
    for index, (stdin, stdout) in enumerate(zip(inputs, outputs, strict=True)):
        files[f"{CASES_DIR}/input_{index}.txt"] = stdin.encode()
        files[f"{CASES_DIR}/output_{index}.txt"] = stdout.encode()
    return files


def case_files_from_dirs(task: TaskFiles) -> dict[str, bytes]:
    """Case files from a template that ships ``inputs/input_<n>.txt`` and ``outputs/output_<n>.txt``."""
    files: dict[str, bytes] = {}
    for path, data in task.under("tests/inputs/").items():
        name = path.rsplit("/", 1)[-1]
        if not (name.startswith("input_") and name.endswith(".txt")):
            continue
        number = name[len("input_") : -len(".txt")]
        expected = task.files.get(f"tests/outputs/output_{number}.txt")
        if expected is None:
            raise ValueError(f"no expected output for case {number}")
        files[f"{CASES_DIR}/input_{number}.txt"] = data
        files[f"{CASES_DIR}/output_{number}.txt"] = expected
    return files


def hidden_case_rejection(files: dict[str, bytes], instruction: str) -> Rejected | None:
    """Reject a case set that cannot grade: none at all, or every input already printed in the prompt.

    A task whose only hidden inputs are the samples in the problem statement is solved by printing
    the sample outputs. Case count alone is not the signal: in codeforces most one-case tasks hold
    an input the prompt never shows, while in TACO and code-contests they are almost all samples.
    """
    inputs = [data.decode(errors="replace") for path, data in files.items() if path.startswith(CASES_DIR + "/input_")]
    if not inputs:
        return Rejected(ConvertStatus.NULL_GRADER, "no stdio cases")
    prompt = collapse_whitespace(instruction)
    if all(collapse_whitespace(stdin) in prompt for stdin in inputs):
        return Rejected(ConvertStatus.GOLD_IN_INSTRUCTION, f"all {len(inputs)} hidden inputs are samples in the prompt")
    return None
