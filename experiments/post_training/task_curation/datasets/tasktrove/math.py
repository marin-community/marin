# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove math sources, graded by each archive's own SymPy scorer in the executable math image.

An archive ships the scorer (``tests/verifier.py``), its runner (``tests/test.sh``) and the typed
reference (``tests/verifier_data.json``). Only scorer and runner revisions seen before are
accepted, and ``math_grade.py`` checks the interpreter and packages the scorer was pinned to before
running it. The solver gets a conversation task: the prompt's answer-file delivery is rewritten to
ask for the answer in the reply, which the grader writes to ``/app/answer.txt`` for the scorer.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from taskcompendium.convert.answers import source_defect, unsupported
from taskcompendium.convert.delivery import replace_phrases, rewritten_task
from taskcompendium.convert.source_scorer import ANSWER_PATH
from taskcompendium.convert.tasktrove import SOLVE_SH, TEST_SH_REWARD, archive_files
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    ResourceGroups,
    ScriptGrader,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.controls import answer_reply, wrong_reply
from taskcompendium.pipeline.models import (
    Controls,
    ControlSubmission,
    ImportRejection,
    IntendedUse,
    NormalizedTask,
    OracleCommand,
    RawRow,
)
from taskcompendium.runtime.resources import inline_resource, resource_bytes
from verifyit.spec import MathType

from experiments.post_training.task_curation.datasets.tasktrove import tasktrove_source
from experiments.post_training.task_curation.images import EXECUTABLE_MATH_IMAGE
from experiments.post_training.task_curation.pipeline import RlDataPipeline, ShellSim

GRADE_SCRIPT = "math_grade.py"
GYM_SCORER = "be1931919ee22ef704f565126353e7edec7b864dbd4a36590ab34593dd2004c7"
SCORER_RUNNERS = {
    GYM_SCORER: "7a92019aeea76076ad02e4bbca717db3c3c9396f068126e08beab34e81c3fa66",
    "703ea4d9abf2eb797c4e23ac6ff26c2f37699a62d9659d5af1475af7e8762f26": (
        "cc69c5b5b676f27249084dd101edfa2ca4dbf96c8d370bebb4f43922bde8943d"
    ),
}
"""SHA-256 of each known ``tests/verifier.py`` mapped to the SHA-256 of the ``tests/test.sh`` that runs it."""
SCORER_PINS = ("python==3.11", "sympy==1.13.3", "antlr4-python3-runtime==4.11.0")
GYM_SCORER_PINS = ("numpy==2.1.3",)
GRADER_TIMEOUT = 600.0
REWRITE_REASON = "Replace the source's answer-file delivery with an answer in the assistant response"

SUBMISSION = "\n## Submitting the answer\n"
TERMINAL_SUBMISSION = "\n## Submitting your answer (IMPORTANT)\n"
FILE_DELIVERY = (
    ("Provide your answer in the file answer.txt", "Return your final answer in the assistant response."),
    (
        "Please place your final answer in a file named `/app/solution.txt`.",
        "Return your final answer in the assistant response.",
    ),
    ("write your final answer at the path `/app/answer.txt`", "return your final answer in the assistant response"),
    ("the answer file", "the assistant response"),
)
ANSWER_DELIVERY = (
    ("write your final answer at the path `/app/answer.txt`", "return your final answer"),
    ("Write your final answer to the path `/app/answer.txt`", "Return your final answer"),
    (
        "write the final numeric answer (a single number) to `/app/answer.txt`",
        "return the final numeric answer (a single number)",
    ),
    ("write ONLY the value of the LAST one to `/app/answer.txt`", "return ONLY the value of the LAST one"),
    ("the answer file", "the assistant response"),
    ("your answer file", "your assistant response"),
)

MATH_CRITERIA = """Require a complete mathematical problem, supplied givens, notation, units, and requested result.

Check hidden reference consistency; difficulty alone is not a defect and a failed control does not prove the problem
is bad.

The source's SymPy scorer and typed reference are hidden from the solver. Assess its exact extraction, typed
comparison and error behavior; distinguish content quality from grading readiness."""

GYM_RUBRIC = f"""
{MATH_CRITERIA}

Check complete contest statements and exact final-answer format; independently verify feasible calculations and flag
hidden references answering a different quantity.
"""

ORACLE_RUBRIC = f"""
{MATH_CRITERIA}

Check that oracle-filtered references solve the public problem; source oracle existence is evidence of grader
compatibility, not proof of mathematical correctness.
"""

PRISM_RUBRIC = f"""
{MATH_CRITERIA}

Check symbolic olympiad statements, quantifiers, strict versus attained extrema, and whether escaped LaTeX keys express
the requested quantity.
"""

STACK_RUBRIC = f"""
{MATH_CRITERIA}

Check mathematical questions for missing prior context, definitions, diagrams, or truncated expressions; a plausible
hidden answer cannot fill absent public premises.
"""

OPENREASONING_RUBRIC = """
Check the full mathematical problem, givens, notation, units, diagrams, and requested result. Reject absent diagrams,
contradictory assumptions, or a hidden key inconsistent with a demonstrated solution.

Independently verify short calculations. For long proofs, assess whether the problem is well posed; difficulty and
inability to solve immediately are not defects. Do not invent a reference conflict.

Assess the source's SymPy scorer and typed key together. Scalar, equation, interval, set, and ordered sequence
distinctions matter, as do answer extraction and the source's unit handling.
"""


def scorer_pins(scorer: bytes) -> tuple[str, ...]:
    """The interpreter and package pins one known scorer revision was validated with."""
    extra = GYM_SCORER_PINS if hashlib.sha256(scorer).hexdigest() == GYM_SCORER else ()
    return (*SCORER_PINS, *extra)


@dataclass(frozen=True)
class MathConverter:
    """Convert a math archive, cutting ``sections`` from the prompt and applying ``phrases`` in order."""

    sections: tuple[str, ...]
    phrases: tuple[tuple[str, str], ...]

    def __call__(self, row: RawRow) -> TaskSpec | NormalizedTask | ImportRejection:
        files = archive_files(row.data).files
        scorer, runner = files.get("tests/verifier.py", b""), files.get("tests/test.sh", b"")
        expected_runner = SCORER_RUNNERS.get(hashlib.sha256(scorer).hexdigest())
        if expected_runner is None or hashlib.sha256(runner).hexdigest() != expected_runner:
            return unsupported("unsupported_math_scorer", "Unrecognized original math scorer/runner")
        if "tests/verifier_data.json" not in files:
            return source_defect("missing_verifier_data", "Original math verifier data is required")
        instruction, data = row.data["instruction"], row.data.get("verifier_data")
        if not instruction.strip() or not isinstance(data, dict):
            return source_defect("missing_instruction", "A mathematical instruction and verifier data are required")
        expected = data.get("expected_answer")
        if not isinstance(expected, str) or not expected.strip():
            return unsupported("unsupported_answer_contract", "A nonempty typed math reference is required")
        try:
            MathType(data.get("answer_type"))
        except ValueError as error:
            return unsupported("unsupported_answer_contract", str(error))
        prompt = instruction
        for section in self.sections:
            prompt = prompt.partition(section)[0]
        prompt = replace_phrases(prompt, self.phrases).strip()
        if not prompt:
            return source_defect("missing_instruction", "The instruction has no problem before its delivery section")
        grader = ScriptGrader(
            argv=("python3", f"/tests/{GRADE_SCRIPT}", *scorer_pins(scorer)),
            cwd="/app",
            environment=EXECUTABLE_MATH_IMAGE.requirements(),
            answer_path=ANSWER_PATH,
            reward=TEST_SH_REWARD,
            timeout=GRADER_TIMEOUT,
        )
        task = TaskSpec(
            id=row.id,
            source=row.source,
            context=ConversationInput(events=(TextMessage(role="user", content=prompt),)),
            environment_requirements=EnvironmentRequirements(),
            resources=ResourceGroups(
                verifier=(
                    inline_resource("verifier.py", scorer),
                    inline_resource("verifier_data.json", files["tests/verifier_data.json"]),
                    inline_resource("test.sh", runner),
                    inline_resource(GRADE_SCRIPT, Path(__file__).with_name(GRADE_SCRIPT).read_bytes()),
                ),
                oracle=tuple(
                    inline_resource(path, content) for path, content in files.items() if path.startswith("solution/")
                ),
            ),
            answer_type=AnswerType.TEXT,
            answer_format=PlainText(),
            grader=grader,
        )
        return rewritten_task(task, original=instruction, reason=REWRITE_REASON)


convert_tasktrove_math = MathConverter(
    sections=(SUBMISSION, TERMINAL_SUBMISSION), phrases=(*FILE_DELIVERY, *ANSWER_DELIVERY)
)
convert_openreasoning = MathConverter(sections=(SUBMISSION,), phrases=ANSWER_DELIVERY)


def math_golden(task: TaskSpec) -> ControlSubmission:
    """The source's ``solution/solve.sh`` when it ships one, else its typed reference in a box."""
    if any(resource.path == SOLVE_SH for resource in task.resources.oracle):
        return OracleCommand(f"bash /{SOLVE_SH}", answer_file=ANSWER_PATH)
    data = next(resource for resource in task.resources.verifier if resource.path == "verifier_data.json")
    expected = json.loads(resource_bytes(data))["expected_answer"]
    return answer_reply(task, rf"\boxed{{{expected}}}")


MATH_CONTROLS = Controls(golden=math_golden, negative=wrong_reply)


@dataclass(frozen=True)
class MathSource:
    name: str
    config: str
    convert: MathConverter
    rubric: str


SOURCES = (
    MathSource("tasktrove-math_gym", "laion__nemotron-gym-math-v5", convert_tasktrove_math, GYM_RUBRIC),
    MathSource(
        "tasktrove-math_openreasoning",
        "laion__nemotron-gym-math-openmathreasoning-v2",
        convert_openreasoning,
        OPENREASONING_RUBRIC,
    ),
    MathSource(
        "tasktrove-math_oracle", "SankalpKJ__nemotron-math-oracle-filtered-v2", convert_tasktrove_math, ORACLE_RUBRIC
    ),
    MathSource("tasktrove-math_prism", "laion__nemo-prism-math-v3", convert_tasktrove_math, PRISM_RUBRIC),
    MathSource(
        "tasktrove-math_stack", "laion__nemotron-gym-math-stack-overflow-v3", convert_tasktrove_math, STACK_RUBRIC
    ),
)


def pipelines() -> list[RlDataPipeline]:
    return [
        RlDataPipeline(
            name=source.name,
            source=tasktrove_source(source.config),
            convert=source.convert,
            version="1",
            environment=ShellSim(),
            intended_use=IntendedUse.TRAIN,
            rubric=source.rubric,
            controls=MATH_CONTROLS,
            atlas_id=f"Task Trove:{source.config}",
        )
        for source in SOURCES
    ]
