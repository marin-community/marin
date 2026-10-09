# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove math sources, graded by verifyit's math mode in the grader sandbox.

Known archive scorer/runner revisions identify the supported typed references. Their code is
kept as source evidence; it is never run on model text. Verifyit extracts the last boxed answer
or nonempty line and compares it symbolically. This does not preserve every source parsing rule.
The solver returns its answer in the reply. Archived oracle scripts still supply golden controls.
"""

import hashlib
import json
from dataclasses import dataclass, field, replace

from taskcompendium.convert.answers import source_defect, unsupported
from taskcompendium.convert.delivery import replace_phrases, rewritten_task
from taskcompendium.convert.tasktrove import ANSWER_PATH, SOLVE_SH, archive_files
from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    ResourceGroups,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.controls import answer_reply
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
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
from verifyit.spec import MathSpec, MathType

from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.datasets.tasktrove.archives import TASKTROVE_RELEASE, tasktrove_source
from experiments.post_training.task_curation.pipeline import RlDataPipeline, ShellSim
from experiments.post_training.task_curation.source import DataSourceMetadata, RlDataSource

TASKTROVE_METADATA = replace(
    TASKTROVE_RELEASE,
    dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
    verifier_revision=None,
    family="math-answer",
    verification="math",
    snapshot_safe=True,
    snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
    upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
    modes=("math",),
)

SCORER_RUNNERS = {
    "be1931919ee22ef704f565126353e7edec7b864dbd4a36590ab34593dd2004c7": (
        "7a92019aeea76076ad02e4bbca717db3c3c9396f068126e08beab34e81c3fa66"
    ),
    "703ea4d9abf2eb797c4e23ac6ff26c2f37699a62d9659d5af1475af7e8762f26": (
        "cc69c5b5b676f27249084dd101edfa2ca4dbf96c8d370bebb4f43922bde8943d"
    ),
}
"""SHA-256 of each known ``tests/verifier.py`` mapped to the SHA-256 of the ``tests/test.sh`` that runs it."""
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

The typed reference and archived scorer are hidden from the solver. Verifyit's math mode grades the last boxed
answer or nonempty line in a sandbox; the archived scorer is evidence only. Assess symbolic equivalence and
ordered sequence answers under this comparator; distinguish content quality from grading readiness."""

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

Assess the typed key under verifyit's symbolic math comparator. Scalar, equation, interval, set, and ordered
sequence distinctions matter. Check answer extraction and unit handling under verifyit.
"""


@dataclass(frozen=True)
class MathConverter:
    """Convert a math archive, cutting ``sections`` from the prompt and applying ``phrases`` in order."""

    sections: tuple[str, ...]
    phrases: tuple[tuple[str, str], ...]

    def __call__(self, row: RawRow, context: ConversionContext) -> TaskSpec | NormalizedTask | ImportRejection:
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
            math_type = MathType(data.get("answer_type"))
        except ValueError as error:
            return unsupported("unsupported_answer_contract", str(error))
        prompt = instruction
        for section in self.sections:
            prompt = prompt.partition(section)[0]
        prompt = replace_phrases(prompt, self.phrases).strip()
        if not prompt:
            return source_defect("missing_instruction", "The instruction has no problem before its delivery section")
        # Both source sequence types compare members in order. Math-verify can interpret a
        # parenthesized tuple as an interval, so use verifyit's ordered-member comparator.
        if math_type is MathType.TUPLE:
            math_type = MathType.LIST
        package = verifyit_package(
            MathSpec(expected=expected, math_type=math_type),
            resources=(
                inline_resource("source/verifier.py", scorer),
                inline_resource("verifier_data.json", files["tests/verifier_data.json"]),
                inline_resource("source/test.sh", runner),
            ),
            environment=required_grader_environment(context),
        )
        task = TaskSpec(
            id=row.id,
            source=row.source,
            context=ConversationInput(events=(TextMessage(role="user", content=prompt),)),
            environment_requirements=EnvironmentRequirements(),
            resources=ResourceGroups(
                verifier=package.resources,
                oracle=tuple(
                    inline_resource(path, content) for path, content in files.items() if path.startswith("solution/")
                ),
            ),
            answer_type=AnswerType.TEXT,
            answer_format=PlainText(),
            grader=package.grader,
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


MATH_CONTROLS = Controls(golden=math_golden)


@dataclass(frozen=True)
class MathSource:
    name: str
    config: str
    convert: MathConverter
    rubric: str
    metadata: DataSourceMetadata = field(kw_only=True)


SOURCES = (
    MathSource(
        "tasktrove-math_gym",
        "laion__nemotron-gym-math-v5",
        convert_tasktrove_math,
        GYM_RUBRIC,
        metadata=replace(
            TASKTROVE_METADATA,
            id="Task Trove:laion__nemotron-gym-math-v5",
            name="laion__nemotron-gym-math-v5",
            display_name="laion/nemotron-gym-math-v5",
            task_count=3891,
            notes="Strict trailing boxed compare. No oracle to validate against, so run a no-op gate at conversion.",
            canonical_source="laion/nemotron-gym-math-v5",
            upstream_repository="laion/nemotron-gym-math-v5",
            upstream_url="https://huggingface.co/datasets/laion/nemotron-gym-math-v5",
            input_count=4096,
        ),
    ),
    MathSource(
        "tasktrove-math_openreasoning",
        "laion__nemotron-gym-math-openmathreasoning-v2",
        convert_openreasoning,
        OPENREASONING_RUBRIC,
        metadata=replace(
            TASKTROVE_METADATA,
            id="Task Trove:laion__nemotron-gym-math-openmathreasoning-v2",
            name="laion__nemotron-gym-math-openmathreasoning-v2",
            display_name="laion/nemotron-gym-math-openmathreasoning-v2",
            task_count=42506,
            notes="Most rigorous math verifier in the corpus (scalar/interval/set/tuple/equation), oracle present.",
            canonical_source="laion/nemotron-gym-math-openmathreasoning-v2",
            upstream_repository="laion/nemotron-gym-math-openmathreasoning-v2",
            upstream_url="https://huggingface.co/datasets/laion/nemotron-gym-math-openmathreasoning-v2",
            input_count=42636,
        ),
    ),
    MathSource(
        "tasktrove-math_oracle",
        "SankalpKJ__nemotron-math-oracle-filtered-v2",
        convert_tasktrove_math,
        ORACLE_RUBRIC,
        metadata=replace(
            TASKTROVE_METADATA,
            id="Task Trove:SankalpKJ__nemotron-math-oracle-filtered-v2",
            name="SankalpKJ__nemotron-math-oracle-filtered-v2",
            display_name="SankalpKJ/nemotron-math-oracle-filtered-v2",
            task_count=57383,
            notes="Scalar sympy compare, hidden gold. Overlaps the other Nemotron math sources; subsample.",
            canonical_source="SankalpKJ/nemotron-math-oracle-filtered-v2",
            upstream_repository="SankalpKJ/nemotron-math-oracle-filtered-v2",
            upstream_url="https://huggingface.co/datasets/SankalpKJ/nemotron-math-oracle-filtered-v2",
            input_count=57777,
        ),
    ),
    MathSource(
        "tasktrove-math_prism",
        "laion__nemo-prism-math-v3",
        convert_tasktrove_math,
        PRISM_RUBRIC,
        metadata=replace(
            TASKTROVE_METADATA,
            id="Task Trove:laion__nemo-prism-math-v3",
            name="laion__nemo-prism-math-v3",
            display_name="laion/nemo-prism-math-v3",
            task_count=2219,
            notes="Symbolic exact compare, hidden gold. Add a numeric tolerance path at conversion.",
            canonical_source="laion/nemo-prism-math-v3",
            upstream_repository="laion/nemo-prism-math-v3",
            upstream_url="https://huggingface.co/datasets/laion/nemo-prism-math-v3",
            input_count=2404,
        ),
    ),
    MathSource(
        "tasktrove-math_stack",
        "laion__nemotron-gym-math-stack-overflow-v3",
        convert_tasktrove_math,
        STACK_RUBRIC,
        metadata=replace(
            TASKTROVE_METADATA,
            id="Task Trove:laion__nemotron-gym-math-stack-overflow-v3",
            name="laion__nemotron-gym-math-stack-overflow-v3",
            display_name="laion/nemotron-gym-math-stack-overflow-v3",
            task_count=110266,
            notes="Typed sympy comparison with oracle per task. Remove the non-boxed fallback extraction.",
            canonical_source="laion/nemotron-gym-math-stack-overflow-v3",
            upstream_repository="laion/nemotron-gym-math-stack-overflow-v3",
            upstream_url="https://huggingface.co/datasets/laion/nemotron-gym-math-stack-overflow-v3",
            input_count=110730,
        ),
    ),
)


def sources() -> list[RlDataSource]:
    return [
        RlDataSource(
            metadata=source.metadata,
            pipeline=RlDataPipeline(
                name=source.name,
                source=tasktrove_source(source.config),
                convert=source.convert,
                version="1",
                environment=ShellSim(),
                intended_use=IntendedUse.TRAIN,
                rubric=source.rubric,
                controls=MATH_CONTROLS,
                grader=GRADER_PACKAGES,
            ),
        )
        for source in SOURCES
    ]
