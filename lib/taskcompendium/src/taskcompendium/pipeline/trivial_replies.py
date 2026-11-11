# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade replies that carry no answer on published evaluation corpora, with each task's own grader.

Evaluation publication checks conversion integrity and grades no replies. Curation controls, which do not run on
evaluation corpora, grade one submission per sampled task: the source's golden or negative one, else an empty reply.
A grader can still credit a short nonempty reply: one word satisfies instruction constraints that only restrict the
reply, and ``0`` matches numeric answers of zero. This module grades a fixed set of replies that ignore the task on
every task of a corpus through ``grade_task``, which calls the in-process and machine graders that evaluation rollouts
call, and reduces the credit per corpus.

Run ``python -m taskcompendium.pipeline.trivial_replies CORPUS... --output DIR --grading-machines local``.
"""

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from shellbox.machine import MachineFactory, MachineSpec
from verifyit.reasoning_markup import COMPLETED_STOP_REASON

from taskcompendium.evaluation.corpus import EvaluationCorpus
from taskcompendium.evaluation.spec import OperationKind
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import (
    CONVERSATION_ANSWERS,
    ConversationTrace,
    GradingAttempt,
    NoGrader,
    ScriptGrader,
    TaskSpec,
    TextMessage,
    grades_with_judge,
)
from taskcompendium.pipeline.controls import GradingMachines
from taskcompendium.pipeline.models import Controls
from taskcompendium.pipeline.verification import answer_event
from taskcompendium.runtime.environment import validate_machine_spec
from taskcompendium.runtime.local import LocalGraderMachines
from taskcompendium.runtime.task_grading import grade_task

MEMORY_MB = Controls.memory_mb
"""Memory requested for each grading machine, the curation controls' default; local machines do not bound it."""
GENERATION_OPERATIONS = frozenset({OperationKind.CHAT_GENERATION, OperationKind.TEXT_GENERATION})
BATCH_TASKS = 16
"""Tasks held in memory at a time; long-context tasks can each carry megabytes of prompt and grader evidence."""
GRADES_FILENAME = "grades.jsonl"
SKIPS_FILENAME = "skips.jsonl"
FLOORS_FILENAME = "floors.json"


class TrivialReply(StrEnum):
    """A reply that ignores the task."""

    EMPTY = "empty"
    WORD = "word"
    ZERO = "zero"
    ABSTAIN = "abstain"
    ECHO = "echo"
    """The task's last user message, repeated."""


CONSTANT_REPLIES = {
    TrivialReply.EMPTY: "",
    TrivialReply.WORD: "hello",
    TrivialReply.ZERO: "0",
    TrivialReply.ABSTAIN: "I don't know.",
}


class SkipReason(StrEnum):
    """Why a task's replies were not graded."""

    LIKELIHOOD = "likelihood"
    """The corpus scores the likelihood of fixed continuations instead of generated replies."""
    NO_GRADER = "no_grader"
    JUDGE = "judge"
    """The grader calls a judge service, which this audit does not contact."""
    WORKSPACE_ANSWER = "workspace_answer"
    """The task is answered with files or state."""
    NO_USER_MESSAGE = "no_user_message"
    TASK_MACHINE = "task_machine"
    """The grader collects inputs from the task's own machine."""
    NO_MACHINES = "no_machines"
    """The grader needs a grading machine and none was provided."""
    UNSUPPORTED_MACHINE = "unsupported_machine"
    """The grading machines cannot run the grader's environment."""


class MachineChoice(StrEnum):
    """Where tasks whose graders need a machine are graded."""

    LOCAL = "local"
    """Bubblewrap machines built from each grader's packages, as local curation runs use."""
    NONE = "none"
    """Skip those tasks."""


@dataclass(frozen=True)
class GradingPlan:
    """A task to grade, with its grading machine, or ``None`` for an in-process grader."""

    task: TaskSpec
    machine: tuple[MachineFactory, MachineSpec] | None


@dataclass(frozen=True)
class TrivialGrade:
    """One reply's grade on one task. ``score`` is the grader's reward, or ``None`` when it gave no verdict."""

    dataset: str
    task_id: str
    source_row: str
    reply: TrivialReply
    status: Outcome
    score: float | None
    error: str | None


@dataclass(frozen=True)
class TaskSkip:
    dataset: str
    task_id: str
    source_row: str
    reason: SkipReason


@dataclass(frozen=True)
class ReplyFloor:
    """One reply's credit over a corpus's graded tasks.

    ``mean_score`` averages the grades with a verdict and ``credited`` counts those with a positive score;
    ``no_verdict`` counts grades that ended without one, such as infrastructure errors.
    """

    reply: TrivialReply
    mean_score: float | None
    credited: int
    no_verdict: int


@dataclass(frozen=True)
class CorpusFloor:
    """The credit per reply on one corpus, identified by its root and its manifest digest."""

    benchmark: str
    corpus: str
    dataset: str
    graded_tasks: int
    skipped: dict[SkipReason, int]
    replies: tuple[ReplyFloor, ...]


def user_messages(task: TaskSpec) -> list[str]:
    return [event.content for event in task.context.events if isinstance(event, TextMessage) and event.role == "user"]


def reply_text(task: TaskSpec, reply: TrivialReply) -> str:
    """The text of ``reply`` for ``task``; ``ECHO`` repeats its last user message."""
    if reply is TrivialReply.ECHO:
        return user_messages(task)[-1]
    return CONSTANT_REPLIES[reply]


def grading_plan(task: TaskSpec, machines: GradingMachines | None) -> GradingPlan | SkipReason:
    """How to grade ``task``'s replies here, or why they cannot be graded."""
    grader = task.grader
    if isinstance(grader, NoGrader):
        return SkipReason.NO_GRADER
    if grades_with_judge(grader):
        return SkipReason.JUDGE
    if task.answer_type not in CONVERSATION_ANSWERS:
        return SkipReason.WORKSPACE_ANSWER
    if not user_messages(task):
        return SkipReason.NO_USER_MESSAGE
    if grader.environment is None:
        return GradingPlan(task, None)
    if isinstance(grader, ScriptGrader) and (grader.collect or grader.artifacts):
        return SkipReason.TASK_MACHINE
    if machines is None:
        return SkipReason.NO_MACHINES
    try:
        factory, spec = machines.machine(grader.environment, MEMORY_MB)
        validate_machine_spec(grader.environment, factory, spec)
    except ValueError:
        return SkipReason.UNSUPPORTED_MACHINE
    return GradingPlan(task, (factory, spec))


def grade_reply(plan: GradingPlan, text: str) -> GradeResult:
    """Grade ``text``, in the task's answer format, as the final reply of a turn that stopped normally."""
    task = plan.task
    attempt = GradingAttempt(
        ConversationTrace(events=(*task.context.events, answer_event(task, text))), stop_reason=COMPLETED_STOP_REASON
    )
    if plan.machine is None:
        return grade_task(task, attempt)
    factory, spec = plan.machine
    return grade_task(task, attempt, machine_factory=factory, machine_spec=spec)


def reply_score(grade: GradeResult) -> float | None:
    """The grader's reward; a rejected submission scores 0, and a grade without a verdict has none."""
    if grade.status not in (Outcome.GRADED, Outcome.SUBMISSION_FAILURE):
        return None
    return grade.reward


def dataset_digest(corpus: EvaluationCorpus) -> str:
    """The corpus snapshot's identity, as evaluation runs record it."""
    return canonical_sha256(corpus.manifest.model_dump(mode="json"))


def grade_corpus(
    corpus: EvaluationCorpus,
    replies: Sequence[TrivialReply],
    machines: GradingMachines | None,
    workers: int,
    batch_tasks: int,
) -> Iterator[tuple[list[TrivialGrade], list[TaskSkip]]]:
    """Grade every reply on every task of ``corpus`` that can be graded here, ``workers`` grades at a time.

    Holds at most ``batch_tasks`` tasks to grade at a time and yields, in task order, their grades and the tasks
    skipped since the previous batch, with why. With one worker, grading runs in the calling thread.
    """
    dataset = dataset_digest(corpus)
    if corpus.protocol.operation not in GENERATION_OPERATIONS:
        yield [], [TaskSkip(dataset, row.task_id, row.source_row, SkipReason.LIKELIHOOD) for row in corpus.rows()]
        return

    def grade(pair: tuple[GradingPlan, TrivialReply]) -> GradeResult:
        plan, reply = pair
        return grade_reply(plan, reply_text(plan.task, reply))

    def graded(plans: list[GradingPlan]) -> list[TrivialGrade]:
        pairs = [(plan, reply) for plan in plans for reply in replies]
        if workers == 1:
            results = [grade(pair) for pair in pairs]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(grade, pairs))
        return [
            TrivialGrade(
                dataset, plan.task.id, plan.task.source.row, reply, result.status, reply_score(result), result.error
            )
            for (plan, reply), result in zip(pairs, results, strict=True)
        ]

    plans: list[GradingPlan] = []
    skips: list[TaskSkip] = []
    for case in corpus.cases():
        plan = grading_plan(case.task, machines)
        if isinstance(plan, SkipReason):
            skips.append(TaskSkip(dataset, case.task.id, case.task.source.row, plan))
        else:
            plans.append(plan)
        if len(plans) == batch_tasks:
            yield graded(plans), skips
            plans, skips = [], []
    if plans or skips:
        yield graded(plans), skips


def corpus_floor(
    corpus: EvaluationCorpus, root: str, grades: Sequence[TrivialGrade], skips: Sequence[TaskSkip]
) -> CorpusFloor:
    """Reduce one corpus's grades and skips to the credit per reply."""
    by_reply: dict[TrivialReply, list[TrivialGrade]] = defaultdict(list)
    for grade in grades:
        by_reply[grade.reply].append(grade)
    replies = []
    for reply, reply_grades in by_reply.items():
        scores = [grade.score for grade in reply_grades if grade.score is not None]
        replies.append(
            ReplyFloor(
                reply,
                sum(scores) / len(scores) if scores else None,
                sum(score > 0 for score in scores),
                len(reply_grades) - len(scores),
            )
        )
    return CorpusFloor(
        corpus.protocol.benchmark,
        root,
        dataset_digest(corpus),
        len({grade.task_id for grade in grades}),
        dict(Counter(skip.reason for skip in skips)),
        tuple(replies),
    )


def floor_table(floors: Sequence[CorpusFloor], replies: Sequence[TrivialReply]) -> str:
    """A Markdown table of each reply's mean score and, in parentheses, the tasks it earned credit on."""
    header = ["Benchmark", "Graded tasks", "Skipped tasks", *replies]
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for floor in floors:
        cells = {item.reply: item for item in floor.replies}
        row = [floor.benchmark, str(floor.graded_tasks), str(sum(floor.skipped.values()))]
        for reply in replies:
            item = cells.get(reply)
            row.append("-" if item is None or item.mean_score is None else f"{item.mean_score:.3f} ({item.credited})")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Grade replies that carry no answer on published evaluation corpora.")
    parser.add_argument("corpora", nargs="+", help="Roots of published evaluation corpora")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help=f"Directory for {GRADES_FILENAME}, {SKIPS_FILENAME} and {FLOORS_FILENAME}",
    )
    parser.add_argument(
        "--grading-machines",
        required=True,
        choices=[choice.value for choice in MachineChoice],
        help="Where to grade tasks whose graders need a machine",
    )
    parser.add_argument("--workers", type=int, default=1, help="Grades to run at a time")
    args = parser.parse_args(argv)
    machines = LocalGraderMachines() if MachineChoice(args.grading_machines) is MachineChoice.LOCAL else None
    corpora = [(root, EvaluationCorpus(root)) for root in args.corpora]
    if len({dataset_digest(corpus) for _, corpus in corpora}) != len(corpora):
        raise ValueError("An audit cannot include the same corpus snapshot twice")
    replies = tuple(TrivialReply)
    args.output.mkdir(parents=True, exist_ok=True)
    floors: list[CorpusFloor] = []
    with (
        (args.output / GRADES_FILENAME).open("w", encoding="utf-8") as grades_file,
        (args.output / SKIPS_FILENAME).open("w", encoding="utf-8") as skips_file,
    ):
        for root, corpus in corpora:
            grades: list[TrivialGrade] = []
            skips: list[TaskSkip] = []
            for batch_grades, batch_skips in grade_corpus(corpus, replies, machines, args.workers, BATCH_TASKS):
                grades_file.writelines(json.dumps(asdict(grade)) + "\n" for grade in batch_grades)
                skips_file.writelines(json.dumps(asdict(skip)) + "\n" for skip in batch_skips)
                grades_file.flush()
                skips_file.flush()
                grades.extend(batch_grades)
                skips.extend(batch_skips)
            floors.append(corpus_floor(corpus, root, grades, skips))
            (args.output / FLOORS_FILENAME).write_text(
                json.dumps([asdict(floor) for floor in floors], indent=1) + "\n", encoding="utf-8"
            )
    print(floor_table(floors, replies))


if __name__ == "__main__":
    main()
