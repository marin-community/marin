# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert Big-Math's verified closed-form problems with a reference judge against the answer."""

from dataclasses import replace

from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.convert.answers import source_defect
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.grader import verifyit_package
from taskcompendium.models import TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, NormalizationChange, NormalizedTask, RawRow
from taskcompendium.pipeline.reviewer_spec import JudgeTier

from experiments.post_training.task_curation.judges import JUDGE_TIERS
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

REPO = "SynthLabsAI/Big-Math-RL-Verified"
REVISION = "c75d2f117cddfecb6bd08756e61e508e59732b21"
SOURCE_ID = "SynthLabs:Big-Math-RL-Verified"
VERSION = "big-math-reference-judge-v1"
COUNT = 251122
NATIVE_FIELDS = frozenset({"problem", "answer", "source", "domain", "llama8b_solve_rate"})
CARD_URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/README.md"
REFERENCE_JUDGE_QUESTION = (
    "Determine whether the candidate's final answer is correct for the posed mathematical problem, "
    "judged against the reference answer. A response that states the same answer in different notation "
    "or phrasing is correct; an answer that disagrees with, contradicts or fails to address the "
    "reference is incorrect."
)


def convert_problem(row: RawRow, _context: ConversionContext) -> NormalizedTask | ImportRejection:
    native = dict(row.data)
    if (
        set(native) != NATIVE_FIELDS
        or not isinstance(native["problem"], str)
        or not native["problem"].strip()
        or not isinstance(native["answer"], str)
        or not native["answer"].strip()
        or not isinstance(native["source"], str)
        or not isinstance(native["domain"], list)
        or not all(isinstance(item, str) for item in native["domain"])
        or not isinstance(native["llama8b_solve_rate"], float)
    ):
        return source_defect(
            "incomplete_big_math_contract", "Expected the five original fields with a nonempty problem and answer"
        )
    judge = replace(
        JUDGE_TIERS[JudgeTier.MEDIUM].spec,
        question=REFERENCE_JUDGE_QUESTION,
        references=(native["answer"].strip(),),
        temperature=0.0,
        max_completion_tokens=64,
    )
    package = verifyit_package(judge)
    events = (TextMessage(role="user", content=native["problem"]),)
    task = conversation_task(row, events=events, package=package).model_copy(
        update={
            "tags": ("train", "math", "judge", "reference", "big-math"),
            "labels": {
                "source_id": SOURCE_ID,
                "native_source": native["source"],
                "domain": ",".join(native["domain"]),
            },
        }
    )
    change = NormalizationChange(
        field="context.events",
        reason=(
            f"Carry only the problem into the context; the dataset's llama8b_solve_rate difficulty column is "
            f"retained as task metadata and never shown to the policy. Source card: {CARD_URL}"
        ),
        original="The native row carries the problem alongside its verified answer and difficulty metadata.",
        replacement=(
            "The task context carries the verbatim problem as the sole user message; the answer stays bound "
            "as the reference judge's gold and the difficulty metadata stays in task labels."
        ),
    )
    return NormalizedTask(task, (change,))


def source(files: tuple[str, ...]) -> RlDataSource:
    return RlDataSource(
        info=SourceInfo(
            origin=SourceOrigin.SYNTHLABS,
            upstream_dataset="Big-Math-RL-Verified",
            upstream_name="Big-Math-RL-Verified",
            build_path=VERSION,
            family="math",
            data_restriction=DataRestriction.OPEN,
            task_format=TaskFormat.TASKSPEC,
            count=COUNT,
            notes=(
                "New MEDIUM reference judge: each task's context is the verbatim verified problem, and the "
                "dataset's closed-form answer is the judge's reference gold (S1 closed-ended precedent). The "
                "llama8b_solve_rate difficulty column stays in task labels, never in the policy context. The "
                "card documents 250,000+ rigorously filtered and verified problems; the sealed inventory measures "
                "251,122 rows. No native executable grader is claimed; no live judge execution is claimed."
            ),
        ),
        config=CurationRecipe(
            name="big-math-reference-judge",
            version=VERSION,
            source=HfSource(REPO, REVISION, files, SourceFormat.PARQUET),
            convert=convert_problem,
            intended_use=IntendedUse.TRAIN,
        ),
        pipeline=process_rows,
    )
