# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream the pinned NUPA sample or resolve a frozen NUPA5K panel without source evaluators."""

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from random import Random
from typing import Any

from rigging.filesystem.storage_path import StoragePath

from taskcompendium.convert.evaluation import source_evidence
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation import ChatGeneration, EvaluationItem
from taskcompendium.importers.evalchemy.grader import source_package
from taskcompendium.importers.evalchemy.scorers.nupa_preparation import NupaAnswerFormat
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.evaluation import source_order
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import RawRow

NUPA_SAMPLE_SEED = 20_222_943
NUPA_SAMPLES_PER_STRATUM = 100
NUPA_PANEL_FILENAME = "nupa5k_manifest.jsonl"


@dataclass(frozen=True)
class NupaIdentity:
    task_name: str
    digit: int
    sha256: str


def source_text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def length_bucket(digit: int, *, max_digit: int) -> str:
    boundaries = (4, 8, 14) if max_digit <= 20 else (10, 20, 60)
    return next(
        (bucket for boundary, bucket in zip(boundaries, ("S", "M", "L"), strict=True) if digit <= boundary), "XL"
    )


def source_tasks(path: StoragePath) -> Iterator[tuple[str, dict[str, list[str]]]]:
    # ijson is a curation-only optional dependency; evaluator/grader environments do not need it.
    import ijson  # noqa: PLC0415

    with path.open("rb") as stream:
        yield from ijson.kvitems(stream, "")


def _record(task: str, digit: int, index: int, text: str, *, max_digit: int, source_index: int) -> dict[str, Any]:
    prompt, answer = text.split("=", 1)
    parts = task.split("_")
    operation = (
        "_".join(parts[:2])
        if len(parts) >= 2 and (parts[0] in {"digit", "get", "to"} or parts[1] in {"easy", "hard"})
        else parts[0]
    )
    return {
        "id": f"test:{task}:{digit}:{index:06d}",
        "task_name": task,
        "operation": operation,
        "answer_format": "Integer" if parts[-1] == "int" else parts[-1],
        "digit": digit,
        "length_bucket": length_bucket(digit, max_digit=max_digit),
        "prompt": prompt.rstrip() + " =",
        "answer": answer.strip(),
        "source_sha256": source_text_sha256(text),
        "source_index": source_index,
    }


def _max_digit(groups: Mapping[str, list[str]]) -> int:
    maximum = max(map(int, groups))
    # The released to_float task contains a 21-digit group in the short-range suite.
    return 20 if maximum == 21 else maximum


def read_nupa_sample(
    path: StoragePath,
    _context: ConversionContext,
    *,
    num_each: int = NUPA_SAMPLES_PER_STRATUM,
    seed: int = NUPA_SAMPLE_SEED,
) -> Iterator[dict[str, Any]]:
    """Preserve source task/group order, global RNG state, and duplicate sampled occurrences."""
    random = Random(seed)
    for task, groups in source_tasks(path):
        maximum = _max_digit(groups)
        for digit, texts in groups.items():
            indices = random.sample(range(len(texts)), min(num_each, len(texts)))
            for ordinal, index in enumerate(indices):
                yield {
                    **_record(task, int(digit), ordinal, texts[index], max_digit=maximum, source_index=index),
                    "sample_seed": seed,
                    "samples_per_stratum": num_each,
                }


def read_nupa_panel(path: StoragePath, context: ConversionContext) -> Iterator[dict[str, Any]]:
    """Resolve the supplied immutable manifest in its order, selecting each text identity once."""
    with (context.inputs["panel"] / NUPA_PANEL_FILENAME).open("r") as stream:
        identities = tuple(NupaIdentity(**json.loads(line)) for line in stream if line.strip())
    if len(set(identities)) != len(identities):
        raise ValueError("NUPA panel manifest repeats a source identity")
    targets = set(identities)
    tasks = {identity.task_name for identity in identities}
    records: dict[NupaIdentity, dict[str, Any]] = {}
    for task, groups in source_tasks(path):
        if task not in tasks:
            continue
        maximum = _max_digit(groups)
        for digit, texts in groups.items():
            for index, text in enumerate(texts):
                identity = NupaIdentity(task, int(digit), source_text_sha256(text))
                if identity not in targets or identity in records:
                    continue
                record = _record(task, int(digit), index, text, max_digit=maximum, source_index=index)
                record["id"] = f"test:{task}:{digit}:{identity.sha256}"
                records[identity] = record
    missing = targets - records.keys()
    if missing:
        raise ValueError(f"Pinned NUPA source is missing {len(missing)} panel identities")
    yield from (records[identity] for identity in identities)


def convert_nupa(row: RawRow, context: ConversionContext, *, benchmark: str) -> TaskSpec:
    task = conversation_task(
        row,
        events=(TextMessage(role="user", content=row.data["prompt"]),),
        package=source_package(
            "nupa",
            {"expected": row.data["answer"], "answer_format": NupaAnswerFormat(row.data["answer_format"])},
            environment=required_grader_environment(context),
        ),
        evidence=row.data,
    )
    return task.model_copy(
        update={"tags": ("eval", "source:evalchemy", "family:numeric-representation", f"benchmark:{benchmark}")}
    )


def nupa_item(task: TaskSpec) -> EvaluationItem:
    source = source_evidence(task)
    return EvaluationItem(
        task_id=task.id,
        source_row=task.source.row,
        source_ordinal=source_order(task.source.row)[1],
        operation=ChatGeneration(),
        groups={
            "source_id": source["id"],
            "task": source["task_name"],
            "bucket": source["length_bucket"],
            "task_bucket": source["task_name"] + "/bucket:" + source["length_bucket"],
        },
    )
