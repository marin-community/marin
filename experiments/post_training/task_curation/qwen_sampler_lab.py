# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the fixed arithmetic, string and list pool for the Qwen sampler experiment."""

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from itertools import accumulate
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType

import click
from finestore.rl.strata import MembershipState, SelectionDefinition, TaskStrata, read_selection_manifest
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from shellbox.machine import CommandSemantics, PackageSpec
from taskcompendium.models import (
    AnswerType,
    Boxed,
    ConversationInput,
    EnvironmentRequirements,
    GraderMachine,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)
from zephyr.writers import write_parquet_file

from experiments.post_training.task_curation import CANONICAL_ARTIFACT_ROOT, publish_once
from experiments.post_training.task_curation.selection import (
    FULL_VALIDATION_FILENAME,
    SUMMARY_FILENAME,
    TASK_MANIFEST_FILENAME,
    TASK_MANIFEST_SCHEMA,
    TASK_SCHEMA,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
    _selection_row,
    task_record,
)

NAME = "qwen-sampler-lab"
VERSION = "2026.11.25.1"
ROOT = str(StoragePath(CANONICAL_ARTIFACT_ROOT) / "rl-data/task-selection" / NAME / VERSION)
TRAIN_SEED, EVAL_SEED = 20261125, 20261126
TRAIN_PER_CELL, FULL_DEV_PER_CELL, FAST_DEV_PER_CELL = 512, 16, 4
_PLACEMENT = GraderMachine(
    environment=EnvironmentRequirements(command_semantics=CommandSemantics.SHELL_SIMULATOR, package_spec=PackageSpec())
)
# The model's answer is always file data, never shell source. Normalize outer whitespace and comma spacing.
_CHECK = """answer=$(sed 's/^[[:space:]]*//;s/[[:space:]]*$//;s/[[:space:]]*,[[:space:]]*/,/g' /app/answer.txt) || exit 2
if [ "$answer" = "$1" ]; then echo 1; else echo 0; fi"""


class Difficulty(StrEnum):
    EASY = "easy"
    INTERMEDIATE = "intermediate"
    HARD = "hard"


def arithmetic(rng: random.Random, difficulty: Difficulty) -> tuple[str, str]:
    a, b = rng.randint(11, 98), rng.randint(2, 49)
    if difficulty == Difficulty.EASY:
        return f"What is {a} + {b}?", str(a + b)
    c = rng.randint(3, 37)
    if difficulty == Difficulty.INTERMEDIATE:
        return f"What is ({a} + {b}) × {c}?", str((a + b) * c)  # noqa: RUF001
    d, modulus = rng.randint(100, 999), rng.randint(17, 97)
    prompt = f"Find the nonnegative remainder of (({a} + {b}) × {d} − {c}) divided by {modulus}."  # noqa: RUF001
    return prompt, str(((a + b) * d - c) % modulus)


def string_transform(rng: random.Random, difficulty: Difficulty) -> tuple[str, str]:
    length = {Difficulty.EASY: 5, Difficulty.INTERMEDIATE: 9, Difficulty.HARD: 15}[difficulty]
    word = "".join(rng.choices("abcdefghijkmnpqrstuvwxyz", k=length))
    if difficulty == Difficulty.EASY:
        return f"Reverse the letters of {word}.", word[::-1]
    if difficulty == Difficulty.INTERMEDIATE:
        return f"Take letters at positions 1, 3, 5, ... of {word}, then reverse those letters.", word[::2][::-1]
    offset = rng.randint(2, 7)
    rotated = word[offset:] + word[:offset]
    return (
        f"Rotate {word} left by {offset} letters. Keep positions 1, 3, 5, ... of the rotated word, "
        "then put the retained letters in alphabetical order, preserving duplicates.",
        "".join(sorted(rotated[::2])),
    )


def integer_list(rng: random.Random, difficulty: Difficulty) -> tuple[str, str]:
    length = {Difficulty.EASY: 4, Difficulty.INTERMEDIATE: 7, Difficulty.HARD: 10}[difficulty]
    values = [rng.randint(-25, 50) for _ in range(length)]
    if difficulty == Difficulty.EASY:
        operation, result = "Sort in ascending order", sorted(values)
    elif difficulty == Difficulty.INTERMEDIATE:
        operation, result = "Double each value, then list the results in reverse input order", [
            2 * x for x in values[::-1]
        ]
    else:
        modulus = rng.randint(5, 13)
        operation = (
            f"Compute running sums from left to right and replace each sum by its nonnegative remainder modulo {modulus}"
        )
        result = [n % modulus for n in accumulate(values)]
    return f"{operation} for {values}. Use comma-separated integers without brackets.", ",".join(map(str, result))


FAMILIES = MappingProxyType({"arithmetic": arithmetic, "string": string_transform, "list": integer_list})


@dataclass(frozen=True)
class Example:
    task: TaskSpec
    strata: TaskStrata
    split: str
    fast_dev: bool


def lab_examples() -> list[Example]:
    seen = set()
    rows = []
    for split, seed, size in (("train", TRAIN_SEED, TRAIN_PER_CELL), ("dev", EVAL_SEED, FULL_DEV_PER_CELL)):
        for family, generator in FAMILIES.items():
            for difficulty in Difficulty:
                rng = random.Random(f"{seed}/{family}/{difficulty}")
                strata = TaskStrata(family=family, source=f"{NAME}/{family}", difficulty=difficulty)
                for index in range(size):
                    prompt, answer = generator(rng, difficulty)
                    while prompt in seen:
                        prompt, answer = generator(rng, difficulty)
                    seen.add(prompt)
                    digest = hashlib.sha256(prompt.encode()).hexdigest()
                    source = Source(dataset=strata.source, revision=VERSION, row=digest, importer_revision=VERSION)
                    task = TaskSpec(
                        id=f"{NAME}-{digest}",
                        context=ConversationInput(
                            events=(
                                TextMessage(role="user", content=prompt + r" Put your final answer inside \boxed{...}."),
                            )
                        ),
                        environment_requirements=EnvironmentRequirements(),
                        answer_type=AnswerType.TEXT,
                        answer_format=Boxed(),
                        grader=ScriptGrader(argv=("sh", "-c", _CHECK, "grade", answer), placement=_PLACEMENT, timeout=5),
                        source=source,
                    )
                    rows.append(Example(task, strata, split, split == "dev" and index < FAST_DEV_PER_CELL))
    # Interleave strata so a prefix of the released parquet is independent of generation order.
    random.Random(TRAIN_SEED).shuffle(rows)
    return rows


def selection_rows(examples: list[Example]) -> tuple[dict[str, list[dict]], list[dict]]:
    manifests, training, development, fast = [], [], [], []
    for example in examples:
        raw = example.task.model_dump_json()
        record = task_record(raw, example.strata)
        active = example.split == "train" or example.fast_dev
        manifests.append(
            _selection_row(
                example.task, raw, example.strata, state=MembershipState.ACTIVE if active else MembershipState.CANDIDATE
            ).model_dump(mode="json")
        )
        if example.split == "train":
            training.append(record)
        else:
            development.append(record)
            if example.fast_dev:
                fast.append(record)
    files = {TRAIN_FILENAME: training, FULL_VALIDATION_FILENAME: development, VALIDATION_FILENAME: fast}
    return files, manifests


def publish(output: str) -> str:
    files, manifests = selection_rows(lab_examples())
    training, development, fast = files[TRAIN_FILENAME], files[FULL_VALIDATION_FILENAME], files[VALIDATION_FILENAME]
    definition = SelectionDefinition(
        name=NAME,
        version=VERSION,
        sampling="seeded, equal family × generated difficulty quotas",  # noqa: RUF001
        task_files=(TRAIN_FILENAME, FULL_VALIDATION_FILENAME),
        active_files=(TRAIN_FILENAME, VALIDATION_FILENAME),
    )
    with TemporaryDirectory(prefix="qwen-sampler-") as temporary:
        directory = Path(temporary)
        for name, rows in files.items():
            write_parquet_file(rows, str(directory / name), schema=TASK_SCHEMA)
        write_parquet_file(manifests, str(directory / TASK_MANIFEST_FILENAME), schema=TASK_MANIFEST_SCHEMA)
        contents = {name: (directory / name).read_bytes() for name in (*files, TASK_MANIFEST_FILENAME)}
        summary = {
            "selection_strata": definition.model_dump(mode="json"),
            "train_seed": TRAIN_SEED,
            "dev_seed": EVAL_SEED,
            "training_rows": len(training),
            "validation_rows": len(fast),
            "full_validation_rows": len(development),
            "training_family_counts": dict(Counter(r["family"] for r in training)),
            "validation_family_counts": dict(Counter(r["family"] for r in fast)),
            "grader_kind_counts": {
                name.removesuffix(".parquet"): {"package": len(rows)} for name, rows in files.items()
            },
            "file_sha256": {n: hashlib.sha256(data).hexdigest() for n, data in contents.items()},
        }
        summary_json = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        contents[SUMMARY_FILENAME] = summary_json.encode()
        (directory / SUMMARY_FILENAME).write_bytes(contents[SUMMARY_FILENAME])
        read_selection_manifest(str(directory))
        publish_once(StoragePath(output), contents)
    return summary_json


@click.command(help=__doc__)
@click.option("--output", default=ROOT, show_default=True)
@click.option("--run", is_flag=True, help="Publish or verify identical existing bytes.")
def main(output: str, run: bool) -> None:
    if run:
        configure_coreweave_s3()
        click.echo(publish(output), nl=False)
    else:
        click.echo(output)


if __name__ == "__main__":
    main()
