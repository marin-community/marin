# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a base-seven carry intervention with independent generator-based panels."""

import json
import random
from collections import Counter
from hashlib import sha256

import click
from finestore.rl.strata import SelectionDefinition, TaskStrata, read_selection_manifest
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation import qwen_sampler_lab as lab

NAME = "qwen-topic-control"
VERSION = "2026.11.26.1"
ROOT = str(lab.SELECTION_ROOT / NAME / VERSION)
TRAIN_TOPICS = ("target", "prerequisite", "decoy", "copy")
HELD_OUT_TOPICS = ("near", "transfer", "decoy", "copy")
DEVELOPMENT_PER_TOPIC = 24
CONFIRMATION_PER_TOPIC = 64
TRAIN_PER_TOPIC = 256


def base_seven(number: int) -> str:
    result = ""
    while number:
        number, digit = divmod(number, 7)
        result = str(digit) + result
    return result or "0"


def problem(topic: str, rng: random.Random) -> tuple[str, str]:
    if topic == "prerequisite":
        a, b = rng.randrange(7, 49), rng.randrange(7, 49)
        digits = base_seven(a), base_seven(b)
        return (
            f"Add only the last digits of the base-seven numerals {digits[0]} and {digits[1]}. "
            "Write the sum in base seven, with no leading zero.",
            base_seven(a % 7 + b % 7),
        )
    if topic == "decoy":
        a, b = rng.randrange(13, 67), rng.randrange(3, 12)
        return f"In ordinary decimal arithmetic, multiply {a} by {b}. Return the product.", str(a * b)
    if topic == "copy":
        value = rng.randrange(10000, 99999)
        return f"Copy this five-digit code exactly: {value}.", str(value)
    digits = 3 if topic == "transfer" else 2
    a, b = (rng.randrange(7 ** (digits - 1), 7**digits) for _ in range(2))
    left, right = base_seven(a), base_seven(b)
    if topic in ("near", "transfer"):
        prompt = f"Use base seven for both inputs and the output. What is {left} + {right}? Omit leading zeros."
    elif topic == "target":
        prompt = f"Add the base-seven numerals {left} and {right}. Give their sum in base seven without leading zeros."
    else:
        raise ValueError(f"Unknown topic {topic!r}")
    return prompt, base_seven(a + b)


def examples() -> list[lab.Example]:
    reference = lab._examples(NAME, VERSION, 1, 1, "train", 0, set())[0].task
    seen = set()
    rows = []
    partitions = (
        ("train", 26601, TRAIN_TOPICS, TRAIN_PER_TOPIC),
        ("probe", 26602, TRAIN_TOPICS, DEVELOPMENT_PER_TOPIC),
        ("dev", 26603, HELD_OUT_TOPICS, DEVELOPMENT_PER_TOPIC),
        ("confirmation", 26604, HELD_OUT_TOPICS, CONFIRMATION_PER_TOPIC),
    )
    for split, seed, topics, count in partitions:
        for topic in topics:
            rng = random.Random(f"{seed}/{topic}")
            source = f"{NAME}/{topic}"
            strata = TaskStrata(family=topic, source=source, topic=topic)
            for _ in range(count):
                while True:
                    prompt, answer = problem(topic, rng)
                    key = prompt
                    if topic in ("target", "near", "transfer"):
                        # Hold out the operands regardless of prompt wording or input order.
                        digits = [token.rstrip(".?") for token in prompt.split()]
                        key = "/".join(sorted(token for token in digits if token.isdigit()))
                    identity = sha256(prompt.encode()).hexdigest()
                    if key not in seen:
                        seen.add(key)
                        break
                task_source = reference.source.model_copy(
                    update={"dataset": source, "row": identity, "revision": VERSION, "importer_revision": VERSION}
                )
                message = reference.context.events[0].model_copy(
                    update={"content": prompt + r" Put your final answer inside \boxed{...}."}
                )
                task = reference.model_copy(
                    update={
                        "id": f"{NAME}-{identity}",
                        "source": task_source,
                        "context": reference.context.model_copy(update={"events": (message,)}),
                        "grader": reference.grader.model_copy(update={"argv": (*reference.grader.argv[:-1], answer)}),
                    }
                )
                rows.append(lab.Example(task, strata, split, split != "train"))
    random.Random(26601).shuffle(rows)
    return rows


def publish(base: StoragePath) -> dict[str, dict]:
    rows = examples()
    identities = {row.task.id for row in rows}
    assert len(identities) == len(rows)
    results = {}
    for split in ("train", "probe", "dev", "confirmation"):
        members = [row for row in rows if row.split == split]
        filename = lab.TRAIN_FILENAME if split == "train" else lab.VALIDATION_FILENAME
        name = NAME if split == "train" else f"{NAME}-{split}"
        obj = base / name / VERSION
        files, manifest = lab.selection_rows(members)
        kind = "training" if split == "train" else "validation"
        summary = {
            f"{kind}_family_counts": dict(Counter(row.strata.family for row in members)),
            f"{kind}_rows": len(members),
            "selection_strata": (
                SelectionDefinition(
                    name=name,
                    version=VERSION,
                    sampling="seeded exact-grader topic strata with disjoint instances",
                    task_files=(filename,),
                    active_files=(filename,),
                ).model_dump(mode="json")
            ),
            "grader_kind_counts": {filename.removesuffix(".parquet"): {"package": len(members)}},
        }
        record = json.loads(lab._publish_selection(str(obj), {filename: files[filename]}, manifest, summary))
        assert read_selection_manifest(str(obj)) is not None
        for file, digest in record["file_sha256"].items():
            assert sha256((obj / file).read_bytes()).hexdigest() == digest
        results[split] = {"root": str(obj), **record}
    return results


@click.command()
@click.option("--output", type=str, default=None)
@click.option("--run", is_flag=True)
def main(output: str | None, run: bool) -> None:
    base = StoragePath(output) if output else lab.SELECTION_ROOT
    if run:
        configure_coreweave_s3()
        click.echo(json.dumps(publish(base), indent=2))
    else:
        click.echo(str(base / NAME / VERSION))


if __name__ == "__main__":
    main()
