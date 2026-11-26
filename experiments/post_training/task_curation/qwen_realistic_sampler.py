# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Release small, disjoint slices of admitted Marin tasks for Qwen sampler tests."""

import json
import random
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

import click
import pyarrow.parquet as pq
from finestore.rl.strata import MembershipState, SelectionDefinition, TaskStrata
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import AnswerType, TaskSpec
from taskcompendium.submission import conversation_messages, submission_instruction
from transformers import AutoTokenizer

from experiments.post_training.task_curation import qwen_sampler_lab as lab
from experiments.post_training.task_curation.selection import _grader_kind, _selection_row

NAME = "qwen-realistic-sampler"
VERSION = "2026.11.26.1"
MAX_PROMPT_TOKENS = 6144
TRAIN_PER_FAMILY = 192
DEV_PER_FAMILY = 24
TRANSFER_PER_FAMILY = 24
TRAIN_FAMILIES = ("arithmetic", "calendar", "instruction_calendar", "structural")
TRANSFER_FAMILIES = ("general", "scientific_reasoning", "structural")
SEED = 26630


def _pool(path: Path, tokenizer, families: tuple[str, ...]) -> dict[str, list[tuple[dict, TaskSpec, str]]]:
    pools = defaultdict(list)
    for row in pq.read_table(path).to_pylist():
        family = row["family"]
        if family not in families:
            continue
        task = TaskSpec.model_validate_json(row["task_spec"])
        if task.answer_type is not AnswerType.TEXT or task.interaction_tools or task.final_tools:
            continue
        messages = conversation_messages(task.context.events)
        if len(messages) != 1 or messages[0]["role"] != "user" or not isinstance(messages[0]["content"], str):
            continue
        instruction = submission_instruction(task.answer_format)
        if instruction:
            messages[0]["content"] += "\n" + instruction
        tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        ).input_ids
        if len(tokens) <= MAX_PROMPT_TOKENS:
            pools[family].append((row, task, canonical_sha256(messages)))
    return pools


def _release(rows, output: StoragePath, filename: str, parents: dict[str, str]) -> dict:
    label = output.parent.name
    counts = dict(Counter(row["family"] for row, _, _ in rows))
    manifest = []
    for row, task, _ in rows:
        fields = {name: row[name] for name in TaskStrata.model_fields if row.get(name) is not None}
        manifest.append(
            _selection_row(task, row["task_spec"], TaskStrata(**fields), state=MembershipState.ACTIVE).model_dump(
                mode="json"
            )
        )
    kind = "training" if filename == lab.TRAIN_FILENAME else "validation"
    summary = {
        "selection_strata": (
            SelectionDefinition(
                name=label,
                version=VERSION,
                sampling="seeded non-agentic single-turn slices below the Qwen prompt window",
                task_files=(filename,),
                active_files=(filename,),
            ).model_dump(mode="json")
        ),
        f"{kind}_rows": len(rows),
        f"{kind}_family_counts": counts,
        "grader_kind_counts": {
            filename.removesuffix(".parquet"): dict(Counter(_grader_kind(task) for _, task, _ in rows))
        },
        "parent_sha256": parents,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
    }
    result = json.loads(lab._publish_selection(str(output), {filename: [row for row, _, _ in rows]}, manifest, summary))
    for basename, digest in result["file_sha256"].items():
        assert sha256((output / basename).read_bytes()).hexdigest() == digest
    return {"root": str(output), **result}


@click.command()
@click.option("--source", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--eval-lite", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--tokenizer", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--output", default=None)
def main(source: Path, eval_lite: Path, tokenizer: Path, output: str | None) -> None:
    configure_coreweave_s3()
    codec = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
    candidate = _pool(source, codec, TRAIN_FAMILIES)
    transfer = _pool(eval_lite, codec, TRANSFER_FAMILIES)
    rng = random.Random(SEED)
    train, development, confirmation = [], [], []
    for family in TRAIN_FAMILIES:
        subset = rng.sample(candidate[family], TRAIN_PER_FAMILY + DEV_PER_FAMILY)
        development.extend(subset[:DEV_PER_FAMILY])
        train.extend(subset[DEV_PER_FAMILY:])
    occupied = {task.id for _, task, _ in (*train, *development)}
    seen = {digest for _, _, digest in (*train, *development)}
    assert len(occupied) == len(seen) == len(train) + len(development)
    for family in TRANSFER_FAMILIES:
        eligible = [item for item in transfer[family] if item[1].id not in occupied and item[2] not in seen]
        confirmation.extend(rng.sample(eligible, TRANSFER_PER_FAMILY))
    all_rows = (*train, *development, *confirmation)
    assert len({task.id for _, task, _ in all_rows}) == len({digest for _, _, digest in all_rows}) == len(all_rows)
    parents = {str(path.name): sha256(path.read_bytes()).hexdigest() for path in (source, eval_lite)}
    base = StoragePath(output) if output else lab.SELECTION_ROOT
    results = {}
    for split, members, filename in (
        ("train", train, lab.TRAIN_FILENAME),
        ("dev", development, lab.VALIDATION_FILENAME),
        ("confirmation", confirmation, lab.VALIDATION_FILENAME),
    ):
        rng.shuffle(members)
        label = NAME if split == "train" else f"{NAME}-{split}"
        results[split] = _release(members, base / label / VERSION, filename, parents)
    click.echo(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
