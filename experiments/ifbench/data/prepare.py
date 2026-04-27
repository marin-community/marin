# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage-1 prompt assembly for the IFBench rollout pipeline.

Loads `allenai/IF_multi_constraints_upto5` (95,373 single-turn rows), carves a
stratified val hold-out, and emits two jsonl files. Includes a contamination
check against `allenai/IFBench_test` (300 OOD eval prompts, never trained on).

See `.agents/logbooks/dpo_sft.md` § D-013 for the split design.

This module is pure-Python data prep. The Marin StepSpec wrapper that
content-addresses the output lives in a sibling module.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import logging
import pathlib
import random
from collections.abc import Iterable, Iterator
from typing import Any

logger = logging.getLogger(__name__)

# These are the only datasets this module touches. The OOD test set is
# named explicitly so that a `grep -r "IFBench_test"` shows every load site.
TRAIN_HF_DATASET = "allenai/IF_multi_constraints_upto5"
TEST_OOD_HF_DATASET = "allenai/IFBench_test"  # NEVER load outside the eval harness.

# Expected row count of the train HF dataset. If this drifts, the upstream
# version moved and we should re-pin and re-validate the verifier port.
EXPECTED_TRAIN_ROWS = 95_373

# Default val carve size. Stratified by `constraint_type`.
DEFAULT_VAL_SIZE = 2_000


@dataclasses.dataclass(frozen=True)
class PreparedRow:
    """Schema for the jsonl artefacts emitted by this stage."""

    prompt_id: str
    messages: list[dict[str, str]]
    constraint: str
    constraint_type: str
    ground_truth: str  # Python repr-style; parse with verifiers.parse.parse_ground_truth
    num_constraints: int  # Stratification key — see _count_constraints docstring.


def _stable_prompt_id(messages: list[dict[str, str]]) -> str:
    """Deterministic id based on message contents - survives reshuffling."""
    payload = json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha1(payload).hexdigest()[:16]


def _count_constraints(ground_truth: str) -> int:
    """Count constraints in a row's ground_truth.

    Inspecting the real dataset: every row has `constraint_type == "multi"`, so
    that field is useless for stratification. The actual difficulty axis is the
    number of constraints per prompt (1-5 per the AI2 paper). We parse the
    ground_truth to extract that count.
    """
    from experiments.ifbench.verifiers.parse import parse_ground_truth

    return len(parse_ground_truth(ground_truth).instruction_id_list)


def _normalise_row(row: dict[str, Any]) -> PreparedRow:
    """Convert a raw HF row into our PreparedRow schema."""
    messages = list(row["messages"])
    ground_truth = str(row["ground_truth"])
    return PreparedRow(
        prompt_id=_stable_prompt_id(messages),
        messages=messages,
        constraint=str(row["constraint"]),
        constraint_type=str(row["constraint_type"]),
        ground_truth=ground_truth,
        num_constraints=_count_constraints(ground_truth),
    )


def stratified_val_split(
    rows: list[PreparedRow],
    val_size: int = DEFAULT_VAL_SIZE,
    seed: int = 0,
) -> tuple[list[PreparedRow], list[PreparedRow]]:
    """Split rows into (train, val), proportional by `num_constraints`.

    Stratification key is the per-row constraint count (1-5). The original
    `constraint_type` field is uninformative - every row in the published
    IF_multi_constraints_upto5 dataset has `constraint_type == "multi"`.
    """
    rng = random.Random(seed)
    by_type: dict[int, list[PreparedRow]] = collections.defaultdict(list)
    for row in rows:
        by_type[row.num_constraints].append(row)

    total = len(rows)
    if val_size <= 0 or val_size >= total:
        raise ValueError(f"val_size must be in (0, {total}); got {val_size}")

    # Proportional allocation, rounded down. Distribute the rounding remainder
    # to the buckets in descending size order until we hit val_size.
    counts = {ct: len(rs) for ct, rs in by_type.items()}
    raw_alloc = {ct: (counts[ct] * val_size) // total for ct in counts}
    remaining = val_size - sum(raw_alloc.values())
    for ct in sorted(counts, key=lambda c: -counts[c]):
        if remaining <= 0:
            break
        if raw_alloc[ct] < counts[ct]:
            raw_alloc[ct] += 1
            remaining -= 1
    if remaining != 0:  # bucket starvation — should be impossible at val_size << total
        raise RuntimeError(f"could not allocate {val_size} rows across {len(counts)} types")

    val: list[PreparedRow] = []
    train: list[PreparedRow] = []
    for ct, bucket in by_type.items():
        rng.shuffle(bucket)
        take = raw_alloc[ct]
        val.extend(bucket[:take])
        train.extend(bucket[take:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def load_train_rows(hf_dataset: str = TRAIN_HF_DATASET) -> Iterator[PreparedRow]:
    """Stream rows from HF, applying the schema normalisation. Lazy."""
    from datasets import load_dataset  # local import — heavy module

    ds = load_dataset(hf_dataset, split="train", streaming=True)
    for raw in ds:
        yield _normalise_row(raw)


def load_test_prompts(hf_dataset: str = TEST_OOD_HF_DATASET) -> list[str]:
    """Load IFBench_test prompts. **Eval-harness use only — see D-013.**

    Imported here purely so the contamination check can verify zero overlap.
    No part of the rollout pipeline should call this function.
    """
    from datasets import load_dataset

    ds = load_dataset(hf_dataset, split="train")  # HF stores it as 'train' split despite name
    return [row["prompt"] for row in ds]


def assert_no_test_contamination(rows: Iterable[PreparedRow]) -> None:
    """Raise if any prepared train/val prompt matches an IFBench_test prompt.

    Compares exact strings AND rstrip-normalised strings (the IFBench paper's
    own fixtures have trailing-whitespace mismatches between train and test
    splits, so byte-equality alone would underreport real contamination).
    """
    test_prompts = load_test_prompts()
    test_set = set(test_prompts) | {p.rstrip() for p in test_prompts}

    overlaps: list[str] = []
    for row in rows:
        # The user prompt is the last user message — our normalised messages
        # are single-turn, so this is just messages[0]["content"].
        for msg in row.messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if content in test_set or content.rstrip() in test_set:
                    overlaps.append(row.prompt_id)
                break

    if overlaps:
        raise RuntimeError(
            f"Contamination detected: {len(overlaps)} train/val prompts match "
            f"IFBench_test (first 5: {overlaps[:5]}). This violates the paper's "
            "by-construction guarantee — investigate before proceeding."
        )


def write_jsonl(rows: list[PreparedRow], path: pathlib.Path) -> None:
    """Write rows to jsonl, one PreparedRow per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(dataclasses.asdict(row), ensure_ascii=False) + "\n")


def prepare_train_val(
    output_dir: pathlib.Path,
    val_size: int = DEFAULT_VAL_SIZE,
    seed: int = 0,
    skip_contamination_check: bool = False,
) -> dict[str, Any]:
    """Top-level entrypoint: load + split + check contamination + write jsonl.

    Returns a metadata dict suitable for inclusion in pipeline-step output.
    """
    logger.info("Loading %s (expected %d rows)…", TRAIN_HF_DATASET, EXPECTED_TRAIN_ROWS)
    rows = list(load_train_rows())
    if len(rows) != EXPECTED_TRAIN_ROWS:
        logger.warning(
            "Train row count %d != expected %d; upstream version may have changed",
            len(rows),
            EXPECTED_TRAIN_ROWS,
        )

    train, val = stratified_val_split(rows, val_size=val_size, seed=seed)
    logger.info("Split: %d train + %d val (val %.2f%%)", len(train), len(val), 100 * len(val) / (len(train) + len(val)))

    if not skip_contamination_check:
        logger.info("Checking IFBench_test contamination…")
        assert_no_test_contamination(train + val)
        logger.info("OK: zero overlap with %d IFBench_test prompts.", len(load_test_prompts()))

    write_jsonl(train, output_dir / "train.jsonl")
    write_jsonl(val, output_dir / "val.jsonl")

    # Stratum summary by num_constraints, so a reviewer can sanity-check.
    val_strata = collections.Counter(r.num_constraints for r in val)
    train_strata = collections.Counter(r.num_constraints for r in train)

    return {
        "train_rows": len(train),
        "val_rows": len(val),
        "seed": seed,
        "train_hf_dataset": TRAIN_HF_DATASET,
        "expected_train_rows": EXPECTED_TRAIN_ROWS,
        "val_strata_by_num_constraints": dict(val_strata),
        "train_strata_by_num_constraints": dict(train_strata),
        "contamination_checked": not skip_contamination_check,
    }
