# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Exp4 curriculum: per-source shares over holdout-excluding stamped pools."""

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.execution.lazy import artifact_identity
from marin.rl.curriculum import ValidationSplit, draw_validation_split

from experiments.post_training.task_curation.datasets.long_loop_rl import (
    AGENTIC_ROUTE,
    AGENTIC_TRAIN_SOURCE,
    NEMOTRON_BLENDS_ROWS,
    NONAGENTIC_ROUTE,
    REASONING_GYM_LABEL,
    REASONING_GYM_ROWS,
    STAMPED_POOL_SCHEMA,
    TASKTROVE_ROWS,
    extract_source_pool,
    extract_validation_pool,
    stamped_pool_rows,
)
from experiments.post_training.task_curation.exp4_curriculum import (
    CARRY_OVER_SHARES,
    CURRICULUM_NAME,
    CURRICULUM_VERSION,
    NEMOTRON_IF_LABEL,
    REASONING_GYM_LABEL,
    TARGET_SHARES,
    VALIDATION_SIZE,
    exp4_pool_steps,
    exp4_rl_curriculum,
    exp4_validation_split,
)

VERSION = "dev"

NEMOTRON_BLENDS_LABEL = "nvidia/Nemotron-RL-Ultra-Training-Blends"
SNOWBALL_SMARTIE_LABEL = "open-athena/Snowball-Smartie-RL-2026.10.15.32-Data"
EURUS_LABEL = "PRIME-RL/Eurus-2-RL-Data"
OPEN_SCIENCE_LABEL = "nvidia/OpenScience"
TASKTROVE_LABEL = "open-thoughts/TaskTrove"


def _stamped_parquet(directory: Path, name: str, labels: dict[str, int]) -> tuple[str, str]:
    """Write one stamped pool parquet: ``labels`` maps teacher_route to a row count."""
    rows = [
        {"task_spec": json.dumps({"source": {"dataset": label}, "id": f"{label}-{index}"}), "teacher_route": label}
        for label, count in labels.items()
        for index in range(count)
    ]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    pq.write_table(pa.Table.from_pylist(rows, schema=STAMPED_POOL_SCHEMA), path)
    return str(directory), hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_pools(tmp_path: Path) -> dict[str, tuple[str, str]]:
    """Two local stamped pools shaped like the regional ones: one agentic label, seven nonagentic."""
    agentic = _stamped_parquet(tmp_path / "agentic", "train.parquet", {AGENTIC_TRAIN_SOURCE: 80})
    nonagentic = _stamped_parquet(
        tmp_path / "nonagentic",
        "train.parquet",
        {
            NEMOTRON_IF_LABEL: 40,
            TASKTROVE_LABEL: 30,
            REASONING_GYM_LABEL: 30,
            OPEN_SCIENCE_LABEL: 30,
            NEMOTRON_BLENDS_LABEL: 20,
            SNOWBALL_SMARTIE_LABEL: 20,
            EURUS_LABEL: 20,
        },
    )
    return {AGENTIC_ROUTE: agentic, NONAGENTIC_ROUTE: nonagentic}


def test_the_target_shares_sum_to_one_and_carry_the_dot55_provenance():
    assert math.isclose(sum(TARGET_SHARES.values()), 1.0, abs_tol=1e-9)
    assert TARGET_SHARES[AGENTIC_TRAIN_SOURCE] == 0.35
    assert TARGET_SHARES[NEMOTRON_IF_LABEL] == 0.15
    assert TARGET_SHARES[TASKTROVE_LABEL] == pytest.approx(0.5 * TASKTROVE_ROWS / 100_000)
    assert TARGET_SHARES[REASONING_GYM_LABEL] == pytest.approx(0.5 * REASONING_GYM_ROWS / 100_000)
    assert CARRY_OVER_SHARES[NEMOTRON_BLENDS_LABEL] == pytest.approx(0.5 * NEMOTRON_BLENDS_ROWS / 100_000)
    assert len(TARGET_SHARES) == 8


def test_the_holdout_follows_the_source_shares(tmp_path):
    """The 200-prompt holdout apportions over the eight shares and partitions the pools."""
    pools = _fixture_pools(tmp_path)
    pool_rows = stamped_pool_rows(pools)
    split = exp4_validation_split(pool_rows)
    assert split.validation.size == VALIDATION_SIZE == 200
    assert split.validation.seed == 20261020
    counts = Counter(row.label for row in split.validation.rows)
    assert counts == {
        AGENTIC_TRAIN_SOURCE: 70,
        NEMOTRON_IF_LABEL: 30,
        TASKTROVE_LABEL: 24,
        REASONING_GYM_LABEL: 21,
        OPEN_SCIENCE_LABEL: 21,
        NEMOTRON_BLENDS_LABEL: 12,
        SNOWBALL_SMARTIE_LABEL: 12,
        EURUS_LABEL: 10,
    }
    held = {(row.label, row.id) for row in split.validation.rows}
    trained = {(label, row.id) for label, rows in split.training.items() for row in rows}
    assert held.isdisjoint(trained)
    assert split == draw_validation_split(dict(TARGET_SHARES), pool_rows, size=VALIDATION_SIZE, seed=20261020)


def test_the_entries_pin_the_holdout_excluding_pools(tmp_path):
    pools = _fixture_pools(tmp_path)
    split = exp4_validation_split(stamped_pool_rows(pools))
    curriculum = exp4_rl_curriculum(VERSION, split)
    assert curriculum.name == CURRICULUM_NAME
    assert curriculum.version == CURRICULUM_VERSION
    assert curriculum.validation == split.validation
    pool_steps = exp4_pool_steps(VERSION, split)
    assert {entry.label for entry in curriculum.entries} == set(pool_steps) == set(TARGET_SHARES)
    for entry in curriculum.entries:
        assert entry.identity == artifact_identity(pool_steps[entry.label])
        assert entry.weight == TARGET_SHARES[entry.label]


def test_extraction_writes_one_label_minus_the_holdout(tmp_path):
    """The per-source pool keeps its stamped schema, drops holdout ids, and count-checks."""
    pools = _fixture_pools(tmp_path)
    split = exp4_validation_split(stamped_pool_rows(pools))
    label = AGENTIC_TRAIN_SOURCE
    excluded = tuple(sorted(row.id for row in split.validation.rows if row.label == label))
    output = tmp_path / "pool-out"
    extract_source_pool(
        output=str(output),
        pool={"uri": pools[AGENTIC_ROUTE][0], "sha256": pools[AGENTIC_ROUTE][1]},
        label=label,
        excluded=excluded,
        expected_rows=len(split.training[label]),
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest == {
        "label": label,
        "rows": len(split.training[label]),
        "excluded_holdout_rows": len(excluded),
    }
    kept = pq.read_table(output / "train.parquet").to_pylist()
    assert {row["teacher_route"] for row in kept} == {label}
    assert {json.loads(row["task_spec"])["id"] for row in kept}.isdisjoint(set(excluded))
    assert len(kept) == len(split.training[label])
    with pytest.raises(ValueError, match="after exclusion"):
        extract_source_pool(
            output=str(output),
            pool={"uri": pools[AGENTIC_ROUTE][0], "sha256": pools[AGENTIC_ROUTE][1]},
            label=label,
            excluded=(),
            expected_rows=len(split.training[label]),
        )


def test_validation_extraction_verifies_the_recorded_rows(tmp_path):
    """The holdout parquet holds exactly the recorded rows, and a hash mismatch is refused."""
    pools = _fixture_pools(tmp_path)
    split: ValidationSplit = exp4_validation_split(stamped_pool_rows(pools))
    rows = [{"label": row.label, "id": row.id, "sha256": row.sha256} for row in split.validation.rows]
    output = tmp_path / "val-out"
    extract_validation_pool(
        output=str(output),
        pools=[{"uri": uri, "sha256": sha} for uri, sha in pools.values()],
        rows=rows,
        seed=20261020,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["validation"]["size"] == 200
    assert manifest["composition"] == dict(sorted(Counter(row["label"] for row in rows).items()))
    held = pq.read_table(output / "validation.parquet").to_pylist()
    assert {(row["teacher_route"], json.loads(row["task_spec"])["id"]) for row in held} == {
        (row["label"], row["id"]) for row in rows
    }
    tampered = [dict(row, sha256="0" * 64) for row in rows[:1]] + rows[1:]
    with pytest.raises(ValueError, match="differs from its recorded sha256"):
        extract_validation_pool(
            output=str(output),
            pools=[{"uri": uri, "sha256": sha} for uri, sha in pools.values()],
            rows=tampered,
            seed=20261020,
        )
    fabricated = rows + [{"label": AGENTIC_TRAIN_SOURCE, "id": "ghost", "sha256": "1" * 64}]
    with pytest.raises(ValueError, match="absent from the pools"):
        extract_validation_pool(
            output=str(output),
            pools=[{"uri": uri, "sha256": sha} for uri, sha in pools.values()],
            rows=fabricated,
            seed=20261020,
        )


def test_stamped_pool_rows_verify_the_pinned_digest(tmp_path):
    pools: Mapping[str, tuple[str, str]] = _fixture_pools(tmp_path)
    rows = stamped_pool_rows(pools)
    assert set(rows) == {
        AGENTIC_TRAIN_SOURCE,
        NEMOTRON_IF_LABEL,
        TASKTROVE_LABEL,
        REASONING_GYM_LABEL,
        OPEN_SCIENCE_LABEL,
        NEMOTRON_BLENDS_LABEL,
        SNOWBALL_SMARTIE_LABEL,
        EURUS_LABEL,
    }
    tampered = dict(pools, **{AGENTIC_ROUTE: (pools[AGENTIC_ROUTE][0], "0" * 64)})
    with pytest.raises(ValueError, match="does not match the pinned"):
        stamped_pool_rows(tampered)
