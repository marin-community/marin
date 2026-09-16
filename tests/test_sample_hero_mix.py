# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.datakit.sample_hero_mix import (
    align_source_shards,
    cell_name,
    expected_sampled_tokens,
    keep_probabilities,
    sample_shard,
)


def _write_mix(tmp_path):
    mix = {
        "available_tokens": {"c00q0": 1000, "c00q1": 10, "c01q0": 5000},
        "phases": [
            {"name": "initial", "weights": {"c00q0": 0.5, "c00q1": 0.0, "c01q0": 0.5}},
            {"name": "main", "weights": {"c00q0": 0.2, "c00q1": 0.3, "c01q0": 0.5}},
            {"name": "cooldown", "weights": {"c00q0": 0.1, "c00q1": 0.1, "c01q0": 0.8}},
        ],
    }
    p = tmp_path / "mix.json"
    p.write_text(json.dumps(mix))
    return str(p)


def test_cell_name():
    assert cell_name(27, 0) == "c27q0"
    assert cell_name(7, 4) == "c07q4"


def test_keep_probabilities_and_capping(tmp_path):
    path = _write_mix(tmp_path)
    target = 10_000
    p = keep_probabilities(path, target, phase="main")
    # c00q0: 10000*0.2/1000 = 2.0 -> capped at 1.0
    assert p["c00q0"] == 1.0
    # c00q1: 10000*0.3/10 = 300 -> capped at 1.0 (tiny cell)
    assert p["c00q1"] == 1.0
    # c01q0: 10000*0.5/5000 = 1.0 -> exactly 1.0
    assert p["c01q0"] == 1.0


def test_keep_probabilities_subunity(tmp_path):
    path = _write_mix(tmp_path)
    target = 1_000  # small target -> all p < 1
    p = keep_probabilities(path, target, phase="main")
    assert abs(p["c00q0"] - 0.2) < 1e-9  # 1000*0.2/1000
    assert abs(p["c01q0"] - 0.1) < 1e-9  # 1000*0.5/5000
    assert p["c00q1"] == 1.0  # 1000*0.3/10 = 30 -> capped
    # expected sampled tokens: 0.2*1000 + 1.0*10 + 0.1*5000 = 200 + 10 + 500 = 710
    assert abs(expected_sampled_tokens(p, path) - 710.0) < 1e-6


def test_zero_weight_cell_kept_never(tmp_path):
    path = _write_mix(tmp_path)
    p = keep_probabilities(path, 1_000, phase="initial")  # c00q1 has weight 0 in initial
    assert p["c00q1"] == 0.0


def _write_shard(tmp_path, rows, dups=(), exact=()):

    dirs = {k: tmp_path / k for k in ("normalized", "decontam", "cluster", "quality", "exact_dedup", "dedup")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    ids = [r["id"] for r in rows]
    pq.write_table(pa.table({"id": ids, "text": [r["text"] for r in rows]}), dirs["normalized"] / "s.parquet")
    pq.write_table(
        pa.table({"id": ids, "contaminated": [r["contaminated"] for r in rows]}), dirs["decontam"] / "s.parquet"
    )
    pq.write_table(pa.table({"id": ids, "cluster40": [r["cluster"] for r in rows]}), dirs["cluster"] / "s.parquet")
    pq.write_table(pa.table({"id": ids, "quality_bucket": [r["quality"] for r in rows]}), dirs["quality"] / "s.parquet")
    pq.write_table(pa.table({"id": list(exact)}), dirs["exact_dedup"] / "s.parquet")
    pq.write_table(pa.table({"id": list(dups), "dup_doc": [True] * len(dups)}), dirs["dedup"] / "s.parquet")
    return {k: str(v) for k, v in dirs.items()}


def test_sample_shard_survival_and_keep(tmp_path):

    rows = [
        {"id": "a", "text": "keep me", "contaminated": False, "cluster": 0, "quality": 0},  # cell c00q0 p=1
        {"id": "b", "text": "contam", "contaminated": True, "cluster": 0, "quality": 0},  # dropped: contaminated
        {"id": "c", "text": "dupd", "contaminated": False, "cluster": 0, "quality": 0},  # dropped: verified dup
        {"id": "d", "text": "exactdup", "contaminated": False, "cluster": 0, "quality": 0},  # dropped: exact dup
        {"id": "e", "text": "never", "contaminated": False, "cluster": 1, "quality": 0},  # cell c01q0 p=0
    ]
    dirs = _write_shard(tmp_path, rows, dups=["c"], exact=["d"])
    specs = align_source_shards("src", dirs)
    assert len(specs) == 1
    keep = {"c00q0": 1.0, "c01q0": 0.0}
    got = list(sample_shard(specs[0], "cluster40", keep, seed=7))
    assert got == [("a", "keep me")]  # b/c/d filtered, e has p=0


def test_sample_shard_id_mismatch_raises(tmp_path):

    rows = [{"id": "a", "text": "x", "contaminated": False, "cluster": 0, "quality": 0}]
    dirs = _write_shard(tmp_path, rows)
    # corrupt the normalized shard's id so it no longer aligns with the attrs

    pq.write_table(pa.table({"id": ["WRONG"], "text": ["x"]}), tmp_path / "normalized" / "s.parquet")
    specs = align_source_shards("src", dirs)
    with pytest.raises(RuntimeError, match="id mismatch"):
        list(sample_shard(specs[0], "cluster40", {"c00q0": 1.0}, seed=1))
