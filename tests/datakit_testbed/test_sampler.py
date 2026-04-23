# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the post-normalize by-provenance sampler."""

import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from marin.datakit.normalize import NormalizedData

from experiments.datakit_testbed.sampler import (
    proportional_sample_fractions,
    sample_normalized_shards,
)
from marin.datakit.sources import DatakitSource


def _write_normalized_shard(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"id": ids, "text": [f"t{i}" for i in ids], "source_id": ids})
    pq.write_table(table, path)


def _make_normalized(tmp_path: Path, shards: dict[str, list[str]]) -> NormalizedData:
    main_dir = tmp_path / "norm" / "outputs" / "main"
    dup_dir = tmp_path / "norm" / "outputs" / "dups"
    dup_dir.mkdir(parents=True, exist_ok=True)
    for rel, ids in shards.items():
        _write_normalized_shard(main_dir / rel, ids)
    return NormalizedData(main_output_dir=str(main_dir), dup_output_dir=str(dup_dir), counters={})


def test_sample_takes_first_k_deterministically(tmp_path: Path):
    source = _make_normalized(
        tmp_path,
        {f"part-{i:04d}.parquet": [f"s{i}"] for i in range(10)},
    )

    out = sample_normalized_shards(
        source=source,
        output_path=str(tmp_path / "sampled"),
        sample_fraction=0.3,  # ceil(10 * 0.3) = 3
    )

    assert isinstance(out, NormalizedData)
    assert out.counters["sampler/total_shards"] == 10
    assert out.counters["sampler/selected_shards"] == 3

    sampled_shards = sorted(Path(out.main_output_dir).glob("*.parquet"))
    assert [p.name for p in sampled_shards] == [
        "part-0000.parquet",
        "part-0001.parquet",
        "part-0002.parquet",
    ]


def test_sample_preserves_dup_output_dir(tmp_path: Path):
    source = _make_normalized(tmp_path, {"part-0000.parquet": ["a"]})
    out = sample_normalized_shards(source=source, output_path=str(tmp_path / "s"), sample_fraction=1.0)
    assert out.dup_output_dir == source.dup_output_dir


def test_sample_is_reproducible_across_invocations(tmp_path: Path):
    source = _make_normalized(tmp_path, {f"part-{i:04d}.parquet": [f"s{i}"] for i in range(20)})
    a = sample_normalized_shards(source=source, output_path=str(tmp_path / "a"), sample_fraction=0.5)
    b = sample_normalized_shards(source=source, output_path=str(tmp_path / "b"), sample_fraction=0.5)
    assert a.counters["sampler/selected_shards"] == b.counters["sampler/selected_shards"] == 10
    assert sorted(p.name for p in Path(a.main_output_dir).glob("*.parquet")) == sorted(
        p.name for p in Path(b.main_output_dir).glob("*.parquet")
    )


def test_sample_fraction_1_copies_everything(tmp_path: Path):
    source = _make_normalized(tmp_path, {f"part-{i:04d}.parquet": [f"s{i}"] for i in range(5)})
    out = sample_normalized_shards(source=source, output_path=str(tmp_path / "s"), sample_fraction=1.0)
    assert out.counters["sampler/selected_shards"] == 5


def test_sample_tiny_fraction_rounds_up_to_one(tmp_path: Path):
    source = _make_normalized(tmp_path, {f"part-{i:04d}.parquet": [f"s{i}"] for i in range(3)})
    out = sample_normalized_shards(source=source, output_path=str(tmp_path / "s"), sample_fraction=0.0001)
    assert out.counters["sampler/selected_shards"] == 1


def test_sample_preserves_content(tmp_path: Path):
    """Selected shards should be byte-equal to the source; sampler never mutates rows."""
    source = _make_normalized(
        tmp_path,
        {"part-0000.parquet": ["a0", "a1", "a2"], "part-0001.parquet": ["b0"]},
    )
    out = sample_normalized_shards(source=source, output_path=str(tmp_path / "s"), sample_fraction=1.0)

    for rel in ["part-0000.parquet", "part-0001.parquet"]:
        src_rows = pq.read_table(Path(source.main_output_dir) / rel).to_pylist()
        dst_rows = pq.read_table(Path(out.main_output_dir) / rel).to_pylist()
        assert src_rows == dst_rows


def test_sample_invalid_fraction_raises(tmp_path: Path):
    source = _make_normalized(tmp_path, {"part-0000.parquet": ["a"]})
    for bad in (0.0, -0.1, 1.1):
        with pytest.raises(ValueError, match="sample_fraction"):
            sample_normalized_shards(source=source, output_path=str(tmp_path / "out"), sample_fraction=bad)


def test_sample_no_shards_raises(tmp_path: Path):
    empty = tmp_path / "norm" / "outputs" / "main"
    empty.mkdir(parents=True)
    source = NormalizedData(
        main_output_dir=str(empty),
        dup_output_dir=str(tmp_path / "d"),
        counters={},
    )
    with pytest.raises(ValueError, match="No parquet shards"):
        sample_normalized_shards(source=source, output_path=str(tmp_path / "s"), sample_fraction=0.5)


def _src(name: str, rough: float | None) -> DatakitSource:
    return DatakitSource(name=name, normalize_steps=(), rough_token_count_b=rough)


def test_proportional_fractions_sum_to_target_across_known_sources():
    sources = [_src("big", 3800.0), _src("small", 200.0)]
    fractions = proportional_sample_fractions(sources, target_total_tokens_b=1000.0)

    # big: 1000 * 3800 / 4000 = 950B target -> 950/3800 = 0.25
    # small: 1000 * 200 / 4000 = 50B target -> 50/200 = 0.25
    assert math.isclose(fractions["big"], 0.25, rel_tol=1e-9)
    assert math.isclose(fractions["small"], 0.25, rel_tol=1e-9)


def test_proportional_fractions_clamp_to_one_when_target_exceeds_source():
    sources = [_src("big", 3800.0), _src("tiny", 10.0)]
    fractions = proportional_sample_fractions(sources, target_total_tokens_b=10_000.0)
    assert fractions["tiny"] == 1.0


def test_proportional_fractions_unknown_sources_take_all(caplog):
    sources = [_src("known", 100.0), _src("mystery", None)]
    with caplog.at_level("WARNING"):
        fractions = proportional_sample_fractions(sources, target_total_tokens_b=50.0)
    assert fractions["mystery"] == 1.0
    assert any("rough_token_count_b" in r.message for r in caplog.records)


def test_proportional_fractions_all_unknown_defaults_to_take_all(caplog):
    sources = [_src("a", None), _src("b", None)]
    with caplog.at_level("WARNING"):
        fractions = proportional_sample_fractions(sources, target_total_tokens_b=1.0)
    assert fractions == {"a": 1.0, "b": 1.0}
    assert any("no source has rough_token_count_b" in r.message for r in caplog.records)
