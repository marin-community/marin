# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for Zephyr benchmark source selection and stage routing."""

import pytest
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication import fuzzy_dups

from experiments.datakit import reference_pipeline, zephyr_benchmark
from experiments.datakit.materialize_zephyr_benchmark_sample import (
    BENCHMARK_SAMPLE_INPUTS_DIR,
    benchmark_zephyr_context,
)
from experiments.datakit.reference_pipeline import SMOKE_SCALE, SOURCE_DISCOVERY_DEPTHS
from experiments.datakit.zephyr_benchmark import (
    BenchmarkTarget,
    SourceShardStats,
    _benchmark_steps,
    _resolve_sources,
    _select_source_fraction,
    _source_shard_stats,
    _target_steps,
)

_SAMPLE_PREFIX = "gs://marin-test/sample_100b"
_ROOT_KEY = "marin-test/sample_100b"


@pytest.fixture(autouse=True)
def _marin_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-test")


class _FakeObjectStore:
    """In-memory stand-in for a gcsfs/s3fs filesystem."""

    def __init__(self, files: dict[str, int]):
        self._files = files

    def find(self, path: str, detail: bool = True) -> dict:
        assert detail
        return {key: {"size": size} for key, size in self._files.items() if key.startswith(path + "/")}


def _patch_store(monkeypatch: pytest.MonkeyPatch, files: dict[str, int]) -> None:
    store = _FakeObjectStore(files)
    monkeypatch.setattr(zephyr_benchmark, "url_to_fs", lambda _url: (store, _ROOT_KEY))


def test_source_shard_stats_groups_parquet_by_source(monkeypatch):
    _patch_store(
        monkeypatch,
        {
            f"{_ROOT_KEY}/hplt_v3/outputs/main/shard-0000.parquet": 100,
            f"{_ROOT_KEY}/hplt_v3/outputs/main/shard-0001.parquet": 50,
            f"{_ROOT_KEY}/cp/wikiteam/outputs/main/a.parquet": 500,
            f"{_ROOT_KEY}/hplt_v3/.artifact.json": 10,
            f"{_ROOT_KEY}/{BENCHMARK_SAMPLE_INPUTS_DIR}/datakit/minhash/hplt_abc123/outputs/shard.parquet": 25,
        },
    )

    assert _source_shard_stats(_SAMPLE_PREFIX) == {
        "hplt_v3": SourceShardStats(150, 2),
        "cp/wikiteam": SourceShardStats(500, 1),
    }


@pytest.mark.parametrize(
    "stray_key",
    ["stray/notes.parquet", "outputs/main/top.parquet"],
)
def test_source_shard_stats_rejects_parquet_outside_source_output_dirs(monkeypatch, stray_key):
    # A stray parquet would otherwise become a bogus source: in fraction mode it
    # is selected first (one tiny shard) and later rejected by sample_sources.
    _patch_store(
        monkeypatch,
        {
            f"{_ROOT_KEY}/hplt_v3/outputs/main/shard-0000.parquet": 100,
            f"{_ROOT_KEY}/{stray_key}": 10,
        },
    )

    with pytest.raises(ValueError):
        _source_shard_stats(_SAMPLE_PREFIX)


def test_source_shard_stats_requires_object_store_prefix():
    with pytest.raises(ValueError):
        _source_shard_stats("/tmp/local-sample")


def test_source_shard_stats_rejects_source_deeper_than_discovery_depth(monkeypatch):
    # sample_sources() only discovers a source's .artifact.json up to
    # len(SOURCE_DISCOVERY_DEPTHS) path segments; a deeper source would parse
    # fine here but be unfindable there, so --source-fraction could select a
    # name sample_sources() later rejects with KeyError.
    too_deep = "/".join(["a"] * (len(SOURCE_DISCOVERY_DEPTHS) + 1))
    _patch_store(monkeypatch, {f"{_ROOT_KEY}/{too_deep}/outputs/main/shard-0000.parquet": 100})

    with pytest.raises(ValueError):
        _source_shard_stats(_SAMPLE_PREFIX)


def test_select_source_fraction_prefers_shard_dense_sources():
    # 10000 bytes total; a 0.2 byte budget must land on the two 10-shard-per-
    # 1000-byte sources and skip the single-shard 8000-byte source entirely.
    stats = {
        "sparse_huge": SourceShardStats(8000, 1),
        "dense_small": SourceShardStats(1000, 10),
        "dense_mid": SourceShardStats(1000, 5),
    }

    assert _select_source_fraction(stats, 0.2) == ["dense_small", "dense_mid"]


def test_select_source_fraction_full_selects_every_source():
    stats = {"a": SourceShardStats(10, 1), "b": SourceShardStats(30, 2)}

    assert _select_source_fraction(stats, 1.0) == ["a", "b"]


@pytest.mark.parametrize("fraction", [0.0, -0.5, 1.5])
def test_select_source_fraction_rejects_out_of_range_fraction(fraction):
    with pytest.raises(ValueError):
        _select_source_fraction({"a": SourceShardStats(10, 1)}, fraction)


def test_resolve_sources_fraction_returns_selected_names(monkeypatch):
    # The CLI passes this list straight to sample_sources; returning None here
    # would silently benchmark the full sample instead of the selected subset.
    files = {f"{_ROOT_KEY}/dense/outputs/main/{i}.parquet": 100 for i in range(4)}
    files[f"{_ROOT_KEY}/sparse/outputs/main/only.parquet"] = 1000
    _patch_store(monkeypatch, files)

    selected = _resolve_sources(_SAMPLE_PREFIX, sources_arg=None, source_fraction=0.25, pool_workers=2)

    assert selected == ["dense"]


def test_resolve_sources_rejects_pool_workers_beyond_available_shards(monkeypatch):
    _patch_store(
        monkeypatch,
        {
            f"{_ROOT_KEY}/hplt_v3/outputs/main/shard-0000.parquet": 100,
            f"{_ROOT_KEY}/cp/wikiteam/outputs/main/a.parquet": 500,
        },
    )

    with pytest.raises(ValueError):
        _resolve_sources(_SAMPLE_PREFIX, sources_arg="all", source_fraction=None, pool_workers=4)


def test_resolve_sources_rejects_unknown_source_names(monkeypatch):
    _patch_store(monkeypatch, {f"{_ROOT_KEY}/hplt_v3/outputs/main/shard-0000.parquet": 100})

    with pytest.raises(KeyError):
        _resolve_sources(_SAMPLE_PREFIX, sources_arg="hplt_v3,nope", source_fraction=None, pool_workers=1)


def test_resolve_sources_does_not_double_count_a_repeated_source(monkeypatch):
    # sample_sources() collapses "hplt_v3,hplt_v3" into one source, so the guard
    # must not sum its 1 shard twice and let --pool-workers=2 pass.
    _patch_store(monkeypatch, {f"{_ROOT_KEY}/hplt_v3/outputs/main/shard-0000.parquet": 100})

    with pytest.raises(ValueError):
        _resolve_sources(_SAMPLE_PREFIX, sources_arg="hplt_v3,hplt_v3", source_fraction=None, pool_workers=2)


def _patch_benchmark_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        zephyr_benchmark,
        "sample_sources",
        lambda sample_prefix, names, run_tag="": {
            name: StepSpec(
                name=f"sample/{name}",
                override_output_path=f"{sample_prefix}/{name}",
                hash_attrs={"run_tag": run_tag},
            )
            for name in names or ["a"]
        },
    )
    monkeypatch.setattr(
        zephyr_benchmark,
        "_benchmark_output_prefix",
        lambda sample_prefix, run_tag: f"{sample_prefix}/benchmarks/{run_tag}",
    )


def test_shuffle_target_reads_sample_minhash_and_writes_fresh_output(monkeypatch):
    _patch_benchmark_graph(monkeypatch)
    monkeypatch.setattr(zephyr_benchmark, "step_is_built", lambda _step: True)
    requested_paths: list[str] = []
    monkeypatch.setattr(fuzzy_dups, "read_artifact", lambda path, _cls: requested_paths.append(path))
    monkeypatch.setattr(fuzzy_dups, "compute_fuzzy_dups_attrs", lambda **kwargs: kwargs["inputs"])

    context = benchmark_zephyr_context("test-benchmark", SMOKE_SCALE, 8)
    steps = _benchmark_steps(
        sample_prefix=_SAMPLE_PREFIX,
        selected_sources=["a"],
        run_tag="shuffle-v2",
        target=BenchmarkTarget.SHUFFLE,
        scale=SMOKE_SCALE,
        zephyr_context=context,
    )

    assert steps.fuzzy_dedup.output_path.startswith(f"{_SAMPLE_PREFIX}/benchmarks/shuffle-v2/")
    sample_inputs = f"{_SAMPLE_PREFIX}/{BENCHMARK_SAMPLE_INPUTS_DIR}/"
    assert all(step.output_path.startswith(sample_inputs) for step in steps.fuzzy_dedup.deps)
    assert steps.fuzzy_dedup.fn is not None
    steps.fuzzy_dedup.fn(steps.fuzzy_dedup.output_path)
    assert requested_paths
    assert all(path.startswith(sample_inputs) for path in requested_paths)


def test_shuffle_target_requires_every_sample_minhash_artifact(monkeypatch):
    _patch_benchmark_graph(monkeypatch)
    monkeypatch.setattr(zephyr_benchmark, "step_is_built", lambda _step: False)

    with pytest.raises(RuntimeError):
        _benchmark_steps(
            sample_prefix=_SAMPLE_PREFIX,
            selected_sources=["a"],
            run_tag="shuffle-v2",
            target=BenchmarkTarget.SHUFFLE,
            scale=SMOKE_SCALE,
            zephyr_context=benchmark_zephyr_context("test-benchmark", SMOKE_SCALE, 8),
        )


def test_map_and_shuffle_targets_select_disjoint_stage_families():
    sources = {"a": StepSpec(name="sample/a")}
    steps = reference_pipeline.zephyr_datakit_steps(sources)

    assert _target_steps(steps, BenchmarkTarget.MAP) == [steps.tokenize["a"], steps.minhash["a"]]
    assert _target_steps(steps, BenchmarkTarget.SHUFFLE) == [steps.exact_dedup, steps.fuzzy_dedup]
