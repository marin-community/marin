# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source-selection behavior for the Zephyr benchmark entrypoint.

``_source_shard_stats`` reads shard metadata through the ``url_to_fs`` boundary;
the fake store here mimics the gcsfs/s3fs ``find(detail=True)`` convention of
bucket-qualified, scheme-stripped keys.
"""

import pytest

from experiments.datakit import zephyr_benchmark
from experiments.datakit.zephyr_benchmark import _resolve_sources, _select_source_fraction, _source_shard_stats

_SAMPLE_PREFIX = "gs://marin-test/sample_100b"
_ROOT_KEY = "marin-test/sample_100b"


class _FakeObjectStore:
    """In-memory stand-in for a gcsfs/s3fs filesystem."""

    def __init__(self, files: dict[str, int]):
        self._files = files

    def find(self, path: str, detail: bool = True) -> dict:
        assert detail
        return {key: {"size": size} for key, size in self._files.items() if key.startswith(path + "/")}


def _patch_store(monkeypatch: pytest.MonkeyPatch, files: dict[str, int]) -> None:
    store = _FakeObjectStore(files)
    monkeypatch.setattr(zephyr_benchmark, "url_to_fs", lambda url: (store, _ROOT_KEY))


def test_source_shard_stats_groups_parquet_by_source(monkeypatch):
    _patch_store(
        monkeypatch,
        {
            f"{_ROOT_KEY}/hplt_v3/outputs/main/shard-0000.parquet": 100,
            f"{_ROOT_KEY}/hplt_v3/outputs/main/shard-0001.parquet": 50,
            f"{_ROOT_KEY}/cp/wikiteam/outputs/main/a.parquet": 500,
            f"{_ROOT_KEY}/hplt_v3/.artifact.json": 10,
        },
    )

    assert _source_shard_stats(_SAMPLE_PREFIX) == {
        "hplt_v3": (150, 2),
        "cp/wikiteam": (500, 1),
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

    with pytest.raises(ValueError, match="not under '<source>/outputs/main/'"):
        _source_shard_stats(_SAMPLE_PREFIX)


def test_source_shard_stats_requires_object_store_prefix():
    with pytest.raises(ValueError, match="gs:// or s3://"):
        _source_shard_stats("/tmp/local-sample")


def test_select_source_fraction_prefers_shard_dense_sources():
    # 10000 bytes total; a 0.2 byte budget must land on the two 10-shard-per-
    # 1000-byte sources and skip the single-shard 8000-byte source entirely.
    stats = {"sparse_huge": (8000, 1), "dense_small": (1000, 10), "dense_mid": (1000, 5)}

    assert _select_source_fraction(stats, 0.2) == ["dense_small", "dense_mid"]


def test_select_source_fraction_full_selects_every_source():
    stats = {"a": (10, 1), "b": (30, 2)}

    assert _select_source_fraction(stats, 1.0) == ["a", "b"]


@pytest.mark.parametrize("fraction", [0.0, -0.5, 1.5])
def test_select_source_fraction_rejects_out_of_range_fraction(fraction):
    with pytest.raises(ValueError, match="source-fraction"):
        _select_source_fraction({"a": (10, 1)}, fraction)


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

    with pytest.raises(ValueError, match="exceeds the 2 parquet shards"):
        _resolve_sources(_SAMPLE_PREFIX, sources_arg="all", source_fraction=None, pool_workers=4)


def test_resolve_sources_rejects_unknown_source_names(monkeypatch):
    _patch_store(monkeypatch, {f"{_ROOT_KEY}/hplt_v3/outputs/main/shard-0000.parquet": 100})

    with pytest.raises(KeyError, match="not found under"):
        _resolve_sources(_SAMPLE_PREFIX, sources_arg="hplt_v3,nope", source_fraction=None, pool_workers=1)
