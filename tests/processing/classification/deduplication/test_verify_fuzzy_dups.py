# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from pathlib import Path
from random import Random

import pyarrow.parquet as pq
import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from fray.types import ActorConfig, ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import datakit_source_key
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData, FuzzyDupsPerSource
from marin.processing.classification.deduplication.fuzzy_minhash import MinHashAttrData, MinHashParams
from marin.processing.classification.deduplication.fuzzy_verification import FuzzyVerificationParams
from marin.processing.classification.deduplication.verify_fuzzy_dups import (
    FuzzyVerificationStoreConfig,
    LocalRepresentativeParams,
)
from marin.processing.classification.deduplication.verify_fuzzy_dups import (
    verify_fuzzy_dups as _verify_fuzzy_dups,
)
from zephyr.stage_io import ZephyrWorkerError
from zephyr.writers import write_parquet_file

TEST_MINHASH_PARAMS = MinHashParams(num_perms=8, num_bands=4, ngram_size=5, seed=0)
TEST_LOCAL_PARAMS = LocalRepresentativeParams(
    maximum_comparisons_per_document=4,
    maximum_representatives_per_cluster=8,
    maximum_local_representative_chars=10_000,
    maximum_local_representative_chars_per_cluster=40_000,
    minimum_local_token_ngram_jaccard=0.9,
    local_char_ngram_size=13,
    minimum_local_char_jaccard=0.9,
)
TEST_STORE_CONFIG = FuzzyVerificationStoreConfig(
    max_actors=2,
    actor_resources=ResourceConfig(cpu=1, ram="256m"),
    actor_config=ActorConfig(max_concurrency=8, max_task_retries=1),
    max_actor_bytes=1 << 20,
    recovery_timeout=30,
    ready_timeout=30,
    lookup_batch_size=2,
)
verify_fuzzy_dups = partial(_verify_fuzzy_dups, store_config=TEST_STORE_CONFIG)


@pytest.fixture(autouse=True)
def flow_backend_ctx():
    with set_current_client(LocalClient()):
        yield


def _write_source(
    *,
    root: Path,
    name: str,
    shards: dict[str, list[dict]],
) -> tuple[str, NormalizedData]:
    main_dir = root / name / "outputs" / "main"
    dup_dir = root / name / "outputs" / "dups"
    main_dir.mkdir(parents=True)
    dup_dir.mkdir(parents=True)
    for basename, rows in shards.items():
        write_parquet_file(sorted(rows, key=lambda row: row["id"]), str(main_dir / basename))
    source = NormalizedData(main_output_dir=str(main_dir), dup_output_dir=str(dup_dir), counters={})
    return datakit_source_key(source.main_output_dir), source


def _write_candidates(
    *,
    root: Path,
    rows_by_source: dict[str, dict[str, list[dict]]],
) -> FuzzyDupsAttrData:
    sources: dict[str, FuzzyDupsPerSource] = {}
    for source_index, (source_key, shards) in enumerate(sorted(rows_by_source.items())):
        attr_dir = root / "candidates" / f"source_{source_index:03d}"
        attr_dir.mkdir(parents=True)
        for basename, rows in shards.items():
            write_parquet_file(sorted(rows, key=lambda row: row["id"]), str(attr_dir / basename))
        sources[source_key] = FuzzyDupsPerSource(attr_dir=str(attr_dir))
    return FuzzyDupsAttrData(params=TEST_MINHASH_PARAMS, sources=sources, counters={})


def _write_minhash(
    *,
    root: Path,
    name: str,
    source: NormalizedData,
    buckets_by_id: dict[str, list[str]] | None = None,
) -> MinHashAttrData:
    attr_dir = root / "minhash" / name
    attr_dir.mkdir(parents=True)
    for source_path in sorted(Path(source.main_output_dir).glob("*.parquet")):
        rows = [
            {
                "id": row["id"],
                "buckets": (buckets_by_id or {}).get(row["id"], [f"bucket-{row['id']}"]),
            }
            for row in pq.read_table(source_path, columns=["id"]).to_pylist()
        ]
        write_parquet_file(rows, str(attr_dir / source_path.name))
    return MinHashAttrData(
        params=TEST_MINHASH_PARAMS,
        source_key=datakit_source_key(source.main_output_dir),
        attr_dir=str(attr_dir),
        counters={},
    )


def _output_rows(verified, source_key: str) -> list[dict]:
    rows = []
    for path in sorted(Path(verified.sources[source_key].attr_dir).glob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def test_verifier_accepts_only_direct_subset_and_filters_singletons(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    representative = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    accepted = "alpha beta gamma delta epsilon zeta eta theta"
    rejected = "alpha beta gamma delta epsilon zeta eta other theta"
    source_key, source = _write_source(
        root=tmp_path,
        name="source",
        shards={
            "part-00000.parquet": [
                {"id": "accepted", "text": accepted},
                {"id": "rejected", "text": rejected},
                {"id": "representative", "text": representative},
            ],
            "part-00001.parquet": [{"id": "singleton", "text": "a document outside all candidate clusters"}],
        },
    )
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={
            source_key: {
                "part-00000.parquet": [
                    {"id": "accepted", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False},
                    {"id": "rejected", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False},
                    {"id": "representative", "dup_cluster_id": "cluster-a", "is_cluster_canonical": True},
                ]
            }
        },
    )

    verified = verify_fuzzy_dups(
        normalized_sources={"source": source},
        minhash_sources={"source": _write_minhash(root=tmp_path, name="source", source=source)},
        candidates=candidates,
        output_path=str(tmp_path / "verified"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=TEST_LOCAL_PARAMS,
    )

    assert verified.counters["dedup/fuzzy/verification/memory_store/actors"] == 2
    assert verified.counters["dedup/fuzzy/verification/memory_store/items"] == 3
    assert _output_rows(verified, source_key) == [
        {
            "id": "accepted",
            "dup_doc": True,
            "dup_cluster_id": "cluster-a",
            "dup_representative_id": "representative",
            "dup_representative_source_key": source_key,
            "dup_representative_kind": "cluster_canonical",
            "dup_shared_lsh_buckets": 0,
            "dup_comparisons": 1,
            "dup_member_containment": 1.0,
            "dup_jaccard": 0.6,
            "dup_under_tokenized": False,
            "dup_char_jaccard": None,
            "dup_local_token_sequence_equal": None,
            "dup_local_char_jaccard": None,
        }
    ]
    empty_path = Path(verified.sources[source_key].attr_dir) / "part-00001.parquet"
    assert empty_path.exists()
    empty = pq.read_table(empty_path)
    assert empty.num_rows == 0
    assert empty.schema.names == [
        "id",
        "dup_doc",
        "dup_cluster_id",
        "dup_representative_id",
        "dup_representative_source_key",
        "dup_representative_kind",
        "dup_shared_lsh_buckets",
        "dup_comparisons",
        "dup_member_containment",
        "dup_jaccard",
        "dup_under_tokenized",
        "dup_char_jaccard",
        "dup_local_token_sequence_equal",
        "dup_local_char_jaccard",
    ]
    assert verified.counters["dedup/fuzzy/verification/candidate_members"] == 3
    assert verified.counters["dedup/fuzzy/verification/candidate_shards_missing"] == 1
    assert verified.counters["dedup/fuzzy/verification/clusters"] == 1
    assert verified.counters["dedup/fuzzy/verification/cluster_members"] == 3
    assert verified.counters["dedup/fuzzy/verification/decision/accepted"] == 1
    assert verified.counters["dedup/fuzzy/verification/decision/retained_no_match"] == 1
    assert verified.counters["dedup/fuzzy/verification/comparison/containment_below_threshold"] == 1
    assert verified.sources[source_key].source_tag == "source_000"


def test_representative_selection_is_stable_across_input_order(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    representative_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    member_text = "alpha beta gamma delta epsilon zeta eta theta"
    source_a_key, source_a = _write_source(
        root=tmp_path,
        name="source-a",
        shards={"part-00000.parquet": [{"id": "representative", "text": representative_text}]},
    )
    source_b_key, source_b = _write_source(
        root=tmp_path,
        name="source-b",
        shards={"part-00000.parquet": [{"id": "member", "text": member_text}]},
    )
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={
            source_a_key: {
                "part-00000.parquet": [
                    {"id": "representative", "dup_cluster_id": "cluster-a", "is_cluster_canonical": True}
                ]
            },
            source_b_key: {
                "part-00000.parquet": [{"id": "member", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False}]
            },
        },
    )
    minhash_a = _write_minhash(root=tmp_path, name="source-a", source=source_a)
    minhash_b = _write_minhash(root=tmp_path, name="source-b", source=source_b)

    first = verify_fuzzy_dups(
        normalized_sources={"z-source-a": source_a, "a-source-b": source_b},
        minhash_sources={
            "z-source-a": minhash_a,
            "a-source-b": minhash_b,
        },
        candidates=candidates,
        output_path=str(tmp_path / "verified-first"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=TEST_LOCAL_PARAMS,
    )
    second = verify_fuzzy_dups(
        normalized_sources={"a-source-b": source_b, "z-source-a": source_a},
        minhash_sources={
            "a-source-b": minhash_b,
            "z-source-a": minhash_a,
        },
        candidates=candidates,
        output_path=str(tmp_path / "verified-second"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=TEST_LOCAL_PARAMS,
    )

    assert _output_rows(first, source_a_key) == _output_rows(second, source_a_key) == []
    assert _output_rows(first, source_b_key) == _output_rows(second, source_b_key)
    assert _output_rows(first, source_b_key)[0]["dup_representative_source_key"] == source_a_key


def test_verifier_defers_exact_copies_to_global_exact_dedup(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    text = "alpha beta gamma delta epsilon zeta eta theta"
    source_a_key, source_a = _write_source(
        root=tmp_path,
        name="source-a",
        shards={"part-00000.parquet": [{"id": "same", "text": text}]},
    )
    source_b_key, source_b = _write_source(
        root=tmp_path,
        name="source-b",
        shards={"part-00000.parquet": [{"id": "same", "text": text}]},
    )
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={
            source_a_key: {
                "part-00000.parquet": [{"id": "same", "dup_cluster_id": "cluster-a", "is_cluster_canonical": True}]
            },
            source_b_key: {
                "part-00000.parquet": [{"id": "same", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False}]
            },
        },
    )

    verified = verify_fuzzy_dups(
        normalized_sources={"z-source-a": source_a, "a-source-b": source_b},
        minhash_sources={
            "z-source-a": _write_minhash(root=tmp_path, name="source-a", source=source_a),
            "a-source-b": _write_minhash(root=tmp_path, name="source-b", source=source_b),
        },
        candidates=candidates,
        output_path=str(tmp_path / "verified"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=TEST_LOCAL_PARAMS,
    )

    assert _output_rows(verified, source_a_key) == []
    assert _output_rows(verified, source_b_key) == []
    assert verified.counters["dedup/fuzzy/verification/decision/delegated_global_exact"] == 1


def test_verifier_recovers_duplicate_through_local_representative(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    local_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau"
    duplicate_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma"
    source_key, source = _write_source(
        root=tmp_path,
        name="source",
        shards={
            "part-00000.parquet": [
                {"id": "a-local", "text": local_text},
                {"id": "b-duplicate", "text": duplicate_text},
                {"id": "canonical", "text": "one two three four five six seven eight nine ten eleven twelve"},
            ]
        },
    )
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={
            source_key: {
                "part-00000.parquet": [
                    {"id": "a-local", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False},
                    {"id": "b-duplicate", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False},
                    {"id": "canonical", "dup_cluster_id": "cluster-a", "is_cluster_canonical": True},
                ]
            }
        },
    )
    minhash = _write_minhash(
        root=tmp_path,
        name="source",
        source=source,
        buckets_by_id={
            "a-local": ["local-a", "local-b"],
            "b-duplicate": ["local-a", "local-b"],
            "canonical": ["canonical"],
        },
    )

    verified = verify_fuzzy_dups(
        normalized_sources={"source": source},
        minhash_sources={"source": minhash},
        candidates=candidates,
        output_path=str(tmp_path / "verified"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=TEST_LOCAL_PARAMS,
    )

    rows = _output_rows(verified, source_key)
    assert [row["id"] for row in rows] == ["b-duplicate"]
    assert rows[0]["dup_representative_id"] == "a-local"
    assert rows[0]["dup_representative_kind"] == "local_representative"
    assert rows[0]["dup_shared_lsh_buckets"] == 2
    assert rows[0]["dup_comparisons"] == 2
    assert rows[0]["dup_local_token_sequence_equal"] is False
    assert rows[0]["dup_local_char_jaccard"] >= 0.9
    assert verified.counters["dedup/fuzzy/verification/accepted_representative/local_representative"] == 1


def test_local_verifier_accepts_equal_token_sequences_with_different_whitespace(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    tokens = [f"t{index}" for index in range(40)]
    representative_text = "   ".join(tokens)
    member_text = "\t".join(tokens)
    source_key, source = _write_source(
        root=tmp_path,
        name="source",
        shards={
            "part-00000.parquet": [
                {"id": "a-local", "text": representative_text},
                {"id": "b-member", "text": member_text},
                {"id": "canonical", "text": "one two three four five six seven"},
            ]
        },
    )
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={
            source_key: {
                "part-00000.parquet": [
                    {"id": "a-local", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False},
                    {"id": "b-member", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False},
                    {"id": "canonical", "dup_cluster_id": "cluster-a", "is_cluster_canonical": True},
                ]
            }
        },
    )
    minhash = _write_minhash(
        root=tmp_path,
        name="source",
        source=source,
        buckets_by_id={"a-local": ["local"], "b-member": ["local"], "canonical": ["canonical"]},
    )

    verified = verify_fuzzy_dups(
        normalized_sources={"source": source},
        minhash_sources={"source": minhash},
        candidates=candidates,
        output_path=str(tmp_path / "verified"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=TEST_LOCAL_PARAMS,
    )

    row = _output_rows(verified, source_key)[0]
    assert row["id"] == "b-member"
    assert row["dup_representative_kind"] == "local_representative"
    assert row["dup_local_token_sequence_equal"] is True
    assert row["dup_local_char_jaccard"] is None


def test_local_verifier_rejects_token_ngram_saturation(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))

    def digit_text(seed: int, size: int) -> str:
        random = Random(seed)
        return " ".join(str(random.randrange(10)) for _ in range(size))

    representative_text = digit_text(1, 40_000)
    member_text = digit_text(2, 20_000)
    source_key, source = _write_source(
        root=tmp_path,
        name="source",
        shards={
            "part-00000.parquet": [
                {"id": "a-local", "text": representative_text},
                {"id": "b-member", "text": member_text},
                {"id": "canonical", "text": "one two three four five six seven"},
            ]
        },
    )
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={
            source_key: {
                "part-00000.parquet": [
                    {"id": "a-local", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False},
                    {"id": "b-member", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False},
                    {"id": "canonical", "dup_cluster_id": "cluster-a", "is_cluster_canonical": True},
                ]
            }
        },
    )
    minhash = _write_minhash(
        root=tmp_path,
        name="source",
        source=source,
        buckets_by_id={"a-local": ["local"], "b-member": ["local"], "canonical": ["canonical"]},
    )
    local_params = LocalRepresentativeParams(
        maximum_comparisons_per_document=2,
        maximum_representatives_per_cluster=8,
        maximum_local_representative_chars=100_000,
        maximum_local_representative_chars_per_cluster=200_000,
        minimum_local_token_ngram_jaccard=0.98,
        local_char_ngram_size=13,
        minimum_local_char_jaccard=0.98,
    )

    verified = verify_fuzzy_dups(
        normalized_sources={"source": source},
        minhash_sources={"source": minhash},
        candidates=candidates,
        output_path=str(tmp_path / "verified"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=local_params,
    )

    assert _output_rows(verified, source_key) == []
    assert verified.counters["dedup/fuzzy/verification/comparison/local_char_jaccard_below_threshold"] == 1


def test_local_representative_selection_is_stable_across_input_order(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    local_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau"
    duplicate_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma"
    source_a_key, source_a = _write_source(
        root=tmp_path,
        name="source-a",
        shards={"part-00000.parquet": [{"id": "canonical", "text": "one two three four five six seven"}]},
    )
    source_b_key, source_b = _write_source(
        root=tmp_path,
        name="source-b",
        shards={"part-00000.parquet": [{"id": "a-local", "text": local_text}]},
    )
    source_c_key, source_c = _write_source(
        root=tmp_path,
        name="source-c",
        shards={"part-00000.parquet": [{"id": "z-duplicate", "text": duplicate_text}]},
    )
    sources = {
        "source-a": (source_a_key, source_a),
        "source-b": (source_b_key, source_b),
        "source-c": (source_c_key, source_c),
    }
    ids = {"source-a": "canonical", "source-b": "a-local", "source-c": "z-duplicate"}
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={
            source_key: {
                "part-00000.parquet": [
                    {
                        "id": ids[source_name],
                        "dup_cluster_id": "cluster-a",
                        "is_cluster_canonical": source_name == "source-a",
                    }
                ]
            }
            for source_name, (source_key, _source) in sources.items()
        },
    )
    minhash_sources = {
        source_name: _write_minhash(
            root=tmp_path,
            name=source_name,
            source=source,
            buckets_by_id={
                "canonical": ["canonical"],
                "a-local": ["local"],
                "z-duplicate": ["local"],
            },
        )
        for source_name, (_source_key, source) in sources.items()
    }

    first = verify_fuzzy_dups(
        normalized_sources={source_name: source for source_name, (_source_key, source) in sources.items()},
        minhash_sources=minhash_sources,
        candidates=candidates,
        output_path=str(tmp_path / "verified-first"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=TEST_LOCAL_PARAMS,
    )
    second = verify_fuzzy_dups(
        normalized_sources={
            source_name: source for source_name, (_source_key, source) in reversed(list(sources.items()))
        },
        minhash_sources={source_name: minhash_sources[source_name] for source_name in reversed(minhash_sources)},
        candidates=candidates,
        output_path=str(tmp_path / "verified-second"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=TEST_LOCAL_PARAMS,
    )

    assert _output_rows(first, source_c_key) == _output_rows(second, source_c_key)
    assert _output_rows(first, source_c_key)[0]["dup_representative_source_key"] == source_b_key
    assert _output_rows(first, source_c_key)[0]["dup_representative_kind"] == "local_representative"


def test_verifier_applies_the_stricter_token_jaccard_gate_to_local_matches(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    local_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau"
    member_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    source_key, source = _write_source(
        root=tmp_path,
        name="source",
        shards={
            "part-00000.parquet": [
                {"id": "a-local", "text": local_text},
                {"id": "b-member", "text": member_text},
                {"id": "canonical", "text": "one two three four five six seven"},
            ]
        },
    )
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={
            source_key: {
                "part-00000.parquet": [
                    {"id": "a-local", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False},
                    {"id": "b-member", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False},
                    {"id": "canonical", "dup_cluster_id": "cluster-a", "is_cluster_canonical": True},
                ]
            }
        },
    )
    minhash = _write_minhash(
        root=tmp_path,
        name="source",
        source=source,
        buckets_by_id={
            "a-local": ["local"],
            "b-member": ["local"],
            "canonical": ["canonical"],
        },
    )

    verified = verify_fuzzy_dups(
        normalized_sources={"source": source},
        minhash_sources={"source": minhash},
        candidates=candidates,
        output_path=str(tmp_path / "verified"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=TEST_LOCAL_PARAMS,
    )

    assert _output_rows(verified, source_key) == []
    assert verified.counters["dedup/fuzzy/verification/comparison/local_token_jaccard_below_threshold"] == 1


def test_verifier_ranks_local_nominees_and_bounds_comparisons(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    b_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau"
    target_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma"
    source_key, source = _write_source(
        root=tmp_path,
        name="source",
        shards={
            "part-00000.parquet": [
                {"id": "a-first", "text": "red orange yellow green blue indigo violet black white gray"},
                {"id": "b-best", "text": b_text},
                {"id": "canonical", "text": "one two three four five six seven eight nine ten eleven twelve"},
                {"id": "z-target", "text": target_text},
            ]
        },
    )
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={
            source_key: {
                "part-00000.parquet": [
                    {"id": candidate_id, "dup_cluster_id": "cluster-a", "is_cluster_canonical": is_canonical}
                    for candidate_id, is_canonical in (
                        ("a-first", False),
                        ("b-best", False),
                        ("canonical", True),
                        ("z-target", False),
                    )
                ]
            }
        },
    )
    minhash = _write_minhash(
        root=tmp_path,
        name="source",
        source=source,
        buckets_by_id={
            "a-first": ["shared-a"],
            "b-best": ["shared-b1", "shared-b2"],
            "canonical": ["canonical"],
            "z-target": ["shared-a", "shared-b1", "shared-b2"],
        },
    )
    local_params = LocalRepresentativeParams(
        maximum_comparisons_per_document=2,
        maximum_representatives_per_cluster=8,
        maximum_local_representative_chars=10_000,
        maximum_local_representative_chars_per_cluster=40_000,
        minimum_local_token_ngram_jaccard=0.9,
        local_char_ngram_size=13,
        minimum_local_char_jaccard=0.9,
    )

    verified = verify_fuzzy_dups(
        normalized_sources={"source": source},
        minhash_sources={"source": minhash},
        candidates=candidates,
        output_path=str(tmp_path / "verified"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=local_params,
    )

    rows = _output_rows(verified, source_key)
    assert [row["id"] for row in rows] == ["z-target"]
    assert rows[0]["dup_representative_id"] == "b-best"
    assert rows[0]["dup_comparisons"] == 2
    assert verified.counters["dedup/fuzzy/verification/comparison_limit_reached"] == 1
    assert (
        max(
            int(key.rsplit("/", maxsplit=1)[1])
            for key in verified.counters
            if key.startswith("dedup/fuzzy/verification/comparisons_per_document/")
        )
        == 2
    )


def test_verifier_bounds_representative_count_and_text(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    long_text = " ".join(f"token-{index}" for index in range(30))
    short_duplicate = " ".join(f"token-{index}" for index in range(29))
    rows = [
        {"id": "a-count", "text": "red orange yellow green blue indigo violet"},
        {"id": "b-count", "text": long_text},
        {"id": "canonical-count", "text": "one two three four five six seven eight nine ten"},
        {"id": "z-count", "text": short_duplicate},
        {"id": "a-long", "text": long_text},
        {"id": "canonical-long", "text": "north south east west up down left right"},
        {"id": "z-long", "text": short_duplicate},
    ]
    source_key, source = _write_source(
        root=tmp_path,
        name="source",
        shards={"part-00000.parquet": rows},
    )
    candidate_rows = [
        {
            "id": row["id"],
            "dup_cluster_id": "cluster-count" if row["id"].endswith("count") else "cluster-long",
            "is_cluster_canonical": row["id"].startswith("canonical"),
        }
        for row in rows
    ]
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={source_key: {"part-00000.parquet": candidate_rows}},
    )
    minhash = _write_minhash(
        root=tmp_path,
        name="source",
        source=source,
        buckets_by_id={
            "a-count": ["count-a"],
            "b-count": ["count-b"],
            "canonical-count": ["canonical-count"],
            "z-count": ["count-b"],
            "a-long": ["long"],
            "canonical-long": ["canonical-long"],
            "z-long": ["long"],
        },
    )
    local_params = LocalRepresentativeParams(
        maximum_comparisons_per_document=2,
        maximum_representatives_per_cluster=2,
        maximum_local_representative_chars=100,
        maximum_local_representative_chars_per_cluster=1_000,
        minimum_local_token_ngram_jaccard=0.9,
        local_char_ngram_size=13,
        minimum_local_char_jaccard=0.9,
    )

    verified = verify_fuzzy_dups(
        normalized_sources={"source": source},
        minhash_sources={"source": minhash},
        candidates=candidates,
        output_path=str(tmp_path / "verified"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=local_params,
    )

    assert _output_rows(verified, source_key) == []
    assert verified.counters["dedup/fuzzy/verification/representative_skipped/cluster_limit"] >= 1
    assert verified.counters["dedup/fuzzy/verification/representative_skipped/document_chars"] >= 1


def test_repeated_noncanonical_ids_stay_available_as_local_representatives(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    exact_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau"
    near_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma"
    source_a_key, source_a = _write_source(
        root=tmp_path,
        name="source-a",
        shards={"part-00000.parquet": [{"id": "canonical", "text": "one two three four five six seven"}]},
    )
    source_b_key, source_b = _write_source(
        root=tmp_path,
        name="source-b",
        shards={"part-00000.parquet": [{"id": "same", "text": exact_text}]},
    )
    source_c_key, source_c = _write_source(
        root=tmp_path,
        name="source-c",
        shards={"part-00000.parquet": [{"id": "same", "text": exact_text}]},
    )
    source_d_key, source_d = _write_source(
        root=tmp_path,
        name="source-d",
        shards={"part-00000.parquet": [{"id": "z-near", "text": near_text}]},
    )
    sources = {
        "source-a": (source_a_key, source_a),
        "source-b": (source_b_key, source_b),
        "source-c": (source_c_key, source_c),
        "source-d": (source_d_key, source_d),
    }
    id_by_source = {"source-a": "canonical", "source-b": "same", "source-c": "same", "source-d": "z-near"}
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={
            source_key: {
                "part-00000.parquet": [
                    {
                        "id": id_by_source[source_name],
                        "dup_cluster_id": "cluster-a",
                        "is_cluster_canonical": source_name == "source-a",
                    }
                ]
            }
            for source_name, (source_key, source) in sources.items()
        },
    )
    minhash_sources = {
        source_name: _write_minhash(
            root=tmp_path,
            name=source_name,
            source=source,
            buckets_by_id={
                "canonical": ["canonical"],
                "same": ["local"],
                "z-near": ["local"],
            },
        )
        for source_name, (_source_key, source) in sources.items()
    }

    verified = verify_fuzzy_dups(
        normalized_sources={source_name: source for source_name, (_source_key, source) in sources.items()},
        minhash_sources=minhash_sources,
        candidates=candidates,
        output_path=str(tmp_path / "verified"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=TEST_LOCAL_PARAMS,
    )

    assert _output_rows(verified, source_b_key) == []
    assert _output_rows(verified, source_c_key) == []
    near_rows = _output_rows(verified, source_d_key)
    assert [row["id"] for row in near_rows] == ["z-near"]
    assert near_rows[0]["dup_representative_id"] == "same"
    assert near_rows[0]["dup_representative_source_key"] == source_b_key
    assert verified.counters["dedup/fuzzy/verification/decision/delegated_global_exact"] == 2


def test_verifier_rejects_different_text_for_canonical_content_id(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    source_a_key, source_a = _write_source(
        root=tmp_path,
        name="source-a",
        shards={"part-00000.parquet": [{"id": "same-id", "text": "canonical text"}]},
    )
    source_b_key, source_b = _write_source(
        root=tmp_path,
        name="source-b",
        shards={"part-00000.parquet": [{"id": "same-id", "text": "different text"}]},
    )
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={
            source_a_key: {
                "part-00000.parquet": [{"id": "same-id", "dup_cluster_id": "cluster-a", "is_cluster_canonical": True}]
            },
            source_b_key: {
                "part-00000.parquet": [{"id": "same-id", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False}]
            },
        },
    )

    with pytest.raises(ZephyrWorkerError, match="different text for content ID"):
        verify_fuzzy_dups(
            normalized_sources={"source-a": source_a, "source-b": source_b},
            minhash_sources={
                "source-a": _write_minhash(root=tmp_path, name="source-a", source=source_a),
                "source-b": _write_minhash(root=tmp_path, name="source-b", source=source_b),
            },
            candidates=candidates,
            output_path=str(tmp_path / "verified"),
            verification_params=FuzzyVerificationParams(),
            local_representative_params=TEST_LOCAL_PARAMS,
        )


@pytest.mark.parametrize(
    "canonical_flags, expected_error",
    [
        ([False, False], "has no canonical member"),
        ([True, True], "has more than one canonical member"),
    ],
)
def test_verifier_requires_one_candidate_canonical(
    tmp_path,
    monkeypatch,
    canonical_flags,
    expected_error,
):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    source_key, source = _write_source(
        root=tmp_path,
        name="source",
        shards={
            "part-00000.parquet": [
                {"id": "first", "text": "alpha beta gamma delta"},
                {"id": "second", "text": "alpha beta gamma delta epsilon"},
            ]
        },
    )
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={
            source_key: {
                "part-00000.parquet": [
                    {
                        "id": candidate_id,
                        "dup_cluster_id": "cluster-a",
                        "is_cluster_canonical": is_canonical,
                    }
                    for candidate_id, is_canonical in zip(("first", "second"), canonical_flags, strict=True)
                ]
            }
        },
    )

    with pytest.raises(ZephyrWorkerError, match=expected_error):
        verify_fuzzy_dups(
            normalized_sources={"source": source},
            minhash_sources={"source": _write_minhash(root=tmp_path, name="source", source=source)},
            candidates=candidates,
            output_path=str(tmp_path / "verified"),
            verification_params=FuzzyVerificationParams(),
            local_representative_params=TEST_LOCAL_PARAMS,
        )


def test_verifier_rejects_mismatched_source_sets(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    source_key, source = _write_source(
        root=tmp_path,
        name="source",
        shards={"part-00000.parquet": [{"id": "doc", "text": "some text"}]},
    )
    candidates = _write_candidates(root=tmp_path, rows_by_source={source_key: {}})
    candidates.sources["extra/source"] = FuzzyDupsPerSource(attr_dir=str(tmp_path / "extra"))

    with pytest.raises(ValueError, match="source sets differ"):
        verify_fuzzy_dups(
            normalized_sources={"source": source},
            minhash_sources={"source": _write_minhash(root=tmp_path, name="source", source=source)},
            candidates=candidates,
            output_path=str(tmp_path / "verified"),
            verification_params=FuzzyVerificationParams(),
            local_representative_params=TEST_LOCAL_PARAMS,
        )
