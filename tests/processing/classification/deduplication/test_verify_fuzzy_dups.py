# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import datakit_source_key
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData, FuzzyDupsPerSource
from marin.processing.classification.deduplication.fuzzy_minhash import MinHashParams
from marin.processing.classification.deduplication.fuzzy_verification import FuzzyVerificationParams
from marin.processing.classification.deduplication.verify_fuzzy_dups import verify_fuzzy_dups
from zephyr.writers import write_parquet_file

TEST_MINHASH_PARAMS = MinHashParams(num_perms=8, num_bands=4, ngram_size=5, seed=0)


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


def _output_rows(verified, source_key: str) -> list[dict]:
    rows = []
    for path in sorted(Path(verified.sources[source_key].attr_dir).glob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def test_verifier_accepts_only_direct_subset_and_writes_typed_empty_shards(tmp_path, monkeypatch):
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
        candidates=candidates,
        output_path=str(tmp_path / "verified"),
        verification_params=FuzzyVerificationParams(),
        max_parallelism=2,
    )

    assert _output_rows(verified, source_key) == [
        {
            "id": "accepted",
            "dup_doc": True,
            "dup_cluster_id": "cluster-a",
            "dup_representative_id": "representative",
            "dup_representative_source_key": source_key,
            "dup_verifier_version": "whitespace_3gram_subset_v1",
            "dup_member_containment": 1.0,
            "dup_jaccard": 0.6,
            "dup_under_tokenized": False,
            "dup_char_jaccard": None,
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
        "dup_verifier_version",
        "dup_member_containment",
        "dup_jaccard",
        "dup_under_tokenized",
        "dup_char_jaccard",
    ]
    assert verified.counters["dedup/fuzzy/verification/candidate_members"] == 3
    assert verified.counters["dedup/fuzzy/verification/decision/accepted"] == 1
    assert verified.counters["dedup/fuzzy/verification/decision/containment_below_threshold"] == 1


def test_representative_selection_is_stable_across_input_order_and_parallelism(tmp_path, monkeypatch):
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

    first = verify_fuzzy_dups(
        normalized_sources={"z-source-a": source_a, "a-source-b": source_b},
        candidates=candidates,
        output_path=str(tmp_path / "verified-first"),
        verification_params=FuzzyVerificationParams(),
        max_parallelism=1,
    )
    second = verify_fuzzy_dups(
        normalized_sources={"a-source-b": source_b, "z-source-a": source_a},
        candidates=candidates,
        output_path=str(tmp_path / "verified-second"),
        verification_params=FuzzyVerificationParams(),
        max_parallelism=4,
    )

    assert _output_rows(first, source_b_key) == _output_rows(second, source_b_key) == []
    assert _output_rows(first, source_a_key) == _output_rows(second, source_a_key)
    assert _output_rows(first, source_a_key)[0]["dup_representative_source_key"] == source_b_key


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
            candidates=candidates,
            output_path=str(tmp_path / "verified"),
            verification_params=FuzzyVerificationParams(),
            max_parallelism=1,
        )


def test_verifier_rejects_non_positive_parallelism(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    source_key, source = _write_source(
        root=tmp_path,
        name="source",
        shards={"part-00000.parquet": [{"id": "doc", "text": "some text"}]},
    )
    candidates = _write_candidates(root=tmp_path, rows_by_source={source_key: {}})

    with pytest.raises(ValueError, match="at least 1"):
        verify_fuzzy_dups(
            normalized_sources={"source": source},
            candidates=candidates,
            output_path=str(tmp_path / "verified"),
            verification_params=FuzzyVerificationParams(),
            max_parallelism=0,
        )
