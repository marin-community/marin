# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.analysis.build_decon_val_sets as build
from scripts.analysis.build_decon_val_sets import build_intent, check_resume_allowed
from scripts.analysis.build_paranoid_val_keep_ids import keep_doc_indices


def test_keep_sets_nest_and_count_across_cutoffs() -> None:
    fully = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    # doc 5 is contaminated but not fully contained -> never kept at any cutoff
    max_j = {1: 0.6, 2: 0.8, 3: 0.95, 5: 0.99}

    keep_050 = keep_doc_indices(fully, max_j, 0.5)
    keep_075 = keep_doc_indices(fully, max_j, 0.75)
    keep_090 = keep_doc_indices(fully, max_j, 0.9)

    assert keep_050.tolist() == [0, 4]
    assert keep_075.tolist() == [0, 1, 4]
    assert keep_090.tolist() == [0, 1, 2, 4]
    assert set(keep_050) < set(keep_075) < set(keep_090)


def _intent(payloads: dict[str, dict]) -> dict:
    return build_intent(payloads)


def _payloads() -> dict[str, dict]:
    return {
        tag: {
            "cutoff": cut,
            "filter": "fully_contained_in_val_windows_and_max_train_jaccard_lt_cutoff",
            "keep_ids_xxh3": f"hash-{tag}",
            "expected_docs": docs,
            "expected_tokens": docs * 100,
        }
        for tag, cut, docs in [("j050", 0.5, 2), ("j075", 0.75, 3), ("j090", 0.9, 4)]
    }


def test_resume_requires_matching_intent() -> None:
    intent = _intent(_payloads())

    check_resume_allowed(intent, intent)  # exact match passes

    with pytest.raises(RuntimeError, match=r"no build_intent\.json"):
        check_resume_allowed(None, intent)

    changed = _payloads()
    changed["j075"]["keep_ids_xxh3"] = "different-hash"
    with pytest.raises(RuntimeError, match="intent mismatch"):
        check_resume_allowed(_intent(changed), intent)

    changed = _payloads()
    changed["j050"]["expected_tokens"] += 1
    with pytest.raises(RuntimeError, match="intent mismatch"):
        check_resume_allowed(_intent(changed), intent)


def test_filter_shard_routes_docs_to_nested_cutoff_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    keep = {
        "j050": {"ids": ["a"]},
        "j075": {"ids": ["a", "b"]},
        "j090": {
            "ids": ["a", "b", "c"],
            "doc_indices": [10, 11, 12],
            "max_jaccard_by_doc": {"11": 0.6, "12": 0.85},
        },
    }
    monkeypatch.setattr(build, "DECON_ROOT", str(tmp_path))
    monkeypatch.setattr(build, "_KEEP", keep)
    monkeypatch.setattr(build, "_KEEP_SETS", None)
    monkeypatch.setattr(build, "_KEEP_META", None)

    shard = tmp_path / "part-00000-of-00001.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["a", "b", "c", "dropped"],
                "text": ["ta", "tb", "tc", "td"],
                "shard": ["s0"] * 4,
                "row": [0, 1, 2, 3],
            }
        ),
        shard,
    )

    stats = list(build._filter_shard(str(shard)))
    assert stats == [{"input_shard": shard.name, "kept_j050": 1, "kept_j075": 2, "kept_j090": 3}]

    rows = {tag: pq.read_table(tmp_path / tag / "docs" / shard.name).to_pylist() for tag in ("j050", "j075", "j090")}
    assert [r["id"] for r in rows["j050"]] == ["a"]
    assert [r["id"] for r in rows["j075"]] == ["a", "b"]
    assert [r["id"] for r in rows["j090"]] == ["a", "b", "c"]
    # provenance columns survive: doc_index always set, max_jaccard null for never-matched docs
    assert rows["j090"][0] == {"id": "a", "text": "ta", "shard": "s0", "row": 0, "doc_index": 10, "max_jaccard": None}
    assert rows["j090"][1]["max_jaccard"] == 0.6
    assert rows["j090"][2]["max_jaccard"] == 0.85
