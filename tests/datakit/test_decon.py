# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for datakit decon step."""

import gzip
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray import LocalClient, set_current_client
from marin.datakit.decon import NGramConfig, decon_to_parquet


@pytest.fixture(autouse=True)
def flow_backend_ctx():
    with set_current_client(LocalClient()):
        yield


def _write_input_parquet(path: Path, records: list[dict]) -> None:
    """Write datakit-normalized-shaped Parquet (id, text, partition_id)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), str(path))


def _write_eval_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _read_attributes(output_dir: Path) -> dict[str, dict]:
    """Concatenate every output parquet under *output_dir* and key by id."""
    rows: dict[str, dict] = {}
    for pf in sorted(output_dir.glob("part-*.parquet")):
        for row in pq.read_table(str(pf)).to_pylist():
            rows[row["id"]] = row
    return rows


@pytest.fixture
def fox_corpus(tmp_path: Path):
    """Two-partition fox-themed corpus inspired by tests/processing/classification/conftest.py.

    Returns a dict with paths for eval source, input parquet dir, and output dir.
    """
    eval_dir = tmp_path / "eval"
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    # Eval source: short questions we want to detect overlap with.
    eval_records = [
        {"id": "eval_arctic", "text": "Arctic predators have superior auditory capabilities for hunting beneath snow."},
        {"id": "eval_red", "text": "Red canids inhabit northern territories worldwide."},
    ]
    _write_eval_jsonl(eval_dir / "eval.jsonl.gz", eval_records)

    # Input partitions, datakit-shaped (id/text/partition_id).
    partition_0 = [
        {  # verbatim match with eval_arctic → contaminated
            "id": "doc_arctic_exact",
            "text": "Arctic predators have superior auditory capabilities for hunting beneath snow.",
            "partition_id": 0,
        },
        {  # 8/9 of 3-grams match eval_arctic (≥ 0.5) → contaminated
            "id": "doc_arctic_high",
            "text": "Arctic predators have superior auditory capabilities for hunting beneath thick snow.",
            "partition_id": 0,
        },
        {  # 1/6 3-grams match (one shared phrase) → below 0.5 → gated out
            "id": "doc_low_overlap",
            "text": "Many arctic predators have evolved in surprising ways across millennia.",
            "partition_id": 0,
        },
    ]
    partition_1 = [
        {  # verbatim match with eval_red → contaminated
            "id": "doc_red_exact",
            "text": "Red canids inhabit northern territories worldwide.",
            "partition_id": 1,
        },
        {  # no overlap at all
            "id": "doc_unique",
            "text": "Desert mammals possess oversized pinnae for thermal regulation.",
            "partition_id": 1,
        },
    ]
    _write_input_parquet(input_dir / "part-00000-of-00002.parquet", partition_0)
    _write_input_parquet(input_dir / "part-00001-of-00002.parquet", partition_1)

    return {
        "eval_dir": str(eval_dir),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
    }


def test_decon_ngram_flags_high_overlap_and_gates_low(fox_corpus):
    """n=3 with threshold=0.5: verbatim and high-overlap records flagged; low-overlap and unique gated out."""
    attrs = decon_to_parquet(
        input_path=fox_corpus["input_dir"],
        decontaminate_source=fox_corpus["eval_dir"],
        output_path=fox_corpus["output_dir"],
        ngram=NGramConfig(ngram_length=3, stride=0, overlap_threshold=0.5),
        estimated_doc_count=10_000,
        false_positive_rate=1e-9,
    )
    assert attrs.num_partitions == 2

    rows = _read_attributes(Path(fox_corpus["output_dir"]))
    assert rows["doc_arctic_exact"]["contaminated"] is True
    assert rows["doc_arctic_exact"]["max_overlap"] == 1.0

    assert rows["doc_arctic_high"]["contaminated"] is True
    assert rows["doc_arctic_high"]["max_overlap"] >= 0.5

    assert rows["doc_low_overlap"]["contaminated"] is False
    assert rows["doc_red_exact"]["contaminated"] is True
    assert rows["doc_unique"]["contaminated"] is False
    assert rows["doc_unique"]["max_overlap"] == 0.0


def test_decon_exact_paragraph_match(fox_corpus):
    """ngram=None: whole-paragraph match. Verbatim records flagged; near-match gated out (different bytes)."""
    decon_to_parquet(
        input_path=fox_corpus["input_dir"],
        decontaminate_source=fox_corpus["eval_dir"],
        output_path=fox_corpus["output_dir"],
        ngram=None,
        estimated_doc_count=10_000,
        false_positive_rate=1e-9,
    )

    rows = _read_attributes(Path(fox_corpus["output_dir"]))
    assert rows["doc_arctic_exact"]["contaminated"] is True
    assert rows["doc_arctic_exact"]["max_overlap"] == 1.0

    # "thick snow" → different bytes → not a paragraph-exact match.
    assert rows["doc_arctic_high"]["contaminated"] is False
    assert rows["doc_red_exact"]["contaminated"] is True
    assert rows["doc_unique"]["contaminated"] is False


def test_decon_preserves_partition_filenames(fox_corpus):
    """Output partition filenames mirror input filenames 1:1 (co-partitioning invariant)."""
    decon_to_parquet(
        input_path=fox_corpus["input_dir"],
        decontaminate_source=fox_corpus["eval_dir"],
        output_path=fox_corpus["output_dir"],
        ngram=NGramConfig(ngram_length=3, overlap_threshold=0.5),
        estimated_doc_count=10_000,
        false_positive_rate=1e-9,
    )
    input_names = sorted(p.name for p in Path(fox_corpus["input_dir"]).glob("*.parquet"))
    output_names = sorted(p.name for p in Path(fox_corpus["output_dir"]).glob("part-*.parquet"))
    assert input_names == output_names


def test_decon_output_schema(fox_corpus):
    """Output Parquet has exactly {id, partition_id, contaminated, max_overlap, matched_hashes}."""
    decon_to_parquet(
        input_path=fox_corpus["input_dir"],
        decontaminate_source=fox_corpus["eval_dir"],
        output_path=fox_corpus["output_dir"],
        ngram=NGramConfig(ngram_length=3, overlap_threshold=0.5),
        estimated_doc_count=10_000,
        false_positive_rate=1e-9,
    )
    output_files = sorted(Path(fox_corpus["output_dir"]).glob("part-*.parquet"))
    assert output_files, "expected at least one output partition"
    schema = pq.read_schema(str(output_files[0]))
    assert set(schema.names) == {"id", "partition_id", "contaminated", "max_overlap", "matched_hashes"}
    assert pa.types.is_string(schema.field("id").type)
    assert pa.types.is_integer(schema.field("partition_id").type)
    assert pa.types.is_boolean(schema.field("contaminated").type)
    assert pa.types.is_floating(schema.field("max_overlap").type)
    matched_field = schema.field("matched_hashes")
    assert pa.types.is_list(matched_field.type)
    assert matched_field.type.value_type == pa.uint64()


def test_decon_emits_eval_hash_index_sidecar(fox_corpus):
    """Build writes a hash → eval_id Parquet sidecar with the expected schema."""
    attrs = decon_to_parquet(
        input_path=fox_corpus["input_dir"],
        decontaminate_source=fox_corpus["eval_dir"],
        output_path=fox_corpus["output_dir"],
        ngram=NGramConfig(ngram_length=3, overlap_threshold=0.5),
        estimated_doc_count=10_000,
        false_positive_rate=1e-9,
    )
    sidecar = Path(attrs.eval_hash_index_path)
    assert sidecar.exists(), f"missing sidecar at {sidecar}"
    schema = pq.read_schema(str(sidecar))
    assert schema.field("hash").type == pa.uint64()
    assert pa.types.is_string(schema.field("eval_id").type)

    rows = pq.read_table(str(sidecar)).to_pylist()
    assert rows, "expected at least one (hash, eval_id) row"
    eval_ids = {r["eval_id"] for r in rows}
    # Both eval records contribute to the sidecar.
    assert eval_ids == {"eval_arctic", "eval_red"}


def test_decon_matched_hashes_join_recovers_eval_id(fox_corpus):
    """A contaminated record's matched_hashes joined with the sidecar attributes back to its eval."""
    attrs = decon_to_parquet(
        input_path=fox_corpus["input_dir"],
        decontaminate_source=fox_corpus["eval_dir"],
        output_path=fox_corpus["output_dir"],
        ngram=NGramConfig(ngram_length=3, overlap_threshold=0.5),
        estimated_doc_count=10_000,
        false_positive_rate=1e-9,
    )
    rows = _read_attributes(Path(fox_corpus["output_dir"]))
    hash_to_eval: dict[int, set[str]] = {}
    for r in pq.read_table(attrs.eval_hash_index_path).to_pylist():
        hash_to_eval.setdefault(r["hash"], set()).add(r["eval_id"])

    arctic_evals: set[str] = set()
    for h in rows["doc_arctic_exact"]["matched_hashes"]:
        arctic_evals |= hash_to_eval.get(h, set())
    assert arctic_evals == {"eval_arctic"}

    red_evals: set[str] = set()
    for h in rows["doc_red_exact"]["matched_hashes"]:
        red_evals |= hash_to_eval.get(h, set())
    assert red_evals == {"eval_red"}

    # Clean record has no matched hashes.
    assert rows["doc_unique"]["matched_hashes"] == []


@pytest.mark.parametrize(
    "threshold, expect_high_flagged",
    [(0.0, True), (0.5, True), (0.95, False), (1.0, False)],
)
def test_decon_overlap_threshold_gates(fox_corpus, threshold, expect_high_flagged):
    """Threshold gates which records are marked contaminated.

    The high-overlap record (doc_arctic_high) hits ~8/9 of eval_arctic's 3-grams (~0.89).
    It's flagged at thresholds ≤ 0.89 and gated above; pin the gate behavior across thresholds.
    """
    decon_to_parquet(
        input_path=fox_corpus["input_dir"],
        decontaminate_source=fox_corpus["eval_dir"],
        output_path=fox_corpus["output_dir"],
        ngram=NGramConfig(ngram_length=3, overlap_threshold=threshold),
        estimated_doc_count=10_000,
        false_positive_rate=1e-9,
    )
    rows = _read_attributes(Path(fox_corpus["output_dir"]))
    assert rows["doc_arctic_high"]["contaminated"] is expect_high_flagged
    # Verbatim record always flagged (max_overlap == 1.0).
    assert rows["doc_arctic_exact"]["contaminated"] is True


def test_decon_empty_input_raises(tmp_path: Path):
    """No .parquet files under input_path → FileNotFoundError."""
    eval_dir = tmp_path / "eval"
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_eval_jsonl(eval_dir / "eval.jsonl.gz", [{"id": "x", "text": "anything"}])
    input_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        decon_to_parquet(
            input_path=str(input_dir),
            decontaminate_source=str(eval_dir),
            output_path=str(output_dir),
            ngram=NGramConfig(ngram_length=3),
        )
