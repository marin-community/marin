# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for datakit decon step."""

import gzip
import json
from pathlib import Path

import dupekit
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.datakit.decon import (
    EvalBloom,
    NGramConfig,
    _bloom_hash,
    _extract_ngrams,
    _load_drop_set,
    _paragraph_overlap_and_matches,
    bloom_paths,
    build_all_source_drop_sets,
    build_eval_bloom,
    build_source_drop_set,
    decon_to_parquet,
    merge_eval_blooms,
)
from marin.datakit.normalize import NormalizedData


@pytest.fixture(autouse=True)
def flow_backend_ctx():
    with set_current_client(LocalClient()):
        yield


def _write_input_parquet(path: Path, records: list[dict]) -> None:
    """Write datakit-normalized-shaped Parquet (id, text)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), str(path))


def _write_eval_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _read_attributes(output_dir: Path) -> dict[str, dict]:
    """Concatenate every output parquet under *output_dir* and key by id.

    Flattens the on-disk ``attributes`` struct (datakit convention) back into
    top-level keys so test assertions stay terse:
    ``rows[doc_id]["contaminated"]`` instead of ``rows[doc_id]["attributes"]["contaminated"]``.
    """
    rows: dict[str, dict] = {}
    for pf in sorted(output_dir.glob("outputs/main/part-*.parquet")):
        for row in pq.read_table(str(pf)).to_pylist():
            attrs = row.pop("attributes", {}) or {}
            rows[row["id"]] = {**row, **attrs}
    return rows


def _as_source(input_dir: Path) -> NormalizedData:
    """Wrap a flat directory of test Parquet files as a NormalizedData artifact."""
    return NormalizedData(
        main_output_dir=str(input_dir),
        dup_output_dir=str(input_dir / "_dups_unused"),
        counters={},
    )


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
        normalized_data=_as_source(Path(fox_corpus["input_dir"])),
        eval_data_sources=fox_corpus["eval_dir"],
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
        normalized_data=_as_source(Path(fox_corpus["input_dir"])),
        eval_data_sources=fox_corpus["eval_dir"],
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
        normalized_data=_as_source(Path(fox_corpus["input_dir"])),
        eval_data_sources=fox_corpus["eval_dir"],
        output_path=fox_corpus["output_dir"],
        ngram=NGramConfig(ngram_length=3, overlap_threshold=0.5),
        estimated_doc_count=10_000,
        false_positive_rate=1e-9,
    )
    input_names = sorted(p.name for p in Path(fox_corpus["input_dir"]).glob("*.parquet"))
    output_names = sorted(p.name for p in Path(fox_corpus["output_dir"]).glob("outputs/main/part-*.parquet"))
    assert input_names == output_names


def test_decon_output_schema(fox_corpus):
    """Output Parquet has exactly ``{id, partition_id, attributes: struct<contaminated, max_overlap, matched_hashes>}``.

    This is the datakit attribute convention consumed by
    :func:`marin.processing.classification.consolidate.consolidate` --
    ``id`` joinable on top, decon facts grouped under ``attributes``.
    """
    decon_to_parquet(
        normalized_data=_as_source(Path(fox_corpus["input_dir"])),
        eval_data_sources=fox_corpus["eval_dir"],
        output_path=fox_corpus["output_dir"],
        ngram=NGramConfig(ngram_length=3, overlap_threshold=0.5),
        estimated_doc_count=10_000,
        false_positive_rate=1e-9,
    )
    output_files = sorted(Path(fox_corpus["output_dir"]).glob("outputs/main/part-*.parquet"))
    assert output_files, "expected at least one output partition"
    schema = pq.read_schema(str(output_files[0]))
    assert set(schema.names) == {"id", "partition_id", "attributes"}
    assert pa.types.is_string(schema.field("id").type)
    assert pa.types.is_integer(schema.field("partition_id").type)

    attrs_field = schema.field("attributes")
    assert pa.types.is_struct(attrs_field.type)
    attrs_fields = {f.name: f for f in attrs_field.type}
    assert set(attrs_fields) == {"contaminated", "max_overlap", "matched_hashes"}
    assert pa.types.is_boolean(attrs_fields["contaminated"].type)
    assert pa.types.is_floating(attrs_fields["max_overlap"].type)
    assert pa.types.is_list(attrs_fields["matched_hashes"].type)
    assert attrs_fields["matched_hashes"].type.value_type == pa.uint64()


def test_decon_emits_eval_hash_index_sidecar(fox_corpus):
    """Build writes a hash → eval_id Parquet sidecar with the expected schema."""
    attrs = decon_to_parquet(
        normalized_data=_as_source(Path(fox_corpus["input_dir"])),
        eval_data_sources=fox_corpus["eval_dir"],
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
        normalized_data=_as_source(Path(fox_corpus["input_dir"])),
        eval_data_sources=fox_corpus["eval_dir"],
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
        normalized_data=_as_source(Path(fox_corpus["input_dir"])),
        eval_data_sources=fox_corpus["eval_dir"],
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
            normalized_data=_as_source(input_dir),
            eval_data_sources=str(eval_dir),
            output_path=str(output_dir),
            ngram=NGramConfig(ngram_length=3),
        )


def test_decon_eval_dir_with_sidecar_files_is_safe(tmp_path: Path):
    """Eval directories with non-data sidecars (README, _SUCCESS, hidden dirs) don't break build.

    Regression: _discover_eval_files previously yielded every non-dot file, then
    load_file rejected unsupported extensions and raised — killing the whole
    decon step. The discovery now filters by zephyr.readers.SUPPORTED_EXTENSIONS
    and skips hidden directories (mirrors normalize._discover_files).
    """
    eval_dir = tmp_path / "eval"
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    eval_dir.mkdir()
    input_dir.mkdir()

    # The actual eval file
    _write_eval_jsonl(eval_dir / "eval.jsonl.gz", [{"id": "eval", "text": "Hello big world example"}])
    # Common sidecar files that would crash load_file:
    (eval_dir / "README.md").write_text("# Eval corpus\nA description.\n")
    (eval_dir / "_SUCCESS").write_text("")
    (eval_dir / ".provenance.json").write_text('{"source": "wherever"}')
    # Hidden directory with stuff inside (.metrics/, .executor_info/, etc.)
    (eval_dir / ".metrics").mkdir()
    (eval_dir / ".metrics" / "stats.json").write_text('{"records": 1}')

    _write_input_parquet(
        input_dir / "part-00000-of-00001.parquet",
        [{"id": "doc", "text": "Hello big world example", "partition_id": 0}],
    )

    # Must not raise.
    decon_to_parquet(
        normalized_data=_as_source(input_dir),
        eval_data_sources=str(eval_dir),
        output_path=str(output_dir),
        ngram=NGramConfig(ngram_length=3, overlap_threshold=0.5),
        estimated_doc_count=1_000,
        false_positive_rate=1e-9,
    )
    # And the legitimate eval record still drove a match.
    rows = _read_attributes(output_dir)
    assert rows["doc"]["contaminated"] is True


def test_decon_fallback_eval_id_uses_full_path_for_uniqueness(tmp_path: Path):
    """Eval records without an ``id`` field get fallback eval_ids built from the full path.

    Regression: the fallback used os.path.basename, so two files at e.g.
    ``source/a/data.jsonl.gz`` and ``source/b/data.jsonl.gz`` would produce the
    same eval_id (``data.jsonl.gz::0``) for their row 0 — collapsing distinct
    eval records under one ID. Now uses the full path so they stay distinct.
    """
    eval_dir = tmp_path / "eval"
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    # Two eval files with the same basename in different subdirs. Records lack `id`.
    _write_eval_jsonl(eval_dir / "a" / "data.jsonl.gz", [{"text": "the quick brown fox jumps over"}])
    _write_eval_jsonl(eval_dir / "b" / "data.jsonl.gz", [{"text": "a wholly distinct evaluation sentence here"}])

    _write_input_parquet(
        input_dir / "part-00000-of-00001.parquet",
        [{"id": "doc", "text": "irrelevant input text", "partition_id": 0}],
    )

    attrs = decon_to_parquet(
        normalized_data=_as_source(input_dir),
        eval_data_sources=str(eval_dir),
        output_path=str(output_dir),
        ngram=NGramConfig(ngram_length=3, overlap_threshold=0.5),
        estimated_doc_count=1_000,
        false_positive_rate=1e-9,
    )

    # Sidecar should have two distinct eval_ids, one per eval file.
    sidecar = pq.read_table(attrs.eval_hash_index_path).to_pylist()
    eval_ids = {r["eval_id"] for r in sidecar}
    assert len(eval_ids) == 2, f"expected 2 distinct eval_ids, got {len(eval_ids)}: {eval_ids}"
    # Both should mention 'data.jsonl.gz' but be path-distinguishable (one under /a/, one under /b/).
    assert all("data.jsonl.gz" in e for e in eval_ids)
    assert any("/a/" in e for e in eval_ids)
    assert any("/b/" in e for e in eval_ids)


def test_decon_synthesizes_partition_id_from_shard_index(tmp_path: Path):
    """Decon synthesizes partition_id from shard.shard_idx (sorted-file order).

    Input records carry only id and text; the output's partition_id column is
    derived at read time from the shard index, matching the input's
    part-NNNNN-of-MMMMM naming.
    """
    eval_dir = tmp_path / "eval"
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    _write_eval_jsonl(eval_dir / "eval.jsonl.gz", [{"id": "eval", "text": "Arctic predators have superior auditory."}])

    # Two flat input partitions, but records do NOT include partition_id.
    pq.write_table(
        pa.Table.from_pylist([{"id": "doc0", "text": "Arctic predators have superior auditory."}]),
        str(input_dir / "part-00000-of-00002.parquet"),
    )
    pq.write_table(
        pa.Table.from_pylist([{"id": "doc1", "text": "Desert mammals possess oversized pinnae."}]),
        str(input_dir / "part-00001-of-00002.parquet"),
    )

    decon_to_parquet(
        normalized_data=_as_source(input_dir),
        eval_data_sources=str(eval_dir),
        output_path=str(output_dir),
        ngram=NGramConfig(ngram_length=3, overlap_threshold=0.5),
        estimated_doc_count=1_000,
        false_positive_rate=1e-9,
    )

    rows = _read_attributes(output_dir)
    # contaminated decisions still correct
    assert rows["doc0"]["contaminated"] is True
    assert rows["doc1"]["contaminated"] is False
    # partition_id synthesized: doc0 came from shard 0, doc1 from shard 1
    assert rows["doc0"]["partition_id"] == 0
    assert rows["doc1"]["partition_id"] == 1


def test_decon_short_paragraphs_below_ngram_length_contribute_nothing(tmp_path: Path):
    """Paragraphs with < ngram_length tokens are silently skipped in n-gram mode.

    Earlier versions (PR #5656 mid-stack) fell back to whole-paragraph hashing
    for paragraphs too short to form an n-gram. That created trivial collisions
    on common short paragraphs like ``"..."``, ``"A."``, etc., generating
    ~18% phantom-contamination flags in the MMLU vs nemotron-math smoke run.
    The fallback was removed; this test pins the new behavior.

    Trade-off: an eval with paragraphs shorter than ``ngram_length`` won't be
    matchable in n-gram mode. Callers who need that should either lower
    ``ngram_length`` or use ``ngram=None`` (exact paragraph mode).
    """
    eval_dir = tmp_path / "eval"
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    # Eval has a 2-token paragraph; with n=8 there are no ngrams → no bloom adds.
    _write_eval_jsonl(eval_dir / "eval.jsonl.gz", [{"id": "short_eval", "text": "Hello world"}])
    _write_input_parquet(
        input_dir / "part-00000-of-00001.parquet",
        [{"id": "doc_short_text", "text": "Hello world", "partition_id": 0}],
    )

    decon_to_parquet(
        normalized_data=_as_source(input_dir),
        eval_data_sources=str(eval_dir),
        output_path=str(output_dir),
        ngram=NGramConfig(ngram_length=8, overlap_threshold=0.5),
        estimated_doc_count=1_000,
        false_positive_rate=1e-9,
    )
    rows = _read_attributes(output_dir)
    # No matchable ngram → not contaminated, even though text is byte-identical to eval.
    assert rows["doc_short_text"]["contaminated"] is False
    assert rows["doc_short_text"]["max_overlap"] == 0.0
    assert rows["doc_short_text"]["matched_hashes"] == []


def test_double_newline_delimiter_spans_single_line_breaks(tmp_path: Path):
    """``paragraph_delimiter="\\n\\n"`` lets n-grams cross single ``\\n`` breaks.

    An eval item wrapped into short lines (each below ``ngram_length``) is
    invisible under the per-line ``"\\n"`` policy (no line forms an n-gram) but
    matchable under the true-paragraph ``"\\n\\n"`` policy, since a blank-line
    block is tokenized as a whole. Pins the short-line recall win (marin#6852).
    """
    words = [f"lexeme{i}" for i in range(10)]
    eval_text = " ".join(words)  # single line
    wrapped = "\n".join(" ".join(words[i : i + 2]) for i in range(0, len(words), 2))  # 2 words/line

    per_line = _run_decon_one_shot(
        tmp_path / "nl",
        eval_records=[{"id": "e", "text": eval_text}],
        input_records=[{"id": "doc", "text": wrapped, "partition_id": 0}],
        ngram=NGramConfig(ngram_length=5, overlap_threshold=0.5, paragraph_delimiter="\n"),
    )
    assert per_line["doc"]["contaminated"] is False  # short lines form no 5-grams

    true_para = _run_decon_one_shot(
        tmp_path / "nlnl",
        eval_records=[{"id": "e", "text": eval_text}],
        input_records=[{"id": "doc", "text": wrapped, "partition_id": 0}],
        ngram=NGramConfig(ngram_length=5, overlap_threshold=0.5, paragraph_delimiter="\n\n"),
    )
    assert true_para["doc"]["contaminated"] is True


def test_double_newline_delimiter_dilutes_isolated_matched_line(tmp_path: Path):
    """``"\\n\\n"`` measures overlap over the whole block, so a lone matched line
    among unrelated lines no longer saturates the fraction.

    Under ``"\\n"`` the matched line is its own paragraph → overlap 1.0 → flagged;
    under ``"\\n\\n"`` it is a small share of the block → below threshold → not
    flagged. Pins the isolated-line false-positive fix (marin#6852)."""
    eval_text = "distinctive alpha bravo charlie delta echo foxtrot"  # 7 tokens
    filler = "\n".join(f"unrelated filler line number {i} here" for i in range(8))
    block = filler + "\n" + eval_text + "\n" + filler  # all one blank-line-free block

    per_line = _run_decon_one_shot(
        tmp_path / "nl",
        eval_records=[{"id": "e", "text": eval_text}],
        input_records=[{"id": "doc", "text": block, "partition_id": 0}],
        ngram=NGramConfig(ngram_length=5, overlap_threshold=0.5, paragraph_delimiter="\n"),
    )
    assert per_line["doc"]["contaminated"] is True

    true_para = _run_decon_one_shot(
        tmp_path / "nlnl",
        eval_records=[{"id": "e", "text": eval_text}],
        input_records=[{"id": "doc", "text": block, "partition_id": 0}],
        ngram=NGramConfig(ngram_length=5, overlap_threshold=0.5, paragraph_delimiter="\n\n"),
    )
    assert true_para["doc"]["contaminated"] is False


# ---------------------------------------------------------------------------
# Functional boundary tests
#
# These exercise reasonable real-world contamination scenarios. Positive cases
# verify the algorithm catches what it should. Limitation cases are xfail with
# strict=True — if a future change improves the algorithm enough to handle
# them, the test will XPASS and force us to update the suite.
# ---------------------------------------------------------------------------


def _run_decon_one_shot(
    tmp_path: Path,
    *,
    eval_records: list[dict],
    input_records: list[dict],
    ngram: NGramConfig | None,
) -> dict[str, dict]:
    """Build eval + input fixtures, run decon, return id → output row mapping."""
    eval_dir = tmp_path / "eval"
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_eval_jsonl(eval_dir / "eval.jsonl.gz", eval_records)
    _write_input_parquet(input_dir / "part-00000-of-00001.parquet", input_records)
    decon_to_parquet(
        normalized_data=_as_source(input_dir),
        eval_data_sources=str(eval_dir),
        output_path=str(output_dir),
        ngram=ngram,
        estimated_doc_count=10_000,
        false_positive_rate=1e-9,
    )
    return _read_attributes(output_dir)


# ----- Positive cases (decon catches these) -----


def test_decon_catches_eval_paragraph_among_other_paragraphs(tmp_path: Path):
    """Pretraining record with the eval text as one of multiple paragraphs is flagged.

    Per-record score takes the max across paragraphs, so even a single
    matching paragraph among many is enough.
    """
    rows = _run_decon_one_shot(
        tmp_path,
        eval_records=[
            {"id": "eval_q", "text": "What is the speed of light in vacuum"},
        ],
        input_records=[
            {
                "id": "doc_buried",
                "partition_id": 0,
                "text": (
                    "Various unrelated physics notes go here.\n\n"
                    "What is the speed of light in vacuum\n\n"
                    "And here is some commentary after the question."
                ),
            },
        ],
        ngram=NGramConfig(ngram_length=4, overlap_threshold=0.5),
    )
    assert rows["doc_buried"]["contaminated"] is True
    assert rows["doc_buried"]["max_overlap"] == 1.0


def test_decon_catches_multi_paragraph_eval_against_single_paragraph_input(tmp_path: Path):
    """Eval spans multiple paragraphs; pretraining has same content inline (no newlines).

    Build adds ngrams from each eval paragraph independently. The pretraining
    paragraph's ngrams that fall inside one eval paragraph's span hit the bloom;
    boundary-spanning ngrams in pretraining don't (they were never in eval), but
    enough of them DO hit to clear the threshold.
    """
    rows = _run_decon_one_shot(
        tmp_path,
        eval_records=[
            {"id": "eval", "text": "What is the capital of France\nThe capital city is Paris"},
        ],
        input_records=[
            {
                "id": "doc_inline",
                "partition_id": 0,
                "text": "What is the capital of France The capital city is Paris",
            },
        ],
        ngram=NGramConfig(ngram_length=4, overlap_threshold=0.5),
    )
    assert rows["doc_inline"]["contaminated"] is True
    # 7 ngrams in input paragraph, 5 match (the cross-boundary 2 don't): 5/7 ≈ 0.71.
    assert rows["doc_inline"]["max_overlap"] >= 0.5


def test_decon_catches_near_verbatim_with_word_insertion(tmp_path: Path):
    """Pretraining has eval text with one extra word inserted; most ngrams still match."""
    rows = _run_decon_one_shot(
        tmp_path,
        eval_records=[
            {"id": "eval", "text": "Arctic predators have superior auditory capabilities for hunting beneath snow"},
        ],
        input_records=[
            {
                "id": "doc_inserted",
                "partition_id": 0,
                # extra word "thick" before "snow"
                "text": "Arctic predators have superior auditory capabilities for hunting beneath thick snow",
            },
        ],
        ngram=NGramConfig(ngram_length=4, overlap_threshold=0.5),
    )
    assert rows["doc_inserted"]["contaminated"] is True
    assert rows["doc_inserted"]["max_overlap"] >= 0.5


# ----- Known limitations (xfail with strict=True — tripwire if behavior improves) -----


@pytest.mark.xfail(
    reason="hashing is case-sensitive; eval and pretraining differing only in case do not match",
    strict=True,
)
def test_decon_misses_case_only_differences(tmp_path: Path):
    """Pretraining text identical to eval modulo case is NOT detected (limitation)."""
    rows = _run_decon_one_shot(
        tmp_path,
        eval_records=[
            {"id": "eval", "text": "lorem ipsum dolor sit amet consectetur"},
        ],
        input_records=[
            {
                "id": "doc_uppercase",
                "partition_id": 0,
                "text": "LOREM IPSUM DOLOR SIT AMET CONSECTETUR",
            },
        ],
        ngram=NGramConfig(ngram_length=4, overlap_threshold=0.5),
    )
    assert rows["doc_uppercase"]["contaminated"] is True


@pytest.mark.xfail(
    reason="punctuation is part of the token; eval with '?' vs pretraining without does not match",
    strict=True,
)
def test_decon_misses_punctuation_only_differences(tmp_path: Path):
    """Pretraining text identical to eval modulo trailing punctuation is NOT detected."""
    rows = _run_decon_one_shot(
        tmp_path,
        eval_records=[
            # tokens end with "?" — every ngram that touches the last token differs
            {"id": "eval", "text": "Who wrote the play Romeo and Juliet?"},
        ],
        input_records=[
            {
                "id": "doc_no_qmark",
                "partition_id": 0,
                "text": "Who wrote the play Romeo and Juliet",
            },
        ],
        # Use n=8 so EVERY ngram includes the last token and thus changes.
        ngram=NGramConfig(ngram_length=8, overlap_threshold=0.5),
    )
    assert rows["doc_no_qmark"]["contaminated"] is True


@pytest.mark.xfail(
    reason="short eval embedded in a long single paragraph dilutes the overlap fraction below threshold",
    strict=True,
)
def test_decon_misses_short_eval_diluted_in_long_paragraph(tmp_path: Path):
    """Eval is a short fragment; pretraining wraps it inside a long single paragraph.

    With n=4, the eval contributes ~1 ngram. The pretraining paragraph has many
    ngrams (the prefix + the eval ngram + the suffix). Score = 1/N → below 0.5.
    A length-decay or substring-aware scorer (cf. allenai/decon) would catch it.
    """
    rows = _run_decon_one_shot(
        tmp_path,
        eval_records=[
            # Eval is 4 tokens → exactly 1 ngram at n=4.
            {"id": "eval", "text": "atomic number of gold"},
        ],
        input_records=[
            {
                "id": "doc_buried",
                "partition_id": 0,
                # The eval ngram appears verbatim, surrounded by long context.
                "text": (
                    "Various trivia facts collected from many encyclopedic sources mention "
                    "the atomic number of gold among other periodic table chemistry topics "
                    "alongside copper silver and platinum which are also widely discussed"
                ),
            },
        ],
        ngram=NGramConfig(ngram_length=4, overlap_threshold=0.5),
    )
    assert rows["doc_buried"]["contaminated"] is True


@pytest.mark.xfail(
    reason="paraphrasing changes most tokens; ngram overlap drops below threshold",
    strict=True,
)
def test_decon_misses_paraphrased_eval(tmp_path: Path):
    """Pretraining expresses the same idea as eval with different words (no ngram overlap)."""
    rows = _run_decon_one_shot(
        tmp_path,
        eval_records=[
            {"id": "eval", "text": "What is the capital of France"},
        ],
        input_records=[
            {
                "id": "doc_paraphrased",
                "partition_id": 0,
                # Same question, different phrasing — no shared 4-grams.
                "text": "Which city serves as France's capital",
            },
        ],
        ngram=NGramConfig(ngram_length=4, overlap_threshold=0.5),
    )
    assert rows["doc_paraphrased"]["contaminated"] is True


@pytest.mark.xfail(
    reason="word order swap breaks every n-gram window; not detected by sliding-window ngram match",
    strict=True,
)
def test_decon_misses_word_order_permutation(tmp_path: Path):
    """Pretraining has the same words as eval in a permuted order; ngrams don't match."""
    rows = _run_decon_one_shot(
        tmp_path,
        eval_records=[
            {"id": "eval", "text": "alpha beta gamma delta epsilon zeta"},
        ],
        input_records=[
            {
                "id": "doc_permuted",
                "partition_id": 0,
                # Same six words, fully reversed.
                "text": "zeta epsilon delta gamma beta alpha",
            },
        ],
        ngram=NGramConfig(ngram_length=4, overlap_threshold=0.5),
    )
    assert rows["doc_permuted"]["contaminated"] is True


# --- prebuilt-bloom path (build_eval_bloom + merge_eval_blooms) ----------


def test_build_eval_bloom_then_decon_matches_inline(fox_corpus):
    """Building bloom separately + decon(prebuilt_bloom_dir=...) gives identical attrs to inline build."""
    # Path A: inline build.
    inline_output = Path(fox_corpus["output_dir"]) / "inline"
    decon_to_parquet(
        normalized_data=_as_source(Path(fox_corpus["input_dir"])),
        eval_data_sources=fox_corpus["eval_dir"],
        output_path=str(inline_output),
        ngram=NGramConfig(ngram_length=3, overlap_threshold=0.5),
        estimated_doc_count=10_000,
        false_positive_rate=1e-9,
    )
    inline_rows = _read_attributes(inline_output)

    # Path B: build_eval_bloom -> decon(prebuilt_bloom_dir=...).
    bloom_dir = Path(fox_corpus["output_dir"]) / "bloom"
    artifact = build_eval_bloom(
        eval_data_sources=fox_corpus["eval_dir"],
        output_path=str(bloom_dir),
        ngram=NGramConfig(ngram_length=3, overlap_threshold=0.5),
        estimated_doc_count=10_000,
        false_positive_rate=1e-9,
    )
    assert isinstance(artifact, EvalBloom)
    bp, ip = bloom_paths(str(bloom_dir))
    assert Path(bp).exists()
    assert Path(ip).exists()
    assert artifact.bloom_path == bp
    assert artifact.eval_hash_index_path == ip
    assert artifact.n_eval_records == 2  # fox_corpus has 2 eval records

    prebuilt_output = Path(fox_corpus["output_dir"]) / "prebuilt"
    decon_to_parquet(
        normalized_data=_as_source(Path(fox_corpus["input_dir"])),
        prebuilt_bloom_dir=str(bloom_dir),
        output_path=str(prebuilt_output),
        ngram=NGramConfig(ngram_length=3, overlap_threshold=0.5),
    )
    prebuilt_rows = _read_attributes(prebuilt_output)

    # Same set of ids, same contamination decisions, same overlap scores.
    assert set(inline_rows.keys()) == set(prebuilt_rows.keys())
    for doc_id, inline in inline_rows.items():
        pre = prebuilt_rows[doc_id]
        assert pre["contaminated"] == inline["contaminated"]
        assert pre["max_overlap"] == inline["max_overlap"]
        assert sorted(pre["matched_hashes"]) == sorted(inline["matched_hashes"])


def test_build_eval_bloom_excludes_named_task_dirs(tmp_path: Path):
    """``exclude_eval_dirs`` drops matching task dirs at read time.

    An already-materialized eval corpus can be pruned without regenerating it:
    the excluded task's records never enter the bloom or the hash index, while a
    kept task in the same tree still drives contamination matches.
    """
    eval_root = tmp_path / "evals"
    _write_eval_jsonl(
        eval_root / "lmh" / "kept_task" / "eval.jsonl.gz", [{"id": "kept-1", "text": "alpha beta gamma delta"}]
    )
    _write_eval_jsonl(
        eval_root / "lmh" / "code2text_python" / "eval.jsonl.gz",
        [{"id": "excl-1", "text": "epsilon zeta eta theta"}],
    )

    bloom_dir = tmp_path / "bloom"
    build_eval_bloom(
        eval_data_sources=str(eval_root),
        output_path=str(bloom_dir),
        ngram=NGramConfig(ngram_length=3, overlap_threshold=0.5),
        estimated_doc_count=1_000,
        false_positive_rate=1e-9,
        exclude_eval_dirs=frozenset({"code2text_python"}),
    )
    _, index_path = bloom_paths(str(bloom_dir))
    eval_ids = set(pq.read_table(index_path).column("eval_id").to_pylist())
    assert eval_ids == {"kept-1"}, "excluded task dir must not contribute to the hash index"

    # A doc matching the excluded eval text is NOT flagged; the kept one is.
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_input_parquet(
        input_dir / "part-00000-of-00001.parquet",
        [
            {"id": "hits-kept", "text": "alpha beta gamma delta", "partition_id": 0},
            {"id": "hits-excluded", "text": "epsilon zeta eta theta", "partition_id": 0},
        ],
    )
    decon_to_parquet(
        normalized_data=_as_source(input_dir),
        prebuilt_bloom_dir=str(bloom_dir),
        output_path=str(output_dir),
        ngram=NGramConfig(ngram_length=3, overlap_threshold=0.5),
    )
    rows = _read_attributes(output_dir)
    assert rows["hits-kept"]["contaminated"] is True
    assert rows["hits-excluded"]["contaminated"] is False


def test_merge_eval_blooms_equals_single_build(tmp_path: Path):
    """merge_eval_blooms over N per-eval builds detects everything a single combined build does.

    Stronger than 'identical bloom bytes' (bf.update may set the same bits in different order
    internally); we check the observable behavior end-to-end via decon.
    """
    eval_a_dir = tmp_path / "eval_a"
    eval_b_dir = tmp_path / "eval_b"
    _write_eval_jsonl(eval_a_dir / "eval.jsonl.gz", [{"id": "ea_1", "text": "alpha beta gamma delta epsilon"}])
    _write_eval_jsonl(eval_b_dir / "eval.jsonl.gz", [{"id": "eb_1", "text": "uno dos tres cuatro cinco"}])

    input_dir = tmp_path / "input"
    _write_input_parquet(
        input_dir / "part-00000-of-00001.parquet",
        [
            {"id": "doc_hits_a", "text": "alpha beta gamma delta epsilon", "partition_id": 0},
            {"id": "doc_hits_b", "text": "uno dos tres cuatro cinco", "partition_id": 0},
            {"id": "doc_unique", "text": "nothing in common with either eval", "partition_id": 0},
        ],
    )
    src = _as_source(input_dir)
    ngram = NGramConfig(ngram_length=3, overlap_threshold=0.5)

    # Combined-build baseline.
    baseline_out = tmp_path / "out_baseline"
    decon_to_parquet(
        normalized_data=src,
        eval_data_sources=[str(eval_a_dir), str(eval_b_dir)],
        output_path=str(baseline_out),
        ngram=ngram,
        estimated_doc_count=10_000,
        false_positive_rate=1e-9,
    )
    baseline_rows = _read_attributes(baseline_out)

    # Per-eval builds + merge.
    bloom_a_dir = tmp_path / "bloom_a"
    bloom_b_dir = tmp_path / "bloom_b"
    build_eval_bloom(
        eval_data_sources=str(eval_a_dir),
        output_path=str(bloom_a_dir),
        ngram=ngram,
        estimated_doc_count=10_000,
        false_positive_rate=1e-9,
    )
    build_eval_bloom(
        eval_data_sources=str(eval_b_dir),
        output_path=str(bloom_b_dir),
        ngram=ngram,
        estimated_doc_count=10_000,
        false_positive_rate=1e-9,
    )
    merged_dir = tmp_path / "bloom_merged"
    merge_eval_blooms(
        per_eval_bloom_dirs=[str(bloom_a_dir), str(bloom_b_dir)],
        output_path=str(merged_dir),
    )

    merged_out = tmp_path / "out_merged"
    decon_to_parquet(
        normalized_data=src,
        prebuilt_bloom_dir=str(merged_dir),
        output_path=str(merged_out),
        ngram=ngram,
    )
    merged_rows = _read_attributes(merged_out)

    # Same contamination decisions on every record.
    for doc_id in ("doc_hits_a", "doc_hits_b", "doc_unique"):
        assert baseline_rows[doc_id]["contaminated"] == merged_rows[doc_id]["contaminated"], doc_id
        assert baseline_rows[doc_id]["max_overlap"] == merged_rows[doc_id]["max_overlap"], doc_id

    # And the merged hash-index sidecar contains entries for BOTH per-eval sources.
    _, merged_index = bloom_paths(str(merged_dir))
    eval_ids = set(pq.read_table(str(merged_index)).column("eval_id").to_pylist())
    assert "ea_1" in eval_ids
    assert "eb_1" in eval_ids


def test_decon_to_parquet_requires_exactly_one_of_eval_or_prebuilt(fox_corpus):
    """Neither / both raise ValueError before any work is done."""
    src = _as_source(Path(fox_corpus["input_dir"]))
    out = fox_corpus["output_dir"]

    # neither
    with pytest.raises(ValueError, match="exactly one"):
        decon_to_parquet(normalized_data=src, output_path=out, ngram=NGramConfig(ngram_length=3))

    # both
    with pytest.raises(ValueError, match="exactly one"):
        decon_to_parquet(
            normalized_data=src,
            eval_data_sources=fox_corpus["eval_dir"],
            prebuilt_bloom_dir="/tmp/whatever",
            output_path=out,
            ngram=NGramConfig(ngram_length=3),
        )


def test_merge_eval_blooms_requires_non_empty(tmp_path: Path):
    with pytest.raises(ValueError):
        merge_eval_blooms(per_eval_bloom_dirs=[], output_path=str(tmp_path / "out"))


# --- cluster D: no-alphabetic-character ngram filter (marin#6852) ------------


def test_extract_ngrams_drops_letterless_ngrams():
    """A 13-gram with no alphabetic character is not emitted; one with a letter is.

    Pins the cluster-D filter: pure numeric sequences and punctuation runs carry
    no distinctive contamination signal but collide with number-list eval items.
    """
    numeric = "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
    assert list(_extract_ngrams(numeric, 13, 0)) == []
    punct = ", . ; : - / ( ) [ ] { } < >"
    assert list(_extract_ngrams(punct, 13, 0)) == []
    # A single letter anywhere in the window keeps it (it now has real content).
    mixed = "x 2 3 4 5 6 7 8 9 10 11 12 13"
    assert list(_extract_ngrams(mixed, 13, 0)) == [mixed]


def test_decon_skips_numeric_only_contamination(tmp_path: Path):
    """Cluster D: a numeric-list eval item does NOT flag a verbatim numeric-list
    corpus doc (no alphabetic 13-gram to key on), while a real textual overlap in
    the same run is still flagged — confirming the filter costs no text recall.
    """
    numbers = "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"
    rows = _run_decon_one_shot(
        tmp_path,
        eval_records=[
            {"id": "eval_numbers", "text": numbers},
            {"id": "eval_text", "text": text},
        ],
        input_records=[
            {"id": "doc_numbers", "partition_id": 0, "text": numbers},  # numeric-only → filtered → not flagged
            {"id": "doc_text", "partition_id": 0, "text": text},  # real overlap → still flagged
        ],
        ngram=NGramConfig(ngram_length=13, overlap_threshold=0.5),
    )
    assert rows["doc_numbers"]["contaminated"] is False
    assert rows["doc_numbers"]["max_overlap"] == 0.0
    assert rows["doc_numbers"]["matched_hashes"] == []
    assert rows["doc_text"]["contaminated"] is True
    assert rows["doc_text"]["max_overlap"] == 1.0


def test_merge_eval_blooms_rejects_size_mismatch(tmp_path: Path):
    """dupekit.Bloom.update requires identical sizing; size mismatch should raise."""
    eval_a = tmp_path / "eval_a"
    eval_b = tmp_path / "eval_b"
    _write_eval_jsonl(eval_a / "e.jsonl.gz", [{"id": "a", "text": "alpha beta gamma delta epsilon zeta eta theta"}])
    _write_eval_jsonl(eval_b / "e.jsonl.gz", [{"id": "b", "text": "uno dos tres cuatro cinco seis siete ocho"}])
    ngram = NGramConfig(ngram_length=3)

    build_eval_bloom(
        eval_data_sources=str(eval_a),
        output_path=str(tmp_path / "bloom_a"),
        ngram=ngram,
        estimated_doc_count=10_000,
        false_positive_rate=1e-9,
    )
    build_eval_bloom(
        eval_data_sources=str(eval_b),
        output_path=str(tmp_path / "bloom_b"),
        ngram=ngram,
        estimated_doc_count=100_000,  # different size -> dupekit will reject
        false_positive_rate=1e-9,
    )
    with pytest.raises(ValueError, match="size and max false positive rate"):
        merge_eval_blooms(
            per_eval_bloom_dirs=[str(tmp_path / "bloom_a"), str(tmp_path / "bloom_b")],
            output_path=str(tmp_path / "merged"),
        )


# ---------------------------------------------------------------------------
# Per-source common-ngram filter (marin#6852)
# ---------------------------------------------------------------------------


def test_paragraph_overlap_drop_hashes_excludes_from_both_sides():
    """``drop_hashes`` removes matched ngrams from numerator and denominator.

    An all-boilerplate paragraph (every ngram dropped) scores 0; a distinctive
    ngram left un-dropped still scores 1.0."""
    ngram = NGramConfig(ngram_length=4, overlap_threshold=0.5)
    para = "be it enacted by the assembled congress today"  # 8 tokens -> 5 four-grams
    grams = list(_extract_ngrams(para, 4, 0))
    bf = dupekit.Bloom(1000, 1e-9)
    for g in grams:
        bf.add(_bloom_hash(g))

    assert _paragraph_overlap_and_matches(para, bf, ngram)[0] == 1.0
    drop = frozenset(_bloom_hash(g) for g in grams)
    score, hits = _paragraph_overlap_and_matches(para, bf, ngram, drop)
    assert score == 0.0 and hits == []


def test_source_drop_set_filters_source_ubiquitous_ngram(tmp_path: Path):
    """An eval ngram present in ~every source doc lands in the drop-set and stops
    flagging boilerplate-only docs, while a distinctive eval match still flags."""
    eval_dir = tmp_path / "eval"
    _write_eval_jsonl(
        eval_dir / "eval.jsonl.gz",
        [
            {"id": "boiler", "text": "be it enacted by the assembled congress today"},
            {"id": "distinct", "text": "the platypus juggled seventeen luminous kumquats"},
        ],
    )
    bloom_dir = tmp_path / "bloom"
    build_eval_bloom(
        eval_data_sources=str(eval_dir),
        output_path=str(bloom_dir),
        ngram=NGramConfig(ngram_length=4, overlap_threshold=0.5),
        estimated_doc_count=10_000,
        false_positive_rate=1e-9,
    )

    # Source: 20 docs carrying only the boilerplate line + 1 genuine leak.
    input_dir = tmp_path / "input"
    docs = [
        {"id": f"d{i}", "text": "be it enacted by the assembled congress today", "partition_id": 0} for i in range(20)
    ]
    docs.append({"id": "leak", "text": "the platypus juggled seventeen luminous kumquats", "partition_id": 0})
    _write_input_parquet(input_dir / "part-00000-of-00001.parquet", docs)

    drop_dir = tmp_path / "drop"
    result = build_source_drop_set(
        df_sample_dir=str(input_dir),
        prebuilt_bloom_dir=str(bloom_dir),
        output_path=str(drop_dir),
        ngram=NGramConfig(ngram_length=4, paragraph_delimiter="\n"),
        sample_docs=1000,
        common_frac=0.5,
        common_min_abs=2,
    )
    assert result.n_dropped > 0  # boilerplate ngrams (df=20) dropped; distinctive (df=1) kept

    out_dir = tmp_path / "out"
    decon_to_parquet(
        normalized_data=_as_source(input_dir),
        prebuilt_bloom_dir=str(bloom_dir),
        output_path=str(out_dir),
        ngram=NGramConfig(ngram_length=4, overlap_threshold=0.5, paragraph_delimiter="\n"),
        drop_set_dir=str(drop_dir),
    )
    rows = _read_attributes(out_dir)
    assert rows["d0"]["contaminated"] is False  # boilerplate-only no longer flags
    assert rows["leak"]["contaminated"] is True  # distinctive leak still flags


def test_source_drop_set_empty_leaves_marks_unchanged(tmp_path: Path):
    """With no ubiquitous ngram, the drop-set is empty and marks are unaffected."""
    eval_dir = tmp_path / "eval"
    _write_eval_jsonl(
        eval_dir / "eval.jsonl.gz", [{"id": "e", "text": "the platypus juggled seventeen luminous kumquats"}]
    )
    bloom_dir = tmp_path / "bloom"
    build_eval_bloom(
        eval_data_sources=str(eval_dir),
        output_path=str(bloom_dir),
        ngram=NGramConfig(ngram_length=4, overlap_threshold=0.5),
        estimated_doc_count=10_000,
        false_positive_rate=1e-9,
    )
    input_dir = tmp_path / "input"
    _write_input_parquet(
        input_dir / "part-00000-of-00001.parquet",
        [{"id": "leak", "text": "the platypus juggled seventeen luminous kumquats", "partition_id": 0}],
    )
    drop_dir = tmp_path / "drop"
    result = build_source_drop_set(
        df_sample_dir=str(input_dir),
        prebuilt_bloom_dir=str(bloom_dir),
        output_path=str(drop_dir),
        ngram=NGramConfig(ngram_length=4, paragraph_delimiter="\n"),
        sample_docs=1000,
        common_frac=0.5,
        common_min_abs=5,
    )
    assert result.n_dropped == 0
    out_dir = tmp_path / "out"
    decon_to_parquet(
        normalized_data=_as_source(input_dir),
        prebuilt_bloom_dir=str(bloom_dir),
        output_path=str(out_dir),
        ngram=NGramConfig(ngram_length=4, overlap_threshold=0.5, paragraph_delimiter="\n"),
        drop_set_dir=str(drop_dir),
    )
    assert _read_attributes(out_dir)["leak"]["contaminated"] is True


def test_build_all_source_drop_sets_distributes_per_source(tmp_path: Path):
    """The distributed (zephyr) builder writes one drop.parquet per source: a
    source's ubiquitous ngram is dropped, another source's distinctive one kept."""
    eval_dir = tmp_path / "eval"
    _write_eval_jsonl(
        eval_dir / "eval.jsonl.gz",
        [
            {"id": "boiler", "text": "be it enacted by the assembled congress today"},
            {"id": "distinct", "text": "the platypus juggled seventeen luminous kumquats"},
        ],
    )
    bloom_dir = tmp_path / "bloom"
    build_eval_bloom(
        eval_data_sources=str(eval_dir),
        output_path=str(bloom_dir),
        ngram=NGramConfig(ngram_length=4, overlap_threshold=0.5),
        estimated_doc_count=10_000,
        false_positive_rate=1e-9,
    )
    a_dir = tmp_path / "srcA"
    _write_input_parquet(
        a_dir / "part-00000-of-00001.parquet",
        [{"id": f"a{i}", "text": "be it enacted by the assembled congress today", "partition_id": 0} for i in range(20)],
    )
    b_dir = tmp_path / "srcB"
    _write_input_parquet(
        b_dir / "part-00000-of-00001.parquet",
        [{"id": "b0", "text": "the platypus juggled seventeen luminous kumquats", "partition_id": 0}],
    )

    out = tmp_path / "drops"
    res = build_all_source_drop_sets(
        sources=[("srcA", str(a_dir)), ("srcB", str(b_dir))],
        prebuilt_bloom_dir=str(bloom_dir),
        output_path=str(out),
        ngram=NGramConfig(ngram_length=4, paragraph_delimiter="\n"),
        sample_docs=1000,
        common_frac=0.5,
        common_min_abs=2,
    )
    assert res.num_sources == 2
    assert len(_load_drop_set(str(out / "srcA"))) > 0  # boilerplate (df=20) dropped
    assert len(_load_drop_set(str(out / "srcB"))) == 0  # distinctive (df=1) kept


def test_decon_flagged_sample_sidecar(fox_corpus):
    """flagged_sample_size writes an `outputs/flagged_sample` sidecar of contaminated
    docs + text, so reports read O(sample) instead of rescanning the corpus."""
    decon_to_parquet(
        normalized_data=_as_source(Path(fox_corpus["input_dir"])),
        eval_data_sources=fox_corpus["eval_dir"],
        output_path=fox_corpus["output_dir"],
        ngram=NGramConfig(ngram_length=3, overlap_threshold=0.5),
        flagged_sample_size=10,
        estimated_doc_count=10_000,
        false_positive_rate=1e-9,
    )
    side = sorted(Path(fox_corpus["output_dir"]).glob("outputs/flagged_sample/*.parquet"))
    assert side, "expected an outputs/flagged_sample sidecar"
    rows = [r for f in side for r in pq.read_table(str(f)).to_pylist()]
    ids = {r["id"] for r in rows}
    assert "doc_arctic_exact" in ids and "doc_red_exact" in ids  # flagged docs captured
    assert "doc_unique" not in ids  # clean doc not sampled
    assert all(r["text"] and r["matched_hashes"] for r in rows)  # text + hashes present
