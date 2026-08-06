# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the PDF dedup + decontamination wiring (#7620).

The stages themselves (normalize, decon, consolidate) have their own suites; these tests pin the
contracts this pipeline leans on when it reuses them over the combined corpus.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.datakit.normalize import DedupMode, generate_id, normalize_to_parquet
from marin.execution.step_spec import StepSpec

from experiments.datakit.build_pdf_source.combine_routes import COMBINED_SCHEMA
from experiments.datakit.build_pdf_source.dedup import dedup_steps
from experiments.datakit.build_pdf_source.document_record import source_id


@pytest.fixture(autouse=True)
def local_fray():
    with set_current_client(LocalClient()):
        yield


def _record(text: str, offset: int) -> dict:
    """A combined-corpus record on the docling side: null OCR diagnostics, ``needs_ocr`` false."""
    return {
        "id": generate_id(text),
        "text": text,
        "source_id": source_id("crawl.warc.gz", offset),
        "source": "common_crawl_focus_2026_22_pdf",
        "warc_filename": "crawl.warc.gz",
        "warc_record_offset": offset,
        "content_digest": f"sha1:{offset}",
        "url": f"https://example.org/{offset}.pdf",
        "num_pages": 2,
        "page_offsets": [len(text) // 2, len(text)],
        "extraction_status": "success",
        "extraction_error": None,
        "boilerplate_lines_removed": 0,
        "needs_ocr": False,
        "pages_ocred": None,
        "pages_failed": None,
        "pages_truncated": None,
        "pages_unrendered": None,
        "mean_render_dpi": None,
        "pages_below_legibility_floor": None,
        "completion_tokens": None,
        "looped_pages": None,
        "loop_chars_dropped": None,
        "layout_backend": "torch_heron",
    }


def _run_normalize(combined_dir: Path, output_dir: Path):
    """Run the exact normalize pass the dedup step configures (see ``dedup._normalize_combined_step``)."""
    return normalize_to_parquet(
        input_path=str(combined_dir),
        output_path=str(output_dir),
        text_field="text",
        id_field="source_id",
        file_extensions=(".parquet",),
        dedup_mode=DedupMode.EXACT,
        output_schema=COMBINED_SCHEMA,
    )


def _write_shard(directory: Path, records: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records, schema=COMBINED_SCHEMA)
    pq.write_table(table, directory / "part-00000-of-00001.parquet")


def test_normalize_pass_splits_exact_duplicates_and_round_trips_the_record(tmp_path):
    """Extraction keeps byte-identical duplicates in place; this pass must split them out while
    leaving every other column -- ``source_id`` especially -- exactly as the combine wrote it."""
    first = _record("alpha document body", offset=100)
    duplicate = _record("alpha document body", offset=200)
    unique = _record("omega document body", offset=300)
    _write_shard(tmp_path / "combine", [first, duplicate, unique])

    result = _run_normalize(tmp_path / "combine", tmp_path / "normalized")

    main = pq.read_table(tmp_path / "normalized/outputs/main/part-00000-of-00001.parquet").to_pylist()
    dups = pq.read_table(tmp_path / "normalized/outputs/dups/part-00000-of-00001.parquet").to_pylist()
    assert result.counters["normalize/unique_records_out"] == 2
    assert sorted(record["text"] for record in main) == ["alpha document body", "omega document body"]
    # One copy of the duplicated text survives; the other lands in dups with its own provenance.
    (dup,) = dups
    kept_offsets = {record["warc_record_offset"] for record in main}
    assert dup["text"] == "alpha document body"
    assert {dup["warc_record_offset"], *kept_offsets} == {100, 200, 300}
    # The surviving records are the combined records, byte for byte.
    by_offset = {record["warc_record_offset"]: record for record in main}
    survivor = by_offset[300]
    assert survivor == unique


def test_normalize_pass_rekeys_the_id_when_it_caps_whitespace(tmp_path):
    """Whitespace-run capping is the one edit this pass may make; the id must follow the text."""
    text = "prologue" + " " * 400 + "epilogue"
    _write_shard(tmp_path / "combine", [_record(text, offset=100)])

    _run_normalize(tmp_path / "combine", tmp_path / "normalized")

    (record,) = pq.read_table(tmp_path / "normalized/outputs/main/part-00000-of-00001.parquet").to_pylist()
    assert record["text"] == "prologue" + " " * 128 + "epilogue"
    assert record["id"] == generate_id(record["text"])
    assert record["source_id"] == "crawl.warc.gz:100"


def test_dedup_dag_ends_in_a_decontaminated_dataset_and_runs_no_fuzzy_dedup(tmp_path, monkeypatch):
    """Fuzzy dedup elects a canonical member, and the quality signal to elect it does not exist
    yet at this point of the chain -- it lands in fuzzy_dedup, after the quality step."""
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    combined = StepSpec(name="data/datakit/combine/common_crawl_focus_2026_22_pdf", hash_attrs={}, fn=lambda p: None)
    names = [step.name for step in dedup_steps(combined, COMBINED_SCHEMA)]
    assert names[-1] == "data/datakit/clean/common_crawl_focus_2026_22_pdf"
    assert not [name for name in names if "minhash" in name or "fuzzy" in name]
    # Exact dedup has to precede the attribute join: consolidate joins by id, so a corpus still
    # holding repeated ids would drop every copy of a marked id, canonical included.
    assert names.index("data/datakit/normalize/common_crawl_focus_2026_22_pdf") < names.index(
        "data/datakit/decontam/common_crawl_focus_2026_22_pdf"
    )


def test_pipeline_dag_builds_without_worker_only_dependencies(tmp_path):
    """The entrypoint job syncs no extras, so building the full DAG must not import them.

    Runs in a subprocess because this process may already have pymupdf loaded for other tests.
    Also checks the invariant ``StepRunner`` needs: every dependency of every step is itself in
    the list the runner is given.
    """
    script = textwrap.dedent(
        """
        import sys

        class Blocker:
            blocked = {"pymupdf", "fitz", "docling", "docling_core"}

            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in self.blocked:
                    raise ImportError(f"worker-only dependency {name} imported at driver scope")
                return None

        sys.meta_path.insert(0, Blocker())

        from experiments.datakit.build_pdf_source.pipeline import build_pdf_source_steps

        steps = build_pdf_source_steps()
        names = [step.name for step in steps]
        assert len(set(names)) == len(names), f"duplicate step names: {names}"
        known = set(names)
        for step in steps:
            for dep in step.deps:
                assert dep.name in known, f"{step.name} depends on {dep.name}, absent from the runner list"
        """
    )
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env={"PATH": "/usr/bin:/bin", "MARIN_PREFIX": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
