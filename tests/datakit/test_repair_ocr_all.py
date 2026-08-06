# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for bringing the all-routes OCR corpus up to the current post-processing.

The corpus this reads was written before loop repair existed and is already boilerplate-stripped,
so the pass has exactly one job: cut repetition loops out of stored documents without disturbing
anything else. The cases here hold that boundary -- a document with no loop must come back byte
for byte, and a repaired one must stay internally consistent (offsets indexing its own text, a
content-hash ``id`` matching it, the repaired pages named).
"""

import contextlib

import marin.datakit.decon as decon_module
import pyarrow as pa
import pytest
from marin.datakit.decon import build_eval_bloom_step
from marin.datakit.normalize import NormalizedData, generate_id

from experiments.build_pdf_source import dedup
from experiments.build_pdf_source import repair_ocr_all as repair_module
from experiments.build_pdf_source.extract_ocr import _OUTPUT_SCHEMA
from experiments.build_pdf_source.loop_repair import LoopOptions
from experiments.build_pdf_source.repair_ocr_all import (
    decontaminate_steps,
    normalize_repaired_step,
    repair_batch,
    repair_document,
    repair_ocr_all_steps,
    repair_step,
    split_pages,
)
from experiments.datakit.decontam.prepare_eval_corpus import DECON_EXCLUDED_EVAL_TASKS

_OPTIONS = LoopOptions()


class _StopProbe(Exception):
    """Raised by the recording context to stop a step before it does real work."""


_PROSE = (
    "# Coastal erosion along the Holderness cliffs\n\n"
    "The Holderness coast retreats faster than any other shoreline in Europe, losing on average "
    "close to two metres of till each year to the North Sea. This section sets out the survey "
    "transects established between Bridlington and Spurn Point, the interval at which each was "
    "resurveyed, and the correction applied for tidal state at the time of measurement.\n\n"
)


def _page(text: str) -> str:
    """A stored page, which always carries the trailing newline ``record`` appends."""
    return text if text.endswith("\n") else text + "\n"


def _document(pages: list[str], **overrides) -> dict:
    """A stored record, assembled the way the extraction step assembled it."""
    text = "".join(pages)
    offsets, running = [], 0
    for page in pages:
        running += len(page)
        offsets.append(running)
    row = {
        "id": generate_id(text),
        "text": text,
        "source_id": "crawl-data/CC-MAIN-0001/warc/x.warc.gz:4096",
        "source": "common_crawl_focus_2026_22",
        "warc_filename": "crawl-data/CC-MAIN-0001/warc/x.warc.gz",
        "warc_record_offset": 4096,
        "content_digest": "sha1:ABCDEF",
        "url": "https://example.org/report.pdf",
        "num_pages": len(pages),
        "page_offsets": offsets,
        "extraction_status": "success",
        "extraction_error": None,
        "boilerplate_lines_removed": 3,
        "pages_ocred": len(pages),
        "pages_failed": 0,
        "pages_truncated": 0,
        "pages_unrendered": 0,
        "mean_render_dpi": 150.0,
        "pages_below_legibility_floor": 0,
        "completion_tokens": 900,
        "looped_pages": [],
        "loop_chars_dropped": 0,
    }
    return {**row, **overrides}


def test_split_pages_recovers_the_stored_pages_exactly():
    pages = [_page(_PROSE), _page("second page body"), _page("third")]
    document = _document(pages)
    assert split_pages(document["text"], document["page_offsets"]) == pages


def test_document_without_a_loop_is_returned_unchanged():
    document = _document([_page(_PROSE), _page("A short second page.")])
    assert repair_document(dict(document), _OPTIONS) == document


def test_looping_page_is_cut_back_and_the_document_stays_consistent():
    looped = _page(_PROSE + "| | | | |\n" * 400)
    clean = _page("A faithful second page of prose about tidal corrections.")
    document = _document([looped, clean])

    repaired = repair_document(dict(document), _OPTIONS)
    assert repaired is not None

    assert repaired["looped_pages"] == [1]
    assert repaired["loop_chars_dropped"] > 0
    assert repaired["extraction_status"] == "partial"
    # The loop is gone but the transcription in front of it, and the untouched page, are not.
    assert "| | | | |\n| | | | |" not in repaired["text"]
    assert "Holderness coast retreats" in repaired["text"]
    assert repaired["text"].endswith(clean)
    # The record still indexes itself: offsets end at the text length and id is its content hash.
    assert repaired["page_offsets"][-1] == len(repaired["text"])
    assert repaired["id"] == generate_id(repaired["text"])
    assert repaired["id"] != document["id"]


def test_repaired_pages_keep_their_separating_newline():
    """A cut page must not fuse with the next one; ``salvage`` right-strips, so it is re-added."""
    document = _document([_page(_PROSE + "0.00 0.00\n" * 400), _page("Next page opening line.")])
    repaired = repair_document(dict(document), _OPTIONS)
    assert repaired is not None
    pages = split_pages(repaired["text"], repaired["page_offsets"])
    assert pages[0].endswith("\n")
    assert pages[1] == "Next page opening line.\n"


def test_document_emptied_by_repair_is_dropped():
    """A page that is loop from its first line leaves nothing, and a document of those is not data."""
    document = _document([_page("| | | | |\n" * 400)])
    assert repair_document(dict(document), _OPTIONS) is None


def test_existing_extraction_error_is_kept_alongside_the_loop_clause():
    document = _document(
        [_page(_PROSE + "| | | | |\n" * 400)],
        extraction_status="partial",
        extraction_error="1 of 1 pages hit the token cap and were cut off",
        pages_truncated=1,
    )
    repaired = repair_document(dict(document), _OPTIONS)
    assert repaired is not None
    assert repaired["extraction_error"].startswith("1 of 1 pages hit the token cap")
    assert "repeated themselves and were cut back" in repaired["extraction_error"]


def test_repair_batch_adds_the_schema_2_loop_columns_to_legacy_records():
    """The stored corpus is schema 1: it has no loop columns, and the output schema requires them."""
    legacy = _document([_page(_PROSE), _page("Second page.")])
    del legacy["looped_pages"]
    del legacy["loop_chars_dropped"]

    records = list(repair_batch(pa.RecordBatch.from_pylist([legacy]), _OPTIONS))

    assert [record["looped_pages"] for record in records] == [[]]
    assert [record["loop_chars_dropped"] for record in records] == [0]
    # The result must satisfy the extraction schema the normalize step is given.
    assert pa.Table.from_pylist(records, schema=_OUTPUT_SCHEMA).num_rows == 1


def test_eval_bloom_matches_the_production_routes_so_it_is_reused():
    """The ~270 MB bloom is shared. Restating a decon parameter here would silently rebuild it.

    Step identity is name plus parameters, so this compares the bloom this module builds against
    the one ``dedup`` builds for the two production routes rather than pinning a hash.
    """
    production = build_eval_bloom_step(
        name="datakit/bloom/_combined_fixed",
        eval_data_sources=[dedup.EVAL_ROOT],
        ngram_length=dedup.NGRAM_LENGTH,
        overlap_threshold=dedup.OVERLAP_THRESHOLD,
        estimated_doc_count=dedup.ESTIMATED_DOC_COUNT,
        false_positive_rate=dedup.FALSE_POSITIVE_RATE,
        exclude_eval_dirs=DECON_EXCLUDED_EVAL_TASKS,
    )
    bloom, _, _ = decontaminate_steps(normalize_repaired_step(repair_step()))
    assert bloom.output_path == production.output_path


def test_pipeline_ends_in_a_decontaminated_dataset_and_runs_no_fuzzy_dedup():
    """Fuzzy dedup elects a canonical member, and the quality signal to elect it does not exist."""
    names = [step.name for step in repair_ocr_all_steps()]
    assert names[-1] == "data/datakit/clean/common_crawl_focus_2026_22_pdf_ocr_all"
    assert not [name for name in names if "minhash" in name or "fuzzy" in name]
    # Exact dedup has to precede the attribute join: consolidate joins by id, so a corpus still
    # holding repeated ids would drop every copy of a marked id, canonical included.
    assert names.index("data/datakit/normalize/common_crawl_focus_2026_22_pdf_ocr_all") < names.index(
        "data/datakit/decontam/common_crawl_focus_2026_22_pdf_ocr_all"
    )


@pytest.mark.parametrize("stage", ["decon", "drop_sets"])
def test_decontamination_stages_forward_their_coordinator_size(monkeypatch, stage):
    """Zephyr's 1 GB default coordinator OOM-kills (exit 137) every stage of this pipeline.

    It killed the repair stage at 1,771 of 1,773 tasks and decontamination at 22 of 23, both times
    after the stage's own output was already written -- so the symptom is a lost run rather than a
    lost shard, and it costs a full re-queue to rediscover.

    ``decon_step`` builds one of two lambdas depending on whether the bloom is prebuilt, and only
    the prebuilt branch runs here. A branch that accepts ``coordinator_resources`` and forgets to
    forward it is invisible from the step object, so this asserts on what the underlying function
    actually receives.
    """
    target = {"decon": "decon_to_parquet", "drop_sets": "build_all_source_drop_sets"}[stage]
    seen: dict = {}

    def record(**kwargs):
        seen.update(kwargs)
        raise _StopProbe

    monkeypatch.setattr(decon_module, target, record)
    # The decon lambda resolves its upstream artifact before calling through, and this test is
    # about argument forwarding rather than about any dataset existing.
    monkeypatch.setattr(
        decon_module,
        "read_artifact",
        lambda *_args, **_kwargs: NormalizedData(main_output_dir="memory://main", dup_output_dir="", counters={}),
    )

    _, drop_sets, decontam = decontaminate_steps(normalize_repaired_step(repair_step()))
    step = {"decon": decontam, "drop_sets": drop_sets}[stage]
    assert step.fn is not None
    with contextlib.suppress(_StopProbe):
        step.fn("memory://probe")

    assert seen, f"{target} was never called; the probe did not reach the stage"
    assert seen.get("coordinator_resources") is not None, (
        f"{stage} does not forward coordinator_resources, so it runs on Zephyr's 1 GB default "
        "and is OOM-killed once its work is already on disk"
    )


def test_consolidate_forwards_its_coordinator_size(monkeypatch):
    """The final filter is the one stage that cannot be probed against a half-built pipeline.

    It resolves both upstream artifacts before doing anything, so until decontamination has
    succeeded once there is nothing to read; that is exactly when a missing coordinator size would
    surface as another lost run.
    """
    seen: dict = {}

    def record(**kwargs):
        seen.update(kwargs)
        raise _StopProbe

    monkeypatch.setattr(repair_module, "consolidate", record)
    monkeypatch.setattr(
        repair_module,
        "read_artifact",
        lambda *_args, **_kwargs: NormalizedData(main_output_dir="memory://main", dup_output_dir="", counters={}),
    )

    with contextlib.suppress(_StopProbe):
        repair_module.consolidate_decontaminated(
            output_path="memory://probe",
            normalized_output_path="memory://normalized",
            decontam_output_path="memory://decontam",
        )

    assert seen, "consolidate was never called; the probe did not reach the stage"
    assert seen.get("coordinator_resources") is not None, (
        "the clean step does not forward coordinator_resources, so the final filter runs on " "Zephyr's 1 GB default"
    )


@pytest.mark.parametrize("pages", [1, 2, 7])
def test_page_offsets_partition_the_text_for_any_page_count(pages):
    document = _document([_page(f"Body of page {index}.") for index in range(pages)])
    repaired = repair_document(dict(document), _OPTIONS)
    assert repaired is not None
    assert "".join(split_pages(repaired["text"], repaired["page_offsets"])) == repaired["text"]
    assert len(repaired["page_offsets"]) == pages
