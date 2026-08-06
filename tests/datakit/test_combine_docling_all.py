# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for joining the two docling passes over the 10% sample into one corpus.

The two passes converted disjoint halves of the same sample with the same converter, so the union
has exactly one job: carry both sides through unchanged while recording which side each document
came from. The cases here hold that boundary -- a record must come back with only ``needs_ocr``
added, the tag must follow the shard rather than be guessed at, and the result must satisfy the
schema the normalize step downstream is given.
"""

import pyarrow as pa
import pytest
from marin.datakit.decon import build_eval_bloom_step
from marin.datakit.normalize import generate_id

from experiments.build_pdf_source import dedup
from experiments.build_pdf_source.combine_docling_all import (
    _COMBINED_SCHEMA,
    _SOURCE_FILE_COLUMN,
    combine_docling_all_steps,
    combine_step,
    decontaminate_steps,
    normalize_combined_step,
    tag_batch,
)
from experiments.build_pdf_source.extract_fleet import _FLEET_OUTPUT_SCHEMA
from experiments.datakit.decontam.prepare_eval_corpus import DECON_EXCLUDED_EVAL_TASKS

_TEXT_MAIN = "s3://bucket/marin/data/datakit/extract/focus_pdf_text_84cbb532/outputs/main/"
_OCR_MAIN = "s3://bucket/marin/data/datakit/extract/focus_pdf_docling_ocr_route_98f8b74a/outputs/main/"
_SHARD = "part-00000-of-01773.parquet"
_TEXT_SHARD = _TEXT_MAIN + _SHARD
_OCR_SHARD = _OCR_MAIN + _SHARD
_ROUTES = ((False, _TEXT_MAIN), (True, _OCR_MAIN))

_PROSE = (
    "# Coastal erosion along the Holderness cliffs\n\n"
    "The Holderness coast retreats faster than any other shoreline in Europe, losing on average "
    "close to two metres of till each year to the North Sea.\n"
)


def _document(text: str = _PROSE, **overrides) -> dict:
    """A stored docling record, assembled the way the conversion step assembled it."""
    row = {
        "id": generate_id(text),
        "text": text,
        "source_id": "crawl-data/CC-MAIN-0001/warc/x.warc.gz:4096",
        "source": "common_crawl_focus_2026_22",
        "warc_filename": "crawl-data/CC-MAIN-0001/warc/x.warc.gz",
        "warc_record_offset": 4096,
        "content_digest": "sha1:ABCDEF",
        "url": "https://example.org/report.pdf",
        "num_pages": 1,
        "page_offsets": [len(text)],
        "extraction_status": "success",
        "extraction_error": None,
        "boilerplate_lines_removed": 3,
        "layout_backend": "torch-heron",
    }
    return {**row, **overrides}


def _batch(shard: str, rows: list[dict]) -> pa.RecordBatch:
    """One row group as the reader hands it over, with the source path column injected."""
    schema = pa.schema([*_FLEET_OUTPUT_SCHEMA, pa.field(_SOURCE_FILE_COLUMN, pa.string(), nullable=False)])
    return pa.RecordBatch.from_pylist([{**row, _SOURCE_FILE_COLUMN: shard} for row in rows], schema=schema)


def test_combined_schema_is_the_conversion_schema_plus_the_router_decision():
    """Both passes wrote the same schema, so the union may only add the column that says which."""
    assert _COMBINED_SCHEMA.names == [*_FLEET_OUTPUT_SCHEMA.names, "needs_ocr"]
    assert _COMBINED_SCHEMA.field("needs_ocr").type == pa.bool_()


def test_records_pass_through_with_only_the_route_added():
    document = _document()
    records = list(tag_batch(_batch(_TEXT_SHARD, [document]), _ROUTES))
    assert records == [{**document, "needs_ocr": False}]


@pytest.mark.parametrize(("shard", "needs_ocr"), [(_TEXT_SHARD, False), (_OCR_SHARD, True)])
def test_each_pass_is_tagged_with_the_route_it_converted(shard, needs_ocr):
    records = list(tag_batch(_batch(shard, [_document()]), _ROUTES))
    assert [record["needs_ocr"] for record in records] == [needs_ocr]


def test_a_shard_under_neither_pass_is_an_error_rather_than_a_guess():
    """The tag comes from the driver's own listing; a stray path means that listing is wrong."""
    with pytest.raises(ValueError, match="belongs to neither docling pass"):
        list(tag_batch(_batch("s3://bucket/elsewhere/" + _SHARD, [_document()]), _ROUTES))


def test_an_empty_row_group_yields_nothing_rather_than_reaching_for_a_missing_path():
    assert list(tag_batch(_batch(_TEXT_SHARD, []), _ROUTES)) == []


def test_tagged_records_satisfy_the_schema_normalize_is_given():
    rows = [_document(), _document("A second converted document.\n")]
    records = [*tag_batch(_batch(_TEXT_SHARD, rows), _ROUTES), *tag_batch(_batch(_OCR_SHARD, rows), _ROUTES)]
    table = pa.Table.from_pylist(records, schema=_COMBINED_SCHEMA)
    assert table.num_rows == 4
    assert table.column("needs_ocr").to_pylist() == [False, False, True, True]


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
    bloom, _, _ = decontaminate_steps(normalize_combined_step(combine_step()))
    assert bloom.output_path == production.output_path


def test_pipeline_ends_in_a_decontaminated_dataset_and_runs_no_fuzzy_dedup():
    """Fuzzy dedup elects a canonical member, and the quality signal to elect it does not exist."""
    names = [step.name for step in combine_docling_all_steps()]
    assert names[-1] == "data/datakit/clean/common_crawl_focus_2026_22_pdf_docling_all"
    assert not [name for name in names if "minhash" in name or "fuzzy" in name]
    # Exact dedup has to precede the attribute join: consolidate joins by id, so a corpus still
    # holding repeated ids would drop every copy of a marked id, canonical included.
    assert names.index("data/datakit/normalize/common_crawl_focus_2026_22_pdf_docling_all") < names.index(
        "data/datakit/decontam/common_crawl_focus_2026_22_pdf_docling_all"
    )
