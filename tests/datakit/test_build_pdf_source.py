# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the focus-crawl PDF fetch plan and the range reader that executes it."""

import io

import numpy as np
import pytest

from experiments.datakit.build_pdf_source.common import RangeFetch
from experiments.datakit.build_pdf_source.fetch import iter_planned_pdfs, read_fetch_tasks
from experiments.datakit.build_pdf_source.plan import IndexScan, coalesce_ranges, pack_tasks, sample_ranges, write_plan

_WARCS = ["crawl/warc/a.warc.gz", "crawl/warc/b.warc.gz"]


def _scan(records: list[tuple[int, int, int]], warc_filenames: list[str] | None = None) -> IndexScan:
    """Build an IndexScan from ``(warc_id, offset, length)`` triples."""
    warc_ids, offsets, lengths = (np.array(column, dtype=np.int64) for column in zip(*records, strict=True))
    return IndexScan(
        warc_filenames=warc_filenames or _WARCS,
        warc_ids=warc_ids.astype(np.int32),
        offsets=offsets,
        lengths=lengths,
    )


def _build_warc(specs: list[tuple[str, bytes, str]]) -> tuple[bytes, list[int]]:
    """Write a gzip-member WARC of HTTP responses; return its bytes and each record's offset."""
    pytest.importorskip("warcio")
    from warcio.statusandheaders import StatusAndHeaders  # noqa: PLC0415
    from warcio.warcwriter import WARCWriter  # noqa: PLC0415

    buffer = io.BytesIO()
    writer = WARCWriter(buffer, gzip=True)
    offsets = []
    for uri, body, content_type in specs:
        offsets.append(buffer.tell())
        writer.write_record(
            writer.create_warc_record(
                uri,
                "response",
                payload=io.BytesIO(body),
                length=len(body),
                http_headers=StatusAndHeaders(
                    "200 OK",
                    [("Content-Type", content_type), ("Content-Length", str(len(body)))],
                    protocol="HTTP/1.1",
                ),
                warc_headers_dict={"WARC-Identified-Payload-Type": content_type},
            )
        )
    return buffer.getvalue(), offsets


def test_coalesce_merges_records_within_the_gap_and_splits_beyond_it():
    # Records at 0-100 and 150-250 sit 50 bytes apart; the third starts 500 bytes after the second.
    scan = _scan([(0, 0, 100), (0, 150, 100), (0, 750, 100)])

    ranges = coalesce_ranges(scan, gap_bytes=100)

    assert [(r.start, r.stop) for r in ranges] == [(0, 250), (750, 850)]
    assert [r.record_offsets for r in ranges] == [(0, 150), (750,)]


def test_coalesce_never_spans_warc_files():
    # Adjacent offsets, but different WARCs: a single range would address the wrong bytes.
    scan = _scan([(0, 0, 100), (1, 100, 100)])

    ranges = coalesce_ranges(scan, gap_bytes=1 << 20)

    assert [r.warc_filename for r in ranges] == _WARCS
    assert [(r.start, r.stop) for r in ranges] == [(0, 100), (100, 200)]


def test_coalesce_is_independent_of_index_row_order():
    """The index parts interleave arbitrarily, so the plan must not depend on arrival order."""
    records = [(0, 0, 100), (0, 150, 100), (1, 40, 100), (1, 900, 100), (0, 900, 100)]
    ordered = coalesce_ranges(_scan(records), gap_bytes=100)

    shuffled = list(np.random.default_rng(0).permutation(len(records)))
    reordered = coalesce_ranges(_scan([records[i] for i in shuffled]), gap_bytes=100)

    assert reordered == ordered


def test_coalesce_keeps_every_input_record_exactly_once():
    scan = _scan([(0, 0, 10), (0, 40, 10), (0, 5_000, 10), (1, 0, 10), (1, 12, 10)])

    ranges = coalesce_ranges(scan, gap_bytes=100)

    assert sorted(offset for r in ranges for offset in r.record_offsets) == [0, 0, 12, 40, 5_000]


def test_sample_ranges_is_reproducible_under_a_seed_and_keeps_plan_order():
    ranges = [RangeFetch(_WARCS[0], i * 100, i * 100 + 50, (i * 100,)) for i in range(100)]

    first = sample_ranges(ranges, fraction=0.2, seed=7)
    again = sample_ranges(ranges, fraction=0.2, seed=7)
    other = sample_ranges(ranges, fraction=0.2, seed=8)

    assert len(first) == 20
    assert first == again
    assert first != other
    assert [r.start for r in first] == sorted(r.start for r in first)


def test_sample_ranges_at_full_fraction_keeps_everything():
    ranges = [RangeFetch(_WARCS[0], i * 100, i * 100 + 50, (i * 100,)) for i in range(10)]

    assert sample_ranges(ranges, fraction=1.0, seed=1) == ranges


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_sample_ranges_rejects_a_fraction_outside_the_unit_interval(fraction):
    with pytest.raises(ValueError):
        sample_ranges([], fraction=fraction, seed=1)


def test_pack_tasks_respects_the_budget_and_preserves_every_range():
    ranges = [RangeFetch(_WARCS[0], i * 1_000, i * 1_000 + 300, (i * 1_000,)) for i in range(10)]

    tasks = pack_tasks(ranges, task_bytes=1_000)

    assert [task.task_id for task in tasks] == list(range(len(tasks)))
    assert all(task.size <= 1_000 for task in tasks)
    assert [selected for task in tasks for selected in task.ranges] == ranges


def test_pack_tasks_gives_an_oversized_range_its_own_task():
    ranges = [
        RangeFetch(_WARCS[0], 0, 100, (0,)),
        RangeFetch(_WARCS[0], 1_000, 6_000, (1_000,)),
        RangeFetch(_WARCS[0], 8_000, 8_100, (8_000,)),
    ]

    tasks = pack_tasks(ranges, task_bytes=1_000)

    assert [len(task.ranges) for task in tasks] == [1, 1, 1]
    assert tasks[1].size == 5_000


def test_plan_parquet_round_trips_into_the_same_fetch_tasks(tmp_path):
    """The plan file is the entire contract between the plan step and the fetch step."""
    ranges = [
        RangeFetch(_WARCS[0], 0, 400, (0, 150, 300)),
        RangeFetch(_WARCS[0], 9_000, 9_100, (9_000,)),
        RangeFetch(_WARCS[1], 12, 512, (12, 400)),
    ]
    tasks = pack_tasks(ranges, task_bytes=500)
    plan_path = str(tmp_path / "plan.parquet")

    write_plan(tasks, plan_path)

    assert read_fetch_tasks(plan_path) == tasks


def test_iter_planned_pdfs_selects_records_by_absolute_offset():
    """A coalesced range spans unplanned records; only the planned offsets may be emitted."""
    warc, offsets = _build_warc(
        [
            ("https://a.test/1.pdf", b"%PDF-1.4 one", "application/pdf"),
            ("https://a.test/page.html", b"<html>gap</html>", "text/html"),
            ("https://a.test/2.pdf", b"%PDF-1.4 two", "application/pdf"),
            ("https://a.test/excluded.pdf", b"%PDF-1.4 truncated", "application/pdf"),
        ]
    )
    # Ranges start mid-WARC in production, so record offsets must be reported absolutely.
    base = 4096
    selected = RangeFetch(_WARCS[0], base, base + len(warc), (base + offsets[0], base + offsets[2]))

    rows = list(iter_planned_pdfs(io.BytesIO(warc), selected))

    assert [row["url"] for row in rows] == ["https://a.test/1.pdf", "https://a.test/2.pdf"]
    assert [row["pdf"] for row in rows] == [b"%PDF-1.4 one", b"%PDF-1.4 two"]
    assert [row["warc_record_offset"] for row in rows] == [base + offsets[0], base + offsets[2]]
    assert [row["warc_filename"] for row in rows] == [_WARCS[0], _WARCS[0]]
    assert all(row["content_digest"].startswith("sha1:") for row in rows)


def test_iter_planned_pdfs_reports_planned_records_missing_from_the_range():
    warc, offsets = _build_warc([("https://a.test/1.pdf", b"%PDF-1.4 one", "application/pdf")])
    selected = RangeFetch(_WARCS[0], 0, len(warc), (offsets[0], 999_999))

    with pytest.raises(RuntimeError, match="999999"):
        list(iter_planned_pdfs(io.BytesIO(warc), selected))
