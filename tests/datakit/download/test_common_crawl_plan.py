# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
from pathlib import Path

from marin.datakit.download.common_crawl_plan import (
    CommonCrawlFilter,
    CommonCrawlIndexKind,
    CommonCrawlPlanOptions,
    CommonCrawlSamplingMode,
    CommonCrawlSelection,
    CommonCrawlSource,
    SelectedCommonCrawlRecord,
    plan_common_crawl_manifest,
    plan_common_crawl_records,
    read_common_crawl_discovery,
    read_common_crawl_tasks,
    write_common_crawl_discovery,
    write_common_crawl_plan,
)
from marin.datakit.download.common_crawl_warc import main_record_from_index_row

RECORD_ID = "<urn:uuid:019f8700-d21d-78d8-8eb1-99eaa22579da>"


def _digest(value: str) -> str:
    digest = hashlib.sha1(value.encode(), usedforsecurity=False).digest()
    return f"sha1:{base64.b32encode(digest).decode().rstrip('=')}"


def _source(crawl_id: str, *, base_url: str = "https://data.commoncrawl.org") -> CommonCrawlSource:
    return CommonCrawlSource(
        crawl_id=crawl_id,
        index_kind=CommonCrawlIndexKind.MAIN,
        paths_manifest_url=f"https://index.commoncrawl.org/{crawl_id}.paths.gz",
        base_url=base_url,
    )


def _selected(
    source: CommonCrawlSource,
    warc_filename: str,
    offset: int,
    length: int,
) -> SelectedCommonCrawlRecord:
    indexed = main_record_from_index_row(
        {
            "url": f"https://example.com/{offset}.docx",
            "warc_filename": warc_filename,
            "warc_record_offset": offset,
            "warc_record_length": length,
            "warc_record_id": RECORD_ID,
            "content_digest": _digest(str(offset)),
        },
        crawl_id=source.crawl_id,
    )
    return SelectedCommonCrawlRecord(source, indexed, CommonCrawlSelection({"reason": "test"}))


def test_filter_selects_matching_rows_and_records_selection_signals() -> None:
    selector = CommonCrawlFilter(
        declared_mime_types=frozenset({"application/docx"}),
        detected_mime_types=frozenset({"application/docx"}),
        url_suffixes=(".docx",),
    )

    selection = selector.select(
        {
            "fetch_status": 200,
            "content_truncated": None,
            "content_mime_type": "application/docx; charset=binary",
            "content_mime_detected": "application/octet-stream",
            "url": "https://example.com/file.DOCX?download=1",
        }
    )

    assert selection == CommonCrawlSelection({"declared_mime": True, "detected_mime": False, "url_suffix": True})
    assert (
        selector.select(
            {
                "fetch_status": 404,
                "content_truncated": None,
                "content_mime_type": "application/docx",
                "url": "https://example.com/file.docx",
            }
        )
        is None
    )


def test_planner_coalesces_gap_boundary_and_keeps_sparse_record_singleton() -> None:
    source = _source("CC-MAIN-2026-30")
    records = [
        _selected(source, "a.warc.gz", 0, 100),
        _selected(source, "a.warc.gz", 150, 50),
        _selected(source, "a.warc.gz", 251, 25),
    ]

    tasks = plan_common_crawl_records(records, CommonCrawlPlanOptions(coalesce_gap_bytes=50, task_bytes=1_000))

    assert len(tasks) == 1
    assert [(selected.start, selected.stop, len(selected.records)) for selected in tasks[0].ranges] == [
        (0, 200, 2),
        (251, 276, 1),
    ]


def test_planner_never_coalesces_or_packs_across_sources() -> None:
    first = _source("CC-MAIN-2026-30")
    second = _source("CC-MAIN-2026-34", base_url="https://mirror.example")

    tasks = plan_common_crawl_records(
        [_selected(second, "same.warc.gz", 0, 10), _selected(first, "same.warc.gz", 0, 10)],
        CommonCrawlPlanOptions(coalesce_gap_bytes=1_000, task_bytes=1_000),
    )

    assert len(tasks) == 2
    assert {task.source.crawl_id for task in tasks} == {"CC-MAIN-2026-30", "CC-MAIN-2026-34"}
    assert all(len(task.ranges) == 1 and len(task.ranges[0].records) == 1 for task in tasks)


def test_per_source_sampling_is_stable_when_another_source_is_added() -> None:
    first = _source("CC-MAIN-2026-30")
    second = _source("CC-MAIN-2026-34")
    first_records = [_selected(first, "a.warc.gz", index * 100, 10) for index in range(10)]
    second_records = [_selected(second, "b.warc.gz", index * 100, 10) for index in range(10)]
    options = CommonCrawlPlanOptions(
        coalesce_gap_bytes=0,
        task_bytes=1_000,
        sampling_mode=CommonCrawlSamplingMode.PER_SOURCE_RANGE,
        sample_fraction=0.5,
        sample_seed=7,
    )

    first_only = plan_common_crawl_records(first_records, options)
    combined = plan_common_crawl_records([*second_records, *first_records], options)

    first_offsets = [selected.start for task in first_only for selected in task.ranges]
    combined_first_offsets = [selected.start for task in combined if task.source == first for selected in task.ranges]
    assert combined_first_offsets == first_offsets
    assert len(first_offsets) == 5


def test_plan_manifest_round_trip_preserves_tasks_and_summary(tmp_path: Path) -> None:
    source = _source("CC-MAIN-2026-30")
    records = [_selected(source, "a.warc.gz", 0, 100), _selected(source, "a.warc.gz", 125, 50)]
    tasks = plan_common_crawl_records(records, CommonCrawlPlanOptions(coalesce_gap_bytes=25, task_bytes=1_000))

    summary = write_common_crawl_plan(tasks, str(tmp_path))
    restored = read_common_crawl_tasks(summary.manifest_path)

    assert restored == tasks
    assert summary.num_sources == 1
    assert summary.num_warcs == 1
    assert summary.num_records == 2
    assert summary.num_ranges == 1
    assert summary.fetch_bytes == 175
    assert summary.num_tasks == 1


def test_discovery_manifest_can_be_replanned_without_index_scan(tmp_path: Path) -> None:
    source = _source("CC-MAIN-2026-30")
    records = [
        _selected(source, "a.warc.gz", 125, 50),
        _selected(source, "a.warc.gz", 0, 100),
    ]

    discovery = write_common_crawl_discovery(records, str(tmp_path / "discovery"))
    restored = list(read_common_crawl_discovery(discovery.manifest_path))
    plan = plan_common_crawl_manifest(
        discovery.manifest_path,
        str(tmp_path / "plan"),
        CommonCrawlPlanOptions(coalesce_gap_bytes=25, task_bytes=1_000),
    )

    assert [record.indexed_record.record_range.offset for record in restored] == [125, 0]
    assert [record.selection for record in restored] == [CommonCrawlSelection({"reason": "test"})] * 2
    assert plan.num_records == 2
    assert plan.num_ranges == 1
