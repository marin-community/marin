# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checks for reducer chart values, grouping, and untrusted labels."""

from dataclasses import replace
from datetime import datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path

import pytest
from zephyr.shuffle_report import render_shuffle_report
from zephyr.stats import ZephyrShuffleStat


class ReportParser(HTMLParser):
    def __init__(self, source: str):
        super().__init__()
        self.elements: list[tuple[str, dict[str, str]]] = []
        self.feed(source)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.elements.append((tag, {key: value for key, value in attrs if value is not None}))


@pytest.mark.parametrize("sizes,expected_bins", [([0, 1, 9, 90], [3, 0, 0, 0, 0, 0, 0, 0, 0, 1]), ([0, 0, 0, 0], [4])])
def test_report_preserves_target_values_and_histogram_counts(tmp_path: Path, sizes: list[int], expected_bins: list[int]):
    records = [
        ZephyrShuffleStat(
            execution_id="execution",
            stage_name="reduce",
            target_shard=target,
            num_targets=len(sizes),
            attempt=0,
            input_rows=size,
            payload_bytes=size * 10,
            num_sources=2,
            ts=datetime(2026, 9, 9),
            job_id="",
        )
        for target, size in enumerate(sizes)
    ]
    output = tmp_path / "report.html"
    render_shuffle_report(records, output)
    parsed = ReportParser(output.read_text())
    ranked = [attrs for tag, attrs in parsed.elements if tag == "tr" and "data-target" in attrs]
    assert [int(attrs["data-value"]) for attrs in ranked] == sorted(sizes, reverse=True) + sorted(
        [size * 10 for size in sizes], reverse=True
    )
    bins = [attrs for tag, attrs in parsed.elements if tag == "tr" and "data-count" in attrs]
    assert [int(attrs["data-count"]) for attrs in bins] == expected_bins * 2
    assert max(int(attrs["data-upper"]) for attrs in bins) == max(sizes) * 10
    summaries = [attrs for _, attrs in parsed.elements if "data-empty-targets" in attrs]
    assert int(summaries[0]["data-empty-targets"]) == sizes.count(0)


def test_report_separates_stages_and_escapes_labels(tmp_path: Path):
    label = '<script>alert("unsafe")</script>'
    records = [
        ZephyrShuffleStat(
            execution_id=label,
            stage_name=stage,
            target_shard=0,
            num_targets=1,
            attempt=0,
            input_rows=1,
            payload_bytes=10,
            num_sources=2,
            ts=datetime(2026, 9, 9),
            job_id=label,
        )
        for stage in ["reduce-a", "reduce-b"]
    ]
    output = tmp_path / "report.html"
    render_shuffle_report(records, output)
    parsed = ReportParser(output.read_text())
    sections = [attrs for tag, attrs in parsed.elements if tag == "section"]
    assert [attrs["data-stage"] for attrs in sections] == ["reduce-a", "reduce-b"]
    assert [attrs["data-execution"] for attrs in sections] == [label, label]
    assert not any(tag == "script" for tag, _ in parsed.elements)


def test_report_retry_observations_prefer_attempt_then_timestamp(tmp_path: Path):
    record = ZephyrShuffleStat(
        execution_id="execution",
        stage_name="reduce",
        target_shard=0,
        num_targets=1,
        attempt=1,
        input_rows=1,
        payload_bytes=10,
        num_sources=2,
        ts=datetime(2026, 9, 9),
        job_id="",
    )
    newer_attempt = replace(record, attempt=2, input_rows=2, payload_bytes=20)
    newer_timestamp = replace(newer_attempt, input_rows=3, payload_bytes=30, ts=record.ts + timedelta(seconds=1))
    delayed_old_attempt = replace(record, input_rows=99, payload_bytes=990, ts=record.ts + timedelta(seconds=2))
    output = tmp_path / "report.html"
    render_shuffle_report([newer_timestamp, newer_attempt, record, record, delayed_old_attempt], output)
    parsed = ReportParser(output.read_text())
    ranked = [attrs for tag, attrs in parsed.elements if tag == "tr" and "data-target" in attrs]
    assert [int(attrs["data-value"]) for attrs in ranked] == [3, 30]
    summary = next(attrs for _, attrs in parsed.elements if "data-targets" in attrs)
    assert summary["data-targets"] == "1"


def test_report_partial_coverage_keeps_unreported_targets_out_of_histograms(tmp_path: Path):
    empty = ZephyrShuffleStat(
        execution_id="execution",
        stage_name="reduce",
        target_shard=1,
        num_targets=4,
        attempt=0,
        input_rows=0,
        payload_bytes=0,
        num_sources=2,
        ts=datetime(2026, 9, 9),
        job_id="",
    )
    populated = replace(empty, target_shard=3, input_rows=25, payload_bytes=250)
    output = tmp_path / "report.html"
    render_shuffle_report([empty, populated], output)
    parsed = ReportParser(output.read_text())
    summary = next(attrs for _, attrs in parsed.elements if "data-targets" in attrs)
    assert summary["data-targets"] == "2"
    assert summary["data-expected-targets"] == "4"
    assert summary["data-unreported-targets"] == "2"
    assert summary["data-empty-targets"] == "1"
    assert summary["data-coverage"] == "partial"
    ranked = [attrs for tag, attrs in parsed.elements if tag == "tr" and "data-target" in attrs]
    assert [int(attrs["data-target"]) for attrs in ranked if "data-value" in attrs] == [3, 1, 3, 1]
    assert [int(attrs["data-target"]) for attrs in ranked if attrs.get("data-status") == "UNREPORTED"] == [0, 2, 0, 2]
    assert all("data-value" not in attrs for attrs in ranked if attrs.get("data-status") == "UNREPORTED")
    bins = [attrs for tag, attrs in parsed.elements if tag == "tr" and "data-count" in attrs]
    assert sum(int(attrs["data-count"]) for attrs in bins) == 4


def test_report_without_observations_has_unknown_coverage(tmp_path: Path):
    output = tmp_path / "report.html"
    render_shuffle_report([], output)
    parsed = ReportParser(output.read_text())
    assert any(attrs.get("data-coverage") == "unknown" for _, attrs in parsed.elements)
    assert not any("data-targets" in attrs or "data-metric" in attrs for _, attrs in parsed.elements)


def test_report_placeholder_only_stage_shows_all_ids_without_zero_measurements(tmp_path: Path):
    placeholder = ZephyrShuffleStat(
        execution_id="execution",
        stage_name="reduce",
        target_shard=1,
        num_targets=3,
        attempt=0,
        input_rows=None,
        payload_bytes=None,
        num_sources=None,
        ts=datetime(2026, 9, 9),
        job_id="",
    )
    output = tmp_path / "report.html"
    render_shuffle_report([placeholder], output)
    parsed = ReportParser(output.read_text())
    summary = next(attrs for _, attrs in parsed.elements if "data-targets" in attrs)
    assert summary["data-targets"] == "0"
    assert summary["data-expected-targets"] == "3"
    assert summary["data-unreported-targets"] == "3"
    assert summary["data-empty-targets"] == "0"
    assert summary["data-coverage"] == "partial"
    ranked = [attrs for tag, attrs in parsed.elements if tag == "tr" and "data-target" in attrs]
    assert [int(attrs["data-target"]) for attrs in ranked] == [0, 1, 2, 0, 1, 2]
    assert all(attrs["data-status"] == "UNREPORTED" and "data-value" not in attrs for attrs in ranked)
    assert not any("data-count" in attrs for _, attrs in parsed.elements)


@pytest.mark.parametrize("input_rows", [0, 25])
def test_report_delayed_placeholder_cannot_replace_measurement(tmp_path: Path, input_rows: int):
    measurement = ZephyrShuffleStat(
        execution_id="execution",
        stage_name="reduce",
        target_shard=0,
        num_targets=2,
        attempt=0,
        input_rows=input_rows,
        payload_bytes=input_rows * 10,
        num_sources=1 if input_rows else 0,
        ts=datetime(2026, 9, 9),
        job_id="",
    )
    delayed_placeholder = replace(
        measurement, input_rows=None, payload_bytes=None, num_sources=None, ts=measurement.ts + timedelta(seconds=1)
    )
    output = tmp_path / "report.html"
    render_shuffle_report([measurement, delayed_placeholder, replace(delayed_placeholder, target_shard=1)], output)
    parsed = ReportParser(output.read_text())
    summary = next(attrs for _, attrs in parsed.elements if "data-targets" in attrs)
    assert summary["data-targets"] == "1"
    assert summary["data-unreported-targets"] == "1"
    assert int(summary["data-empty-targets"]) == int(input_rows == 0)
    values = [int(attrs["data-value"]) for tag, attrs in parsed.elements if tag == "tr" and "data-value" in attrs]
    assert values == [input_rows, input_rows * 10]
