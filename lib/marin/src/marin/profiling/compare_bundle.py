# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-shot compare/track/report bundle orchestration for profile summaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from marin.profiling.query import compare_profile_summaries
from marin.profiling.report import build_markdown_report
from marin.profiling.schema import ProfileSummary
from marin.profiling.tracking import (
    RegressionThresholds,
    append_regression_record,
    assess_profile_regression,
    make_regression_record,
)


@dataclass(frozen=True)
class ComparisonBundleResult:
    """Paths and key metadata emitted by a compare bundle run."""

    output_dir: Path
    before_summary_path: Path
    after_summary_path: Path
    comparison_path: Path
    tracking_path: Path
    before_report_path: Path
    after_report_path: Path
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "before_summary_path": str(self.before_summary_path),
            "after_summary_path": str(self.after_summary_path),
            "comparison_path": str(self.comparison_path),
            "tracking_path": str(self.tracking_path),
            "before_report_path": str(self.before_report_path),
            "after_report_path": str(self.after_report_path),
            "status": self.status,
        }


def run_profile_comparison_bundle(
    *,
    before: ProfileSummary,
    after: ProfileSummary,
    output_dir: Path,
    thresholds: RegressionThresholds,
    top_k: int = 10,
    label: str | None = None,
    history_path: Path | None = None,
) -> ComparisonBundleResult:
    """
    Run a one-shot profile comparison bundle and write deterministic outputs.

    Emits:
    - `before.summary.json`
    - `after.summary.json`
    - `compare.json`
    - `track.json`
    - `before.report.md`
    - `after.report.md`
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    before_summary_path = output_dir / "before.summary.json"
    after_summary_path = output_dir / "after.summary.json"
    comparison_path = output_dir / "compare.json"
    tracking_path = output_dir / "track.json"
    before_report_path = output_dir / "before.report.md"
    after_report_path = output_dir / "after.report.md"

    before_summary_path.write_text(before.to_json() + "\n", encoding="utf-8")
    after_summary_path.write_text(after.to_json() + "\n", encoding="utf-8")

    comparison = compare_profile_summaries(before, after, top_k=top_k)
    comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    assessment = assess_profile_regression(before, after, thresholds=thresholds, top_k=top_k)
    record = make_regression_record(before=before, after=after, assessment=assessment, label=label)
    tracking_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if history_path is not None:
        append_regression_record(history_path, record)

    before_report_path.write_text(build_markdown_report(before, top_k=top_k), encoding="utf-8")
    after_report_path.write_text(build_markdown_report(after, top_k=top_k), encoding="utf-8")

    return ComparisonBundleResult(
        output_dir=output_dir,
        before_summary_path=before_summary_path,
        after_summary_path=after_summary_path,
        comparison_path=comparison_path,
        tracking_path=tracking_path,
        before_report_path=before_report_path,
        after_report_path=after_report_path,
        status=str(assessment.get("status", "unknown")),
    )
