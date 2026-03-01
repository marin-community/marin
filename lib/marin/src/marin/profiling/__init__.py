# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for ingesting and querying JAX/xprof profile artifacts."""

from marin.profiling.ingest import (
    DEFAULT_ARTIFACT_ALIAS,
    download_latest_profile_artifact_for_run,
    download_wandb_profile_artifact,
    summarize_profile_artifact,
    summarize_trace,
)
from marin.profiling.compare_bundle import ComparisonBundleResult, run_profile_comparison_bundle
from marin.profiling.publish import (
    PROFILE_SUMMARY_ARTIFACT_TYPE,
    publish_profile_summary_artifact,
)
from marin.profiling.query import compare_profile_summaries, query_profile_summary
from marin.profiling.report import build_markdown_report
from marin.profiling.schema import ProfileSummary, profile_summary_from_dict
from marin.profiling.tracking import (
    RegressionThresholds,
    assess_profile_regression,
    summarize_regression_history,
)

__all__ = [
    "DEFAULT_ARTIFACT_ALIAS",
    "PROFILE_SUMMARY_ARTIFACT_TYPE",
    "ComparisonBundleResult",
    "ProfileSummary",
    "RegressionThresholds",
    "assess_profile_regression",
    "build_markdown_report",
    "compare_profile_summaries",
    "download_latest_profile_artifact_for_run",
    "download_wandb_profile_artifact",
    "profile_summary_from_dict",
    "publish_profile_summary_artifact",
    "query_profile_summary",
    "run_profile_comparison_bundle",
    "summarize_profile_artifact",
    "summarize_regression_history",
    "summarize_trace",
]
