# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lexical scan artifacts are keyed by source and run; foreign directories refuse writes."""

import pytest

from experiments.post_training.task_curation.launch_lexical_scan import (
    artifact_paths,
    build_plan_text,
    plan_source_slug,
    refuse_foreign_artifacts,
)

ROOT = "s3://marin-us-east-02a/marin/rl-data/atlas/lexical/text-cohort"
PLAN_TEMPLATE = (
    "source_slug\t{slug}\n"
    "population_version\t2026.11.27.1\n"
    "populations_uri\ts3://example/populations.json\n"
    "report_uri\ts3://example/lexical-report.json\n"
)


def test_artifacts_are_keyed_by_source_and_run():
    paths = artifact_paths("crosscodeeval-typescript", "2026.11.27.2", root=ROOT)
    assert paths["artifact_dir"] == f"{ROOT}/lexical-scan/crosscodeeval-typescript/2026.11.27.2"
    assert paths["plan"] == f"{paths['artifact_dir']}/plan.tsv"
    assert paths["report"] == f"{paths['artifact_dir']}/lexical-report.json"
    other = artifact_paths("openswe-zero-task", "2026.11.27.1", root=ROOT)
    # the same population version under a different source never shares a directory
    assert other["artifact_dir"] != paths["artifact_dir"]


def test_artifact_paths_reject_path_traversing_segments():
    with pytest.raises(ValueError, match="single path segments"):
        artifact_paths("../shared", "2026.11.27.2", root=ROOT)
    with pytest.raises(ValueError, match="single path segments"):
        artifact_paths("crosscodeeval-typescript", "2026.11.27.2/report", root=ROOT)


def test_plan_carries_the_source_marker():
    paths = artifact_paths("crosscodeeval-typescript", "2026.11.27.2", root=ROOT)
    plan = build_plan_text(paths, "populations.json", "source-ids.json", "2026.11.27.2", "crosscodeeval-typescript")
    assert plan_source_slug(plan) == "crosscodeeval-typescript"


def test_refuses_a_foreign_source_directory():
    directory = artifact_paths("openswe-zero-task", "2026.11.27.1", root=ROOT)["artifact_dir"]
    foreign_plan = PLAN_TEMPLATE.format(slug="crosscodeeval-typescript")
    with pytest.raises(SystemExit, match="names source_slug"):
        refuse_foreign_artifacts({"plan.tsv", "lexical-report.json"}, foreign_plan, "openswe-zero-task", directory)
    with pytest.raises(SystemExit, match="names source_slug"):
        refuse_foreign_artifacts({"plan.tsv"}, foreign_plan, "openswe-zero-task", directory)


def test_refuses_an_unmarked_legacy_directory():
    directory = artifact_paths("openswe-zero-task", "2026.11.27.1", root=ROOT)["artifact_dir"]
    legacy_plan = "population_version\t2026.11.27.1\nreport_uri\ts3://example/lexical-report.json\n"
    with pytest.raises(SystemExit, match="predates the source keying"):
        refuse_foreign_artifacts({"plan.tsv", "lexical-report.json"}, legacy_plan, "openswe-zero-task", directory)
    with pytest.raises(SystemExit, match="predates the source keying"):
        refuse_foreign_artifacts({"plan.tsv"}, legacy_plan, "openswe-zero-task", directory)


def test_refuses_an_unattributed_report():
    directory = artifact_paths("crosscodeeval-typescript", "2026.11.27.3", root=ROOT)["artifact_dir"]
    with pytest.raises(SystemExit, match=r"holds a lexical-report\.json with no plan.tsv"):
        refuse_foreign_artifacts({"lexical-report.json"}, None, "crosscodeeval-typescript", directory)


def test_allows_fresh_and_own_directories():
    directory = artifact_paths("crosscodeeval-typescript", "2026.11.27.3", root=ROOT)["artifact_dir"]
    # an absent or empty directory
    refuse_foreign_artifacts(set(), None, "crosscodeeval-typescript", directory)
    # the source's own run (resume/overwrite within its own artifact version)
    own_plan = PLAN_TEMPLATE.format(slug="crosscodeeval-typescript")
    refuse_foreign_artifacts({"plan.tsv", "lexical-report.json"}, own_plan, "crosscodeeval-typescript", directory)


def test_marker_parsing_handles_the_missing_row():
    assert plan_source_slug("population_version\tv\n") is None
    assert plan_source_slug("source_slug\topenswe-zero-task\n") == "openswe-zero-task"
