# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import math
import sys
from pathlib import Path


def test_ancestor_comparison_signs_ties_pending_and_source_links(tmp_path, monkeypatch):
    directory = Path(__file__).parent
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("campaign_ancestor_deltas", directory / "ancestor_deltas.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model, parent = "open-athena/Grug-Child", "open-athena/Grug-Parent"
    tracker = tmp_path / "TRACKER.md"
    tracker.write_text(
        "| Model | gain | loss | tie | waiting |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| Metric | accuracy | accuracy | accuracy | accuracy |\n"
        f"| {model} | 0.900 (s3://bucket/child/gain/results) | 0.200 (s3://bucket/child/loss/results) "
        "| 0.500 (s3://bucket/child/tie/results) | RUNNING |\n"
        f"| {parent} | 0.800 (s3://bucket/parent/gain/results) | 0.400 (s3://bucket/parent/loss/results) "
        "| 0.500 (s3://bucket/parent/tie/results) | 0.700 (s3://bucket/parent/waiting/results) |\n"
    )
    lineage = module.Lineage(
        model, "Child", "child", (module.Parent(parent, "Parent"),), "https://example.com/provenance"
    )
    rows = module.comparisons(tracker, (lineage,))
    assert len(rows) == 4
    assert math.isclose(rows[0].delta_pp, 10, abs_tol=1e-12)
    assert math.isclose(rows[1].delta_pp, -20, abs_tol=1e-12)
    assert rows[2].delta_pp == 0
    assert rows[3].delta_pp is None and rows[3].model_score is None
    assert rows[3].parent_score == 0.7
    assert rows[0].model_result == "s3://bucket/child/gain/results"
    assert rows[3].parent_result == "s3://bucket/parent/waiting/results"
