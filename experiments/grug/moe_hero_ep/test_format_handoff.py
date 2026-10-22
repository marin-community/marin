# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest

from experiments.grug.moe_hero_ep.format_eval import Candidate, Serialization
from experiments.grug.moe_hero_ep.format_handoff import assemble_candidate
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest


def test_handoff_assembles_actual_suite_files_and_derived_pt_counts(completed_suite):
    request, manifest = completed_suite
    candidate, provenance = assemble_candidate(request)
    assert candidate.heldout_loss == 2
    assert candidate.routing == {
        "max_load_factor": 2.0,
        "max_balance_violation": 1.0,
        "routing_entropy_mean": 0.5,
        "router_z_loss": 0.2,
    }
    assert candidate.content[Serialization.MARIN] == {"content": True}
    assert candidate.content[Serialization.LEGACY] == {"content": False}
    assert candidate.format_report["metrics"]["simple"]["total"] == 1
    # Tiny A panels are retained; absolute500-case acceptance is enforced later.
    assert not candidate.serving_parity_passed and candidate.serving_parity_report is None
    assert provenance["suite_manifest"]["manifest_digest"] == manifest["manifest_digest"]
    assert provenance["evaluation_id"] == candidate.evaluation_id


@pytest.mark.parametrize("swapped", ("content-checkpoint", "pt-checkpoint", "pt-counts", "raw-bytes"))
def test_handoff_rejects_other_checkpoint_results_and_reconciles_counts(completed_suite, swapped):
    request, manifest = completed_suite
    root = Path(request.suite_root)
    if swapped == "raw-bytes":
        path = root / manifest["panels"][1]["result_key"] / "content.json"
        path.write_text(path.read_text() + " ")
    elif swapped == "content-checkpoint":
        output = manifest["panels"][1]
        path = root / output["result_key"] / "samples.json"
        samples = json.loads(path.read_text())
        samples["request"]["checkpoint"]["uri"] = "other-checkpoint"
        path.write_text(json.dumps(samples))
        output["file_sha256"]["samples.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        path = root / "pt/report.json"
        report = json.loads(path.read_text())
        if swapped == "pt-checkpoint":
            report["request"]["checkpoint"]["uri"] = "other-checkpoint"
        else:
            report["routing_counts_per_layer"] = [[8, 0, 0, 0]]
        report["report_digest"] = digest({key: value for key, value in report.items() if key != "report_digest"})
        path.write_text(json.dumps(report))
        manifest["pt"]["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["pt"]["report_digest"] = report["report_digest"]
    manifest["manifest_digest"] = digest({key: value for key, value in manifest.items() if key != "manifest_digest"})
    (root / "manifest.json").write_text(json.dumps(manifest))
    # Even an explicitly repinned transport bundle cannot mix checkpoint results.
    changed = request.model_copy(update={"suite_manifest_digest": manifest["manifest_digest"]})
    with pytest.raises(ValueError):
        assemble_candidate(changed)


def test_handoff_candidate_letters_require_actual_checkpoint_lineage(completed_suite):
    request, _ = completed_suite
    with pytest.raises(ValueError):
        assemble_candidate(request.model_copy(update={"candidate": Candidate.D, "parent": Candidate.C}))
