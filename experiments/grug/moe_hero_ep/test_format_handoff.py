# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import jax
import numpy as np
import pytest

from experiments.grug.moe_hero_ep.format_eval import Candidate, Serialization
from experiments.grug.moe_hero_ep.format_handoff import CandidateHandoffRequest, assemble_candidate
from experiments.grug.moe_hero_ep.format_suite import PanelPurpose, SuitePanel, run_suite
from experiments.grug.moe_hero_ep.model import Transformer
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest

pytest_plugins = ["experiments.grug.moe_hero_ep.test_format_suite"]


@pytest.fixture
def completed_suite(tmp_path, suite_inputs):
    request, tokenizer = suite_inputs
    checkpoint = tmp_path / "A"
    checkpoint.mkdir()
    metadata = {"is_temporary": False, "step": 294000}
    (checkpoint / "metadata.json").write_text(json.dumps(metadata))
    reference = request.checkpoint.model_copy(update={"uri": str(checkpoint), "metadata_digest": digest(metadata)})
    content = request.panels[1]
    request = request.model_copy(
        update={
            "checkpoint": reference,
            "panels": (
                *request.panels,
                SuitePanel(panel=content.panel, serialization=Serialization.LEGACY, purpose=PanelPurpose.CONTENT),
            ),
        }
    )
    answer = tokenizer.encode("x")[0]

    def logits_backend(model, mesh):
        def logits(tokens, positions):
            scores = np.full((len(tokens), len(tokenizer)), -100.0)
            for row in range(len(tokens)):
                stop = 128009 if tokens[row, positions[row]] == answer else answer
                # Legacy does not stop on EOT: emitting it still supplies a wrong,
                # fully retained answer rather than fabricating an exact success.
                scores[row, stop] = 0
            return scores

        return logits

    def pt_backend(model, panel, mesh):
        return {
            "rows": [
                {
                    "row_index": 0,
                    "arrays_sha256": panel.rows[0].arrays_sha256,
                    "loss_sum": 5.0,
                    "loss_weight": 2.5,
                    "loss": 2.0,
                    "valid_tokens": 4,
                }
            ],
            "routing_batches": [
                {
                    "row_indices": [0],
                    "filler_rows": 0,
                    "valid_tokens": 4,
                    "routing_counts_per_layer": [[4, 4, 0, 0]],
                    "routing_entropy_mean": 0.5,
                    "router_z_loss": 0.2,
                }
            ],
            "batch_size": 1,
            "forward_passes": 1,
            "loss_sum": 5.0,
            "loss_weight": 2.5,
            "loss": 2.0,
            "valid_tokens": 4,
            "routing_counts_per_layer": [[4, 4, 0, 0]],
            "routing_entropy_mean": 0.5,
            "router_z_loss": 0.2,
            "routing_policy": "all non-padding inputs",
            "routing_aggregation": "valid-token-weighted means",
        }

    destination = tmp_path / "suite"
    with (
        patch("experiments.grug.moe_hero_ep.format_suite.load_tokenizer", return_value=tokenizer),
        patch(
            "experiments.grug.moe_hero_ep.format_suite.restore_weights",
            side_effect=lambda uri, metadata_digest, config, mesh: Transformer.init(config, key=jax.random.key(0)),
        ),
        patch("experiments.grug.moe_hero_ep.format_suite.native_logits", side_effect=logits_backend),
        patch("experiments.grug.moe_hero_ep.format_suite.measure_pt_model", side_effect=pt_backend),
    ):
        manifest = run_suite(request, str(destination))
    handoff = CandidateHandoffRequest(
        candidate=Candidate.A,
        parent=None,
        suite_root=str(destination),
        suite_manifest_digest=manifest["manifest_digest"],
        checkpoint=reference,
        model_config_digest=manifest["model_digest"],
        tokenizer=request.tokenizer,
        tokenizer_revision=request.tokenizer_revision,
        format_panel_id=request.panels[0].panel.panel_id,
        content_panel_id=content.panel.panel_id,
        pt_panel_path=request.pt_panel_path,
        pt_panel_sha256=request.pt_panel_sha256,
    )
    return handoff, manifest


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
