# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import NamedTuple
from unittest.mock import patch

import draccus
import jax
import numpy as np
import pytest
from fray.types import ResourceConfig
from marin.execution.lazy import materialized_config

from experiments.grug.moe_hero_ep.averaging import _DONORS
from experiments.grug.moe_hero_ep.format_eval import (
    AcquisitionCase,
    AcquisitionPanel,
    Candidate,
    CandidateEvaluation,
    GatePolicy,
    Interface,
    PilotBudget,
    Serialization,
)
from experiments.grug.moe_hero_ep.format_handoff import CandidateHandoffRequest, assemble_candidate
from experiments.grug.moe_hero_ep.format_pipeline import (
    FormatSftRecipe,
    LaunchEvidence,
    PilotContinuation,
    build_format_sft,
    validate_launch_evidence,
    verify_training_data,
)
from experiments.grug.moe_hero_ep.format_sample import native_evaluation_model
from experiments.grug.moe_hero_ep.format_suite import NativeSuiteRequest, run_suite
from experiments.grug.moe_hero_ep.format_test_support import smoke_gradient_report, smoke_optimizer_report, smoke_record
from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
from experiments.grug.moe_hero_ep.sft import BaseReference, SftConfig, sft_manifest
from experiments.grug.moe_hero_ep.sft_launch import (
    HeroSftCheckpoint,
    HeroSftHandoff,
    HeroSftInspection,
    PilotInspection,
    PilotStop,
    save_pilot_inspection,
    sft_measurement_identity,
)


class PreparedRecipe(NamedTuple):
    recipe: FormatSftRecipe
    manifest: dict
    acquisition: AcquisitionPanel


@pytest.fixture
def prepared_recipe(tmp_path_factory):
    # Training URLs enforce split names; a pytest function directory contains
    # '/test_' and would correctly trip the real holdout-leakage guard.
    raw = tmp_path_factory.mktemp("hero-format") / "raw"
    (raw / "train").mkdir(parents=True)
    sources = {}
    for slug, response, weight in (("first", "Hello", 1.0), ("second", "Goodbye", 3.0)):
        contents = (
            json.dumps({"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": response}]})
            + "\n"
        )
        (raw / "train" / f"{slug}.jsonl").write_text(contents)
        sources[slug] = {"weight": weight, "train": {"sha256": hashlib.sha256(contents.encode()).hexdigest()}}
    manifest: dict = {
        "tokenizer_revision": "a" * 40,
        "recipe": {"tokenizer_path": "marin-tokenizer"},
        "source_outputs": sources,
    }
    manifest["manifest_digest"] = digest(manifest)
    acquisition = AcquisitionPanel(
        source_artifact_digest=manifest["manifest_digest"],
        tokenizer_revision="a" * 40,
        cases=(AcquisitionCase(source_id="first", prefix_token_ids=(1, 2), target_token_id=3),),
    )
    recipe = FormatSftRecipe(
        name="checkpoints/hero-format-test",
        version="2026.10.19",
        reference=BaseReference("C", "c-digest", "A", 294000, "A", "marin-tokenizer", "a" * 40),
        model=GrugModelConfig(vocab_size=128256, max_seq_len=16384),
        sft=SftConfig(
            rank=16,
            alpha=16,
            token_ids=(128006, 128007, 128002, 128003, 128009),
            adapter_learning_rate=1e-4,
            vocab_learning_rate=1e-5,
            num_steps=20,
            warmup_steps=2,
            seed=0,
        ),
        data_manifest=str(raw / "manifest.json"),
        data_manifest_digest=manifest["manifest_digest"],
        data_root=str(raw),
        resources=ResourceConfig.with_gpu("GB200", count=4, replicas=16),
        batch_size=64,
        expert_axis_size=64,
        context_axis_size=1,
        replica_axis_size=1,
        max_processed_tokens=16000000,
        save_every_steps=10,
        maximum_metrics=(("loss", 10.0),),
        gate_record="measured-gates.json",
        acquisition_panel_path="acquisition.json",
        acquisition_panel_digest=acquisition.panel_digest,
    )
    return PreparedRecipe(recipe, manifest, acquisition)


def test_multisource_graph_isolates_chat_caches_and_preserves_raw_data(prepared_recipe, tmp_path):
    recipe, manifest, acquisition = prepared_recipe
    raw = Path(recipe.data_root)
    before = {path: path.read_bytes() for path in (raw / "train").iterdir()}
    verify_training_data(recipe, manifest)
    step = build_format_sft(recipe, manifest, acquisition)
    config = materialized_config(step, str(tmp_path / "artifacts"))
    caches = [materialized_config(cache, str(tmp_path / "artifacts")) for cache in step.deps]
    assert len({cache.cache_path for cache in caches}) == 2
    assert all(not cache.cache_path.startswith(recipe.data_root + "/") for cache in caches)
    assert {cache.train_paths[0] for cache in caches} == {
        str(raw / "train" / "first.jsonl"),
        str(raw / "train" / "second.jsonl"),
    }
    assert len({component.cache_dir for component in config.data.components.values()}) == 2
    assert config.data.auto_build_caches is False
    assert sorted(config.data.train_weights.values()) == [1.0, 3.0]
    assert step.artifact_type is HeroSftCheckpoint
    assert {path: path.read_bytes() for path in before} == before
    # The complete graph, including its acquisition callback, must fingerprint.
    first_identity = step.fingerprint()
    changed = dataclasses.replace(recipe, sft=dataclasses.replace(recipe.sft, rank=8))
    assert build_format_sft(changed, manifest, acquisition).fingerprint() != first_identity


def test_changed_training_bytes_are_rejected_before_training(prepared_recipe):
    recipe, manifest, _ = prepared_recipe
    verify_training_data(recipe, manifest)
    (Path(recipe.data_root) / "train" / "second.jsonl").write_text('{"messages": []}\n')
    with pytest.raises(ValueError, match="training bytes changed for source second"):
        verify_training_data(recipe, manifest)


def _prepared_evidence(recipe, root):
    model = draccus.encode(recipe.model)

    def prepared(name, provenance):
        path = root / name
        path.mkdir()
        metadata = {"is_temporary": False, "step": 294000, "averaging_digest": digest(provenance)}
        (path / "metadata.json").write_text(json.dumps(metadata))
        (path / "averaging.json").write_text(json.dumps(provenance))
        return {"path": str(path), "metadata_digest": digest(metadata)}

    source = {"path": "source-A", "metadata_digest": "permanent-A"}
    averaged = prepared(
        "B",
        {
            "candidate": "B",
            "model": model,
            "model_digest": digest(model),
            "request": {"checkpoints": [source]},
        },
    )
    child = prepared(
        "C",
        {
            "candidate": "C",
            "model": model,
            "model_digest": digest(model),
            "donors": list(_DONORS),
            "selected_parent": averaged,
            "source_pt_step": 294000,
            "tokenizer_pin": {"name": recipe.reference.tokenizer, "revision": recipe.reference.tokenizer_revision},
        },
    )
    recipe = dataclasses.replace(
        recipe,
        reference=dataclasses.replace(
            recipe.reference,
            checkpoint=child["path"],
            metadata_digest=child["metadata_digest"],
            source_pt_checkpoint=source["path"],
            selected_parent="B",
        ),
    )
    correct = {str(i): True for i in range(500)}
    baseline = CandidateEvaluation(
        candidate=Candidate.A,
        parent=None,
        checkpoint=source["path"],
        checkpoint_metadata_digest=source["metadata_digest"],
        model_config_digest=digest(draccus.encode(native_evaluation_model(recipe.model))),
        panel_id="fixed-panel",
        content_panel_id="fixed-content-panel",
        tokenizer_revision=recipe.reference.tokenizer_revision,
        backend="native",
        evaluation_dtype="bf16",
        heldout_loss_panel="heldout",
        heldout_loss=2,
        routing={"max_load": 1.1},
        content={s: correct for s in Serialization},
        format_report={},
    )
    parent = baseline.model_copy(
        update={
            "candidate": Candidate.B,
            "parent": Candidate.A,
            "checkpoint": averaged["path"],
            "checkpoint_metadata_digest": averaged["metadata_digest"],
        }
    )
    child_evaluation = baseline.model_copy(
        update={
            "candidate": Candidate.C,
            "parent": Candidate.B,
            "checkpoint": child["path"],
            "checkpoint_metadata_digest": child["metadata_digest"],
        }
    )
    policy = GatePolicy(
        baseline_evaluation_id=baseline.evaluation_id,
        baseline_variability_record="repeat-A.json",
        heldout_loss_increase=0.02,
        content_accuracy_decline=0.02,
        routing_maximums={"max_load": 1.2},
        enabled_interfaces=(Interface.SIMPLE,),
        observed_format_minimum=0.99,
        minimum_simple_cases=500,
        minimum_interface_cases=100,
    )
    return recipe, LaunchEvidence(
        baseline=baseline,
        parent=parent,
        prepared=child_evaluation,
        policy=policy,
        local_validation_reports=("local-tests.json",),
    )


def test_launch_gates_actual_b_ancestry_and_cumulative_regression(prepared_recipe, tmp_path):
    recipe, evidence = _prepared_evidence(prepared_recipe[0], tmp_path)
    assert validate_launch_evidence(recipe, evidence)["stage"] == "learning_smoke"
    degraded = evidence.model_copy(
        update={
            "parent": evidence.parent.model_copy(update={"heldout_loss": 10}),
            "prepared": evidence.prepared.model_copy(update={"heldout_loss": 10}),
        }
    )
    with pytest.raises(ValueError, match="Selected B has not passed"):
        validate_launch_evidence(recipe, degraded)
    cumulative = evidence.model_copy(
        update={
            "parent": evidence.parent.model_copy(update={"heldout_loss": 2.015}),
            "prepared": evidence.prepared.model_copy(update={"heldout_loss": 2.03}),
        }
    )
    with pytest.raises(ValueError, match="cumulative A regression"):
        validate_launch_evidence(recipe, cumulative)
    unrelated = evidence.model_copy(update={"parent": evidence.parent.model_copy(update={"checkpoint": "other-B"})})
    with pytest.raises(ValueError, match="actual ancestry"):
        validate_launch_evidence(recipe, unrelated)


def _pilot_evidence(recipe, evidence, root):
    original = dataclasses.replace(recipe.sft, num_steps=4, warmup_steps=1)
    identity = sft_measurement_identity(
        reference=recipe.reference,
        model=recipe.model,
        sft=original,
        data_manifest_digest=recipe.data_manifest_digest,
        acquisition_panel_digest=recipe.acquisition_panel_digest,
        batch_size=recipe.batch_size,
        expert_axis_size=recipe.expert_axis_size,
        context_axis_size=recipe.context_axis_size,
        replica_axis_size=recipe.replica_axis_size,
        resources=recipe.resources,
    )
    compact = root / "smoke-compact"
    compact.mkdir()
    counters = {"sft_step": 4, "processed_tokens": 200000, "supervised_tokens": 80000}
    metadata = {
        "is_temporary": False,
        "step": 4,
        "hero_sft": {"base": dataclasses.asdict(recipe.reference), "sft": dataclasses.asdict(original)},
        "hero_sft_counters": counters,
    }
    (compact / "metadata.json").write_text(json.dumps(metadata))
    parameter_bytes = 2048 + 32 * len(original.token_ids)
    optimizer_report = root / "smoke-compact-optimizer-report.json"
    optimizer_report.write_text(
        json.dumps(
            {
                **smoke_optimizer_report(parameter_bytes),
                "checkpoint_metadata_digest": recipe.reference.metadata_digest,
                "measurement_identity_digest": identity,
            }
        )
    )
    gradient_report = root / "smoke-gradient-structure-report.json"
    gradient_report.write_text(
        json.dumps(
            {
                **smoke_gradient_report(len(original.token_ids)),
                "checkpoint_metadata_digest": recipe.reference.metadata_digest,
                "measurement_identity_digest": identity,
            }
        )
    )
    smoke = smoke_record().model_copy(
        update={
            "checkpoint_metadata_digest": recipe.reference.metadata_digest,
            "measurement_identity_digest": identity,
            "sft_config": original,
            "trainable_parameter_bytes": parameter_bytes,
            "optimizer_state_bytes": 2 * parameter_bytes + 16,
            "terminal_checkpoint": str(compact),
            "terminal_checkpoint_metadata_digest": digest(metadata),
            "compact_optimizer_report": str(optimizer_report),
            "compact_optimizer_report_sha256": hashlib.sha256(optimizer_report.read_bytes()).hexdigest(),
            "gradient_structure_report": str(gradient_report),
            "gradient_structure_report_sha256": hashlib.sha256(gradient_report.read_bytes()).hexdigest(),
            "global_batch_executed_tokens": recipe.batch_size * recipe.model.max_seq_len,
            "updates": tuple(
                update.model_copy(
                    update={
                        "executed_tokens": recipe.batch_size * recipe.model.max_seq_len,
                    }
                )
                for update in smoke_record().updates
            ),
        }
    )
    budget = PilotBudget(
        format_panel_id="heldout-format-acceptance",
        inspection_non_padding_tokens=500000,
        cap_non_padding_tokens=1000000,
        minimum_learning_updates=5,
        early_inspection_update=3,
        warmup_updates=2,
        cooldown_updates=18,
        adapter_peak_lr=recipe.sft.adapter_learning_rate,
        vocab_peak_lr=recipe.sft.vocab_learning_rate,
        smoke_learning_review="reviewed-smoke.json",
        reuse_smoke_optimizer=True,
    )
    recipe = dataclasses.replace(
        recipe,
        max_processed_tokens=1000000,
        resume_checkpoint=str(compact),
        resume_sft_config=original,
    )
    return recipe, evidence.model_copy(update={"smoke": smoke, "pilot_budget": budget})


def test_pilot_binds_measurements_and_exact_resume_state(prepared_recipe, tmp_path):
    recipe, evidence = _prepared_evidence(prepared_recipe[0], tmp_path)
    recipe, evidence = _pilot_evidence(recipe, evidence, tmp_path)
    assert validate_launch_evidence(recipe, evidence)["stage"] == "pilot"
    changes = (
        {"batch_size": recipe.batch_size * 2},
        {"data_manifest_digest": "other-data"},
        {"acquisition_panel_digest": "other-diagnostic"},
        {"expert_axis_size": 32},
        {"resources": ResourceConfig.with_gpu("GB200", count=4, replicas=8)},
        {"sft": dataclasses.replace(recipe.sft, rank=8)},
    )
    for change in changes:
        with pytest.raises(ValueError, match="measurement identity"):
            validate_launch_evidence(dataclasses.replace(recipe, **change), evidence)
    with pytest.raises(ValueError, match="exact terminal compact"):
        validate_launch_evidence(dataclasses.replace(recipe, resume_checkpoint="other-smoke"), evidence)
    metadata_path = Path(recipe.resume_checkpoint) / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["hero_sft_counters"]["processed_tokens"] += 1
    metadata_path.write_text(json.dumps(metadata))
    # Even a repinned record cannot reconcile incorrect retained token counters.
    evidence = evidence.model_copy(
        update={
            "smoke": evidence.smoke.model_copy(
                update={
                    "terminal_checkpoint_metadata_digest": digest(metadata),
                }
            )
        }
    )
    with pytest.raises(ValueError, match="actual counters"):
        validate_launch_evidence(recipe, evidence)


@pytest.fixture
def paused_pilot(request, prepared_recipe, completed_suite, suite_inputs, tmp_path):
    # Reuse genuine retained suite bytes and rerun the suite at a paused D export.
    # Only model execution/storage and tokenizer acquisition are external fakes.
    handoff_request, prior_manifest = completed_suite
    suite = NativeSuiteRequest.model_validate(prior_manifest["request"])
    _, tokenizer = suite_inputs
    loss, format_cases = getattr(request, "param", (2.0, 1))
    original_format = suite.panels[0]
    formatting = original_format.model_copy(
        update={
            "panel": original_format.panel.model_copy(
                update={
                    "cases": tuple(
                        original_format.panel.cases[0].model_copy(update={"id": f"format-{i}"})
                        for i in range(format_cases)
                    ),
                }
            )
        }
    )
    recipe = dataclasses.replace(
        prepared_recipe[0],
        model=draccus.decode(GrugModelConfig, suite.model),
        reference=dataclasses.replace(prepared_recipe[0].reference, tokenizer=suite.tokenizer),
    )
    recipe, evidence = _prepared_evidence(recipe, tmp_path)
    recipe, evidence = _pilot_evidence(recipe, evidence, tmp_path)
    manifest = sft_manifest(recipe.reference, recipe.sft)
    compact, native = tmp_path / "paused-compact", tmp_path / "paused-export"
    compact.mkdir()
    native.mkdir()
    counters = {"sft_step": 4, "processed_tokens": 200000, "supervised_tokens": 80000}
    compact_metadata = {"is_temporary": False, "step": 4, "hero_sft": manifest, "hero_sft_counters": counters}
    native_metadata = {"is_temporary": False, "step": 4, "hero_sft": manifest, "merge_base": "BF16-rounded candidate C"}
    (compact / "metadata.json").write_text(json.dumps(compact_metadata))
    (native / "metadata.json").write_text(json.dumps(native_metadata))
    checkpoint = suite.checkpoint.model_copy(
        update={"uri": str(native), "metadata_digest": digest(native_metadata), "step": 4}
    )
    suite = suite.model_copy(update={"checkpoint": checkpoint, "panels": (formatting, *suite.panels[1:])})
    measured = json.loads((Path(handoff_request.suite_root) / "pt/report.json").read_text())["repeat_measurements"][0]
    measured["loss"] = loss
    measured["loss_sum"] = loss * measured["loss_weight"]
    measured["rows"][0].update(loss=loss, loss_sum=loss * measured["rows"][0]["loss_weight"])
    answer = tokenizer.encode("x")[0]

    def logits_backend(model, mesh):
        def logits(tokens, positions):
            scores = np.full((len(tokens), len(tokenizer)), -100.0)
            for row in range(len(tokens)):
                scores[row, 128009 if tokens[row, positions[row]] == answer else answer] = 0
            return scores

        return logits

    destination = tmp_path / "paused-suite"
    with (
        patch("experiments.grug.moe_hero_ep.format_suite.load_tokenizer", return_value=tokenizer),
        patch(
            "experiments.grug.moe_hero_ep.format_suite.restore_weights",
            side_effect=lambda uri, metadata_digest, config, mesh: Transformer.init(config, key=jax.random.key(0)),
        ),
        patch("experiments.grug.moe_hero_ep.format_suite.native_logits", side_effect=logits_backend),
        patch("experiments.grug.moe_hero_ep.format_suite.measure_pt_model", return_value=measured),
    ):
        suite_manifest = run_suite(suite, str(destination))
    candidate_request = CandidateHandoffRequest.model_validate(
        {
            **handoff_request.model_dump(),
            "candidate": Candidate.D,
            "parent": Candidate.C,
            "suite_root": str(destination),
            "suite_manifest_digest": suite_manifest["manifest_digest"],
            "checkpoint": checkpoint,
            "format_panel_id": formatting.panel.panel_id,
        }
    )
    candidate, _ = assemble_candidate(candidate_request)
    shared = {
        key: getattr(candidate, key)
        for key in (
            "model_config_digest",
            "panel_id",
            "content_panel_id",
            "tokenizer_revision",
            "backend",
            "evaluation_dtype",
            "heldout_loss_panel",
            "content",
            "routing",
        )
    }
    baseline = evidence.baseline.model_copy(update=shared)
    evidence = evidence.model_copy(
        update={
            "baseline": baseline,
            "parent": evidence.parent.model_copy(update=shared),
            "prepared": evidence.prepared.model_copy(update=shared),
            "policy": evidence.policy.model_copy(
                update={
                    "baseline_evaluation_id": baseline.evaluation_id,
                    "content_accuracy_decline": 1.0,
                    "routing_maximums": {"max_load_factor": 3.0},
                }
            ),
        }
    )
    evidence = evidence.model_copy(
        update={"pilot_budget": evidence.pilot_budget.model_copy(update={"format_panel_id": formatting.panel.panel_id})}
    )
    boundary = PilotInspection(
        evidence.pilot_budget.early_inspection_update,
        evidence.pilot_budget.inspection_non_padding_tokens,
        digest(evidence.policy.model_dump(mode="json")),
        digest(evidence.pilot_budget.model_dump(mode="json")),
    )
    identity = evidence.smoke.measurement_identity_digest
    record = HeroSftInspection(
        handoff=HeroSftHandoff(
            native_checkpoint=str(native),
            native_metadata_digest=digest(native_metadata),
            compact_checkpoint=str(compact),
            step=4,
            processed_tokens=200000,
            supervised_tokens=80000,
            base=recipe.reference,
        ),
        compact_metadata_digest=digest(compact_metadata),
        sft=recipe.sft,
        measurement_identity_digest=identity,
        inspection=boundary,
        max_processed_tokens=recipe.max_processed_tokens,
        stop_reason=PilotStop.EARLY_UPDATE,
    )
    output = tmp_path / "paused"
    output.mkdir()
    save_pilot_inspection(str(output), record)
    inspection_path = output / "inspection.json"
    continuation = PilotContinuation(
        inspection_path=str(inspection_path),
        inspection_digest=digest(json.loads(inspection_path.read_text())),
        candidate_handoff=candidate_request,
    )
    recipe = dataclasses.replace(recipe, resume_checkpoint=str(compact), resume_sft_config=recipe.sft)
    return recipe, evidence.model_copy(update={"continuation": continuation})


def test_pilot_continuation_recomputes_exact_paused_suite_and_preserves_original_boundaries(paused_pilot):
    recipe, evidence = paused_pilot
    result = validate_launch_evidence(recipe, evidence)
    assert result["pilot_inspection"]["reviewed_step"] == 4
    assert result["pilot_inspection"]["reviewed_tokens"] == 200000
    assert result["pilot_inspection"]["non_padding_tokens"] == evidence.pilot_budget.inspection_non_padding_tokens
    with pytest.raises(ValueError, match="exact terminal compact"):
        validate_launch_evidence(recipe, evidence.model_copy(update={"continuation": None}))
    with pytest.raises(ValueError, match="state, data, schedule"):
        validate_launch_evidence(dataclasses.replace(recipe, resume_sft_config=evidence.smoke.sft_config), evidence)
    changed = evidence.continuation.candidate_handoff.model_copy(
        update={
            "checkpoint": evidence.continuation.candidate_handoff.checkpoint.model_copy(
                update={"metadata_digest": "a" * 64}
            )
        }
    )
    with pytest.raises(ValueError, match="exact paused export"):
        validate_launch_evidence(
            recipe,
            evidence.model_copy(
                update={"continuation": evidence.continuation.model_copy(update={"candidate_handoff": changed})}
            ),
        )
    changed_budget = evidence.pilot_budget.model_copy(update={"smoke_learning_review": "other-review"})
    with pytest.raises(ValueError, match="original inspection budget"):
        validate_launch_evidence(recipe, evidence.model_copy(update={"pilot_budget": changed_budget}))
    wrong_panel = evidence.continuation.candidate_handoff.model_copy(update={"format_panel_id": "other-panel"})
    with pytest.raises(ValueError, match="exact paused export and panels"):
        validate_launch_evidence(
            recipe,
            evidence.model_copy(
                update={"continuation": evidence.continuation.model_copy(update={"candidate_handoff": wrong_panel})}
            ),
        )
    report = Path(evidence.continuation.candidate_handoff.suite_root) / "pt/report.json"
    report.write_text(report.read_text() + " ")
    with pytest.raises(ValueError, match="output bytes changed"):
        validate_launch_evidence(recipe, evidence)


@pytest.mark.parametrize("paused_pilot", [(3.0, 1)], indirect=True)
def test_pilot_continuation_rejects_measured_regression(paused_pilot):
    recipe, evidence = paused_pilot
    with pytest.raises(ValueError, match="diagnosis of native regression"):
        validate_launch_evidence(recipe, evidence)


@pytest.mark.parametrize("paused_pilot", [(2.0, 500)], indirect=True)
def test_pilot_continuation_stops_on_native_acceptance_with_serving_pending(paused_pilot):
    recipe, evidence = paused_pilot
    assert evidence.continuation.candidate_handoff.serving_parity_report is None
    with pytest.raises(ValueError, match="already pass; stop training"):
        validate_launch_evidence(recipe, evidence)
