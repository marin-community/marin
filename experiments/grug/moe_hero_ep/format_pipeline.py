# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind prepared Hero weights and pinned chat data to the shared SFT artifact graph.

The default CLI only writes a reviewable launch configuration. ``--run`` is an
explicit training submission and must follow the recorded experiment gates.
"""

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import draccus
from fray.types import ResourceConfig
from marin.execution.lazy import ArtifactStep, StepContext, lower, materialized_config
from marin.execution.step_runner import StepRunner
from pydantic import Field
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.storage_path import prefix_join

from experiments.grug.moe_hero_ep.averaging import _DONORS, read_prepared_provenance
from experiments.grug.moe_hero_ep.format_eval import (
    AcquisitionDiagnostic,
    AcquisitionPanel,
    Candidate,
    CandidateEvaluation,
    GatePolicy,
    PilotBudget,
    SmokeRecord,
    compare_candidates,
    compare_native_candidates,
    plan_pilot,
    validate_smoke_checkpoint,
)
from experiments.grug.moe_hero_ep.format_handoff import CandidateHandoffRequest, assemble_candidate
from experiments.grug.moe_hero_ep.format_sample import native_evaluation_model
from experiments.grug.moe_hero_ep.model import GrugModelConfig
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import Record, digest
from experiments.grug.moe_hero_ep.sft import BaseReference, SftConfig
from experiments.grug.moe_hero_ep.sft_launch import (
    HeroSftCheckpoint,
    HeroSftModel,
    PilotInspection,
    build_chat_sft_inputs,
    read_pilot_inspection,
    sft_measurement_identity,
)
from experiments.sft.launcher import (
    _TRAIN_RESOURCES,
    _prebuilt_chat_data_config,
)


@dataclasses.dataclass(frozen=True)
class FormatSftRecipe:
    name: str
    version: str
    reference: BaseReference
    model: GrugModelConfig
    sft: SftConfig
    data_manifest: str
    data_manifest_digest: str
    # The materialized dataset root can be uploaded once; the manifest retains
    # the original source content hashes and relative per-source filenames.
    data_root: str
    resources: ResourceConfig
    batch_size: int
    expert_axis_size: int
    context_axis_size: int
    replica_axis_size: int
    max_processed_tokens: int
    save_every_steps: int
    maximum_metrics: tuple[tuple[str, float], ...]
    gate_record: str
    acquisition_panel_path: str
    acquisition_panel_digest: str
    resume_checkpoint: str | None = None
    resume_sft_config: SftConfig | None = None


class PilotContinuation(Record):
    inspection_path: str
    inspection_digest: str
    candidate_handoff: CandidateHandoffRequest


class LaunchEvidence(Record):
    baseline: CandidateEvaluation
    parent: CandidateEvaluation
    prepared: CandidateEvaluation
    policy: GatePolicy
    local_validation_reports: tuple[str, ...] = Field(min_length=1)
    # A bounded learning smoke precedes the measured pilot budget.
    smoke: SmokeRecord | None = None
    pilot_budget: PilotBudget | None = None
    continuation: PilotContinuation | None = None


def validate_pilot_continuation(recipe: FormatSftRecipe, evidence: LaunchEvidence) -> PilotInspection:
    """Recompute native gates on the exact previous segment before any more updates."""
    assert evidence.pilot_budget is not None and evidence.continuation is not None
    continuation = evidence.continuation
    previous = read_pilot_inspection(continuation.inspection_path, continuation.inspection_digest)
    budget = evidence.pilot_budget
    identity = sft_measurement_identity(
        reference=recipe.reference,
        model=recipe.model,
        sft=recipe.sft,
        data_manifest_digest=recipe.data_manifest_digest,
        acquisition_panel_digest=recipe.acquisition_panel_digest,
        batch_size=recipe.batch_size,
        expert_axis_size=recipe.expert_axis_size,
        context_axis_size=recipe.context_axis_size,
        replica_axis_size=recipe.replica_axis_size,
        resources=recipe.resources,
    )
    if (
        previous.status != "awaiting_evaluation"
        or previous.handoff.step >= recipe.sft.num_steps
        or previous.handoff.processed_tokens >= recipe.max_processed_tokens
        or previous.sft != recipe.sft
        or recipe.resume_sft_config != recipe.sft
        or previous.handoff.base != recipe.reference
        or recipe.resume_checkpoint != previous.handoff.compact_checkpoint
        or previous.measurement_identity_digest != identity
        or previous.max_processed_tokens != recipe.max_processed_tokens
        or previous.inspection.early_update != budget.early_inspection_update
        or previous.inspection.non_padding_tokens != budget.inspection_non_padding_tokens
        or previous.inspection.policy_digest != digest(evidence.policy.model_dump(mode="json"))
        or previous.inspection.budget_digest != digest(budget.model_dump(mode="json"))
    ):
        raise ValueError("Pilot continuation changed its exact state, data, schedule or original inspection budget")
    request = continuation.candidate_handoff
    if (
        request.candidate != Candidate.D
        or request.parent != Candidate.C
        or request.checkpoint.uri != previous.handoff.native_checkpoint
        or request.checkpoint.metadata_digest != previous.handoff.native_metadata_digest
        or request.checkpoint.step != previous.handoff.step
        or request.tokenizer != recipe.reference.tokenizer
        or request.tokenizer_revision != recipe.reference.tokenizer_revision
        or request.format_panel_id != budget.format_panel_id
    ):
        raise ValueError("Pilot continuation evaluation does not pin the exact paused export and panels")
    candidate, _ = assemble_candidate(request)
    native = []
    for parent in (evidence.prepared, evidence.baseline):
        regression = compare_native_candidates(
            evidence.baseline, parent, candidate, evidence.policy, require_format=False
        )
        if regression["decision"] != "accept":
            raise ValueError(f"Pilot continuation requires diagnosis of native regression evidence: {regression}")
        native.append(
            compare_native_candidates(evidence.baseline, parent, candidate, evidence.policy, require_format=True)
        )
    if all(result["decision"] == "accept" for result in native):
        raise ValueError("Native formatting and content gates already pass; stop training even while serving is pending")
    if any(
        result["uncertain"] or any(not name.startswith("format_") for name in result["violations"]) for result in native
    ):
        raise ValueError("Only measured formatting failures can authorize another bounded pilot segment")
    return PilotInspection(
        early_update=budget.early_inspection_update,
        non_padding_tokens=budget.inspection_non_padding_tokens,
        policy_digest=previous.inspection.policy_digest,
        budget_digest=previous.inspection.budget_digest,
        reviewed_step=previous.handoff.step,
        reviewed_tokens=previous.handoff.processed_tokens,
        reviewed_inspection_digest=continuation.inspection_digest,
        reviewed_inspection_path=continuation.inspection_path,
        reviewed_evaluation_id=candidate.evaluation_id,
    )


def validate_prepared_ancestry(recipe: FormatSftRecipe, evidence: LaunchEvidence) -> None:
    """Bind evaluated A/B parents to C's permanent preparation provenance."""
    provenance = read_prepared_provenance(recipe.reference.checkpoint, recipe.reference.metadata_digest)
    if provenance["candidate"] != "C" or provenance["donors"] != [list(row) for row in _DONORS]:
        raise ValueError("Launch requires the permanent donor-initialized candidate C")
    if tuple(row[1] for row in provenance["donors"]) != recipe.sft.token_ids:
        raise ValueError("Launch vocabulary differs from candidate C's donor initialization")
    parent = provenance["selected_parent"]
    if (parent["path"], parent["metadata_digest"]) != (
        evidence.parent.checkpoint,
        evidence.parent.checkpoint_metadata_digest,
    ):
        raise ValueError("Evaluated parent differs from candidate C's actual ancestry")
    if evidence.parent.candidate == Candidate.B:
        averaged = read_prepared_provenance(parent["path"], parent["metadata_digest"])
        if averaged["candidate"] != "B":
            raise ValueError("Selected averaged parent is not candidate B")
        source = averaged["request"]["checkpoints"][-1]
    else:
        if "averaging_digest" in provenance["source_metadata"][0]:
            raise ValueError("An averaged C parent cannot be declared as baseline A")
        if evidence.parent.evaluation_id != evidence.baseline.evaluation_id:
            raise ValueError("Selected A parent must be the pinned baseline evaluation")
        source = parent
    if (source["path"], source["metadata_digest"]) != (
        evidence.baseline.checkpoint,
        evidence.baseline.checkpoint_metadata_digest,
    ):
        raise ValueError("Candidate C's latest source PT differs from the evaluated A baseline")
    if (recipe.reference.source_pt_checkpoint, recipe.reference.source_pt_step) != (
        source["path"],
        provenance["source_pt_step"],
    ):
        raise ValueError("Launch reference differs from candidate C's source PT ancestry")
    saved_model = draccus.encode(draccus.decode(GrugModelConfig, provenance["model"]))
    if digest(saved_model) != digest(draccus.encode(recipe.model)):
        raise ValueError("Launch model differs from candidate C's exact forward configuration")
    if provenance["tokenizer_pin"] != {
        "name": recipe.reference.tokenizer,
        "revision": recipe.reference.tokenizer_revision,
    }:
        raise ValueError("Launch tokenizer differs from candidate C's preparation")


def verify_training_data(recipe: FormatSftRecipe, manifest: dict) -> None:
    """Check the materialized training bytes against the prepared source manifest."""
    for slug, source in manifest["source_outputs"].items():
        if "train" not in source:
            continue
        fs, path = filesystem_for(prefix_join(recipe.data_root, f"train/{slug}.jsonl"))
        with fs.open(path, "rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != source["train"]["sha256"]:
            raise ValueError(f"Prepared training bytes changed for source {slug}")


def validate_launch_evidence(recipe: FormatSftRecipe, evidence: LaunchEvidence) -> dict:
    """Recompute C's regression gate and, for a pilot, its measured budget."""
    prepared = evidence.prepared
    if prepared.candidate != Candidate.C or prepared.parent != Candidate(recipe.reference.selected_parent):
        raise ValueError("Launch evidence does not describe candidate C and its selected parent")
    if (prepared.checkpoint, prepared.checkpoint_metadata_digest) != (
        recipe.reference.checkpoint,
        recipe.reference.metadata_digest,
    ):
        raise ValueError("Launch evidence refers to different prepared weights")
    if prepared.model_config_digest != digest(draccus.encode(native_evaluation_model(recipe.model))):
        raise ValueError("Launch forward configuration differs from the evaluated prepared candidate")
    validate_prepared_ancestry(recipe, evidence)
    if evidence.parent.candidate == Candidate.B:
        averaged_gate = compare_candidates(
            evidence.baseline, evidence.baseline, evidence.parent, evidence.policy, require_format=False
        )
        if averaged_gate["decision"] != "accept":
            raise ValueError(f"Selected B has not passed its A regression gate: {averaged_gate['decision']}")
    decision = compare_candidates(evidence.baseline, evidence.parent, prepared, evidence.policy, require_format=False)
    if decision["decision"] != "accept":
        raise ValueError(f"Prepared candidate has not passed its regression gate: {decision['decision']}")
    # A→B and B→C can each consume a regression allowance. Check the total too.
    cumulative = compare_candidates(
        evidence.baseline,
        evidence.baseline,
        prepared.model_copy(update={"parent": Candidate.A}),
        evidence.policy,
        require_format=False,
    )
    if cumulative["decision"] != "accept":
        raise ValueError(f"Prepared C has not passed its cumulative A regression gate: {cumulative['decision']}")
    if (evidence.smoke is None) != (evidence.pilot_budget is None):
        raise ValueError("Pilot evidence needs both measured smoke and budget; omit both for the initial smoke")
    if evidence.smoke is None:
        if (
            recipe.resume_checkpoint is not None
            or recipe.resume_sft_config is not None
            or evidence.continuation is not None
        ):
            raise ValueError("The initial learning smoke must start with fresh optimizer state")
        return {"stage": "learning_smoke", "candidate_gate": decision}
    assert evidence.pilot_budget is not None
    budget = plan_pilot(evidence.smoke, evidence.pilot_budget)
    if evidence.smoke.checkpoint_metadata_digest != recipe.reference.metadata_digest:
        raise ValueError("Pilot smoke used a different prepared base")
    if evidence.smoke.measurement_identity_digest != sft_measurement_identity(
        reference=recipe.reference,
        model=recipe.model,
        sft=recipe.sft,
        data_manifest_digest=recipe.data_manifest_digest,
        acquisition_panel_digest=recipe.acquisition_panel_digest,
        batch_size=recipe.batch_size,
        expert_axis_size=recipe.expert_axis_size,
        context_axis_size=recipe.context_axis_size,
        replica_axis_size=recipe.replica_axis_size,
        resources=recipe.resources,
    ):
        raise ValueError(
            "Pilot measurement identity differs from the actual smoke model, data, batch, mesh or resources"
        )
    if (
        evidence.smoke.gpu_count != recipe.resources.chip_count()
        or evidence.smoke.global_batch_executed_tokens != recipe.batch_size * recipe.model.max_seq_len
    ):
        raise ValueError("Measured smoke device count or executed batch shape differs from the pilot")
    validate_smoke_checkpoint(evidence.smoke)
    expected = evidence.pilot_budget
    if (
        recipe.max_processed_tokens != expected.cap_non_padding_tokens
        or recipe.sft.num_steps != budget["expected_cap_updates"]
        or recipe.sft.warmup_steps != expected.warmup_updates
        or recipe.sft.adapter_learning_rate != expected.adapter_peak_lr
        or recipe.sft.vocab_learning_rate != expected.vocab_peak_lr
    ):
        raise ValueError("Pilot launch does not match its measured token/update/LR budget")
    if evidence.continuation is not None:
        inspection = validate_pilot_continuation(recipe, evidence)
    else:
        if expected.reuse_smoke_optimizer != (recipe.resume_checkpoint is not None):
            raise ValueError("Pilot optimizer lineage disagrees with its measured budget")
        if expected.reuse_smoke_optimizer and recipe.resume_checkpoint != evidence.smoke.terminal_checkpoint:
            raise ValueError("Pilot must reuse the measured smoke's exact terminal compact checkpoint")
        if recipe.resume_sft_config != (evidence.smoke.sft_config if expected.reuse_smoke_optimizer else None):
            raise ValueError("Pilot must pin its reused smoke's original SFT schedule explicitly")
        inspection = PilotInspection(
            expected.early_inspection_update,
            expected.inspection_non_padding_tokens,
            digest(evidence.policy.model_dump(mode="json")),
            digest(expected.model_dump(mode="json")),
        )
    return {
        "stage": "pilot",
        "candidate_gate": decision,
        "pilot_budget": budget,
        "pilot_inspection": dataclasses.asdict(inspection),
    }


def build_format_sft(
    recipe: FormatSftRecipe,
    manifest: dict,
    acquisition: AcquisitionPanel,
    *,
    pilot_inspection: PilotInspection | None = None,
) -> ArtifactStep[HeroSftCheckpoint]:
    """Build the current-Hero backend with weighted, individually pinned chat sources."""
    contents = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    if digest(contents) != recipe.data_manifest_digest or manifest.get("manifest_digest") != recipe.data_manifest_digest:
        raise ValueError("Prepared dataset manifest does not match the launch pin")
    if manifest["tokenizer_revision"] != recipe.reference.tokenizer_revision:
        raise ValueError("Prepared chat data and candidate C have different tokenizer revisions")
    if manifest["recipe"]["tokenizer_path"] != recipe.reference.tokenizer:
        raise ValueError("Prepared chat data and candidate C have different tokenizer identities")
    if recipe.model.max_seq_len != 16384 or recipe.model.yarn is not None:
        raise ValueError("Format acquisition runs at 16K; prepare the separate YaRN candidate after D")
    if not recipe.gate_record:
        raise ValueError("Record the baseline/candidate/local-correctness gates before building a training recipe")
    if digest(acquisition.model_dump()) != recipe.acquisition_panel_digest:
        raise ValueError("Acquisition panel changed since launch preparation")
    if (acquisition.source_artifact_digest, acquisition.tokenizer_revision) != (
        recipe.data_manifest_digest,
        recipe.reference.tokenizer_revision,
    ):
        raise ValueError("Acquisition panel belongs to a different training dataset or tokenizer")
    backend = HeroSftModel(
        reference=recipe.reference,
        model=recipe.model,
        sft=recipe.sft,
        expert_axis_size=recipe.expert_axis_size,
        context_axis_size=recipe.context_axis_size,
        replica_axis_size=recipe.replica_axis_size,
        max_processed_tokens=recipe.max_processed_tokens,
        save_every_steps=recipe.save_every_steps,
        maximum_metrics=recipe.maximum_metrics,
        diagnostic=AcquisitionDiagnostic.from_panel(acquisition),
        data_manifest_digest=recipe.data_manifest_digest,
        acquisition_panel_digest=recipe.acquisition_panel_digest,
        resume_checkpoint=recipe.resume_checkpoint,
        resume_sft_config=recipe.resume_sft_config,
        pilot_inspection=pilot_inspection,
    )
    inputs = build_chat_sft_inputs(
        name=recipe.name,
        version=recipe.version,
        backend=backend,
        batch_size=recipe.batch_size,
        data_root=recipe.data_root,
        data_manifest_digest=recipe.data_manifest_digest,
        manifest=manifest,
    )
    spec, caches = inputs.spec, inputs.caches

    def build_config(ctx: StepContext):
        resources = ctx.runtime_arg(_TRAIN_RESOURCES)
        data = _prebuilt_chat_data_config(
            spec, [ctx.artifact_path(cache) for cache in caches], backend.resolve_tokenizer(ctx)
        )
        return backend.build_train_config(ctx, spec, data, resources, recipe.sft.num_steps)

    return ArtifactStep(
        name=spec.name,
        version=spec.version,
        artifact_type=HeroSftCheckpoint,
        run=backend.run,
        build_config=build_config,
        deps=(*caches, *backend.init_deps()),
        runtime_args={_TRAIN_RESOURCES: recipe.resources},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path, help="FormatSftRecipe JSON with exact data/base pins and regression limits")
    parser.add_argument("--output-prefix", required=True, help="Durable artifact root for config materialization")
    parser.add_argument("--launch-config", type=Path, required=True)
    parser.add_argument(
        "--run", action="store_true", help="Submit training after the recorded gates and launch approval"
    )
    args = parser.parse_args()
    recipe = draccus.decode(FormatSftRecipe, json.loads(args.recipe.read_text()))
    fs, path = filesystem_for(recipe.data_manifest)
    with fs.open(path) as stream:
        manifest = json.load(stream)
    gate_fs, gate_path = filesystem_for(recipe.gate_record)
    with gate_fs.open(gate_path) as stream:
        evidence = LaunchEvidence.model_validate_json(stream.read())
    gates = validate_launch_evidence(recipe, evidence)
    acquisition_fs, acquisition_path = filesystem_for(recipe.acquisition_panel_path)
    with acquisition_fs.open(acquisition_path) as stream:
        acquisition = AcquisitionPanel.model_validate_json(stream.read())
    inspection = draccus.decode(PilotInspection, gates["pilot_inspection"]) if "pilot_inspection" in gates else None
    step = build_format_sft(recipe, manifest, acquisition, pilot_inspection=inspection)
    verify_training_data(recipe, manifest)
    launch = materialized_config(step, args.output_prefix)
    with args.launch_config.open("x") as stream:
        stream.write(
            json.dumps(
                {"recipe": draccus.encode(recipe), "gates": gates, "run_config": draccus.encode(launch)},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    if args.run:
        if marin_prefix().rstrip("/") != args.output_prefix.rstrip("/"):
            raise ValueError("Runtime MARIN_PREFIX must match the reviewed --output-prefix before submission")
        StepRunner().run([lower(step)])


if __name__ == "__main__":
    main()
