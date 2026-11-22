# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch Exp5 SFT with R1 parallelism and special-token rates.

The 2026-11-20 user decision restores R1's sqrt(32) token multiplier,
reversing the earlier 1.0 ruling. SFT uses a 1000-step safety cap, 30 warmup
steps and a peak plateau before its wall-clock-only cooldown.
OpenResearcher's reserved share prevents downstream RL submission.
"""

import hashlib
import json
import os
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import click
import draccus
from iris.cluster.client.job_info import get_job_info
from marin.execution.lazy import ArtifactStep
from marin.rl.cli import rl_build_options
from marin.rl.curriculum import RlCurriculum, consumed_route_quotas
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training.long_loop.__main__ import validate_graph
from experiments.post_training.long_loop.exp3_production import StoragePin
from experiments.post_training.long_loop.exp3_recipe import (
    SftLrSchedule,
    verified_initializer_model,
    whole_run_lr_receipt,
)
from experiments.post_training.long_loop.exp4 import release_artifact
from experiments.post_training.long_loop.exp4_data import NEMOTRON_SOURCE
from experiments.post_training.long_loop.exp5_data import OUTPUT_PREFIX, ROUTES, pool_steps
from experiments.post_training.long_loop.optimizer_admission import optimizer_group_receipt
from experiments.post_training.long_loop.recipe_bundle import require_clean_revision, runtime_source_manifest
from experiments.post_training.long_loop.round_runner import (
    DATA_RELEASES,
    INITIALIZERS,
    RoundSpec,
    load_round,
    require_archive_binds_final_sft_checkpoint,
    require_permanent_round_storage,
    round_plan,
    round_sft_spec,
)
from experiments.post_training.long_loop.shared_sft import native_sft_step
from experiments.post_training.long_loop.spec_pipeline import build_spec_graph
from marin.rl.skyrl import SkyRLRolePlan
from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.exp4_curriculum import SOURCE_ROUTES
from experiments.post_training.task_curation.exp5_curriculum import (
    HARDMATH_LABEL,
    HARDMATH_REPO,
    IFBENCH_LABEL,
    IFBENCH_SOURCE,
    OPENRESEARCHER_LABEL,
    TARGET_SHARES,
    exp5_rl_curriculum,
    exp5_validation_from_manifest,
)
from experiments.post_training.task_curation.images.build import environment_artifact

# The Exp5 pools artifact the round binds: published at the build's run-wide version.
EXP5_POOLS_VERSION = "2026.11.22.1"
INITIAL_EVAL_INPUTS = "s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp5/setup/2026.11.20.2/initial-eval"
INITIAL_EVAL_FILES = (
    StoragePin(
        prefix_join(INITIAL_EVAL_INPUTS, "c4-data.json"),
        "e512763cfbefaa72dacd18ad40dcb0af7e391257be970e3c07cbdfc82c01fa48",
    ),
    StoragePin(
        prefix_join(INITIAL_EVAL_INPUTS, "c4-reference.json"),
        "f44c06454e7a44a2d28fc1dd9fd674e6d40d84481724065de22e49cd595a880b",
    ),
)
# The inherited 64-group window apportions the prepared 0.95 to 14 SWE groups.
# This diagnostic subset remains unlaunchable while the research slot is pending.
EXPECTED_ROUTE_QUOTAS = {"agentic": 13, "nonagentic": 51}
"""The curriculum-derived window quotas at the round's 64-group batch: 0.20 of 64 rounds to 13."""


def exp5_round_pools(resolved: dict) -> tuple[dict[str, ArtifactStep], RlCurriculum]:
    """Adopt the published Exp5 pools, verify them against the producer, and return the fixed curriculum."""
    producer = pool_steps(EXP5_POOLS_VERSION, grader_environment=environment_artifact(GRADER_PACKAGES))["pools"]
    path = producer.path(OUTPUT_PREFIX)
    record_bytes = (StoragePath(path) / ".artifact.json").read_bytes()
    if json.loads(record_bytes)["fingerprint"] != producer.fingerprint():
        raise ValueError("Regional Exp5 pool producer differs")
    manifest_bytes = (StoragePath(path) / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    curriculum = exp5_rl_curriculum(exp5_validation_from_manifest(manifest["validation"]))
    if manifest["curriculum"] != curriculum.identity:
        raise ValueError("Regional Exp5 pools were prepared for another curriculum")
    if not manifest.get("training_ready"):
        raise ValueError("The Exp5 pools are not training-ready")
    pins = [
        {"uri": prefix_join(path, name), "sha256": sha256}
        for name, sha256 in {**manifest["files"], "manifest.json": hashlib.sha256(manifest_bytes).hexdigest()}.items()
    ]
    pools = {
        "pools": ArtifactStep.adopt(
            producer.name,
            producer.version,
            path,
            config={
                "producer_fingerprint": producer.fingerprint(),
                "producer_record_sha256": hashlib.sha256(record_bytes).hexdigest(),
                "files": pins,
            },
        )
    }
    resolved["datasets"] = {
        **{key: value for key, value in resolved["datasets"].items() if key not in ("agentic", "nonagentic", "heldout")},
        "pools": {
            "uri": path,
            "rows": manifest["rows"],
            "files": pins,
            "composition": manifest["composition"],
            "routes": manifest["routes"],
        },
    }
    return pools, curriculum


@click.command(help=__doc__)
@click.option("--round", "round_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--source-revision", required=True)
@click.option("--validation-output", type=click.Path(path_type=Path), required=True)
@click.option("--stage", type=click.Choice(["sft", "archive"]), default="archive", show_default=True)
@rl_build_options
def main(round_path: Path, source_revision: str, validation_output: Path, stage: str) -> ArtifactStep:
    context = click.get_current_context()
    if context.params["do_run"] and stage != "sft":
        raise click.UsageError("Exp5 RL is on hold: OpenResearcher's 5% slot is not admitted; use --stage sft")
    configure_coreweave_s3()
    spec = load_round(round_path)
    if not isinstance(spec, RoundSpec) or spec.pools != "exp5-prepared":
        raise click.UsageError("Exp5 setup requires a training round with pools: exp5-prepared")
    if context.params["default_version"] != spec.version:
        raise click.UsageError("--version must equal the round's immutable version")
    workspace = Path(__file__).resolve().parents[3]
    if get_job_info() is None:
        require_clean_revision(workspace, source_revision)
    require_permanent_round_storage(spec)
    for pin in INITIAL_EVAL_FILES:
        pin.read()
    binding = DATA_RELEASES[spec.data_release]
    release, handle = release_artifact(binding.record, binding.record_sha256, require_ready=False)
    resolved = round_plan(
        replace(spec, pools="exp4-regional"),
        release,
        source_revision,
        spec.version,
        initial_eval_inputs=INITIAL_EVAL_INPUTS,
    )
    resolved["runtime_source"] = runtime_source_manifest(workspace, source_revision)
    resolved["sft"]["special_token_lr_multiplier"] = spec.sft.special_token_lr_multiplier
    resolved["exp5_admission"] = {
        "launch_ready": stage == "sft",
        "terminal_stage": stage,
        "launch_hold": None if stage == "sft" else "RL launches after the certified SFT handoff",
        "target_shares": dict(TARGET_SHARES),
        "prepared_absolute_share": 1.0,
        "reserved_share": 0,
        "pending_sources": {},
        "sampling_scope": (
            "All eleven sources are training-ready in the pools; OpenResearcher rides the thin-image plus "
            "hosted-retrieval service track, so no share stays reserved and none can redistribute."
        ),
        "openresearcher": {
            "track": "thin-image + hosted retrieval service (pin 0c4f3958, 10/10 native grades)",
            "runtime": "Daytona actor with the frozen BM25 retrieval service; the full 55 GB image is not a "
            "Daytona snapshot",
        },
        "science": "Reuse the exact qualified V7 science components; existing checked repairs are already present",
        "sft_ruling": {
            "reference": "9662fdba:experiments/post_training/long_loop/specs/sft.yaml",
            "replaces_parallelism": "CP32/EP8",
            "expert_axis_size": 1,
            "parameter_precision": spec.sft.mp,
            "optimizer_grouping": (
                "Retain corrected hero-EP grouping (#11639): stacked expert matrices use MuonH, "
                "RMSNorm scales use Adam, lm_head/output_proj use AdamH; "
                "R1 instead put experts on Adam and stacked norm scales on MuonH"
            ),
            "remat_mode": "recompute_all",
            "remat_implementation": "Native Snowball applies jax.checkpoint to every scanned layer",
            "special_token_lr_reason": "Restore R1 sqrt(32); the 2026-11-20 user decision reverses the 1.0 ruling",
            "memory_fit": "Unmeasured; requires execution on the requested GPU mesh",
            "native_axis_adaptation": (
                "R1 executed mesh: data32/replica_dcn1/context8/expert1; expert matrices "
                "partition experts over (expert,context) and hidden over data"
            ),
        },
        "initial_eval_inputs": {"prefix": INITIAL_EVAL_INPUTS, "files": [asdict(pin) for pin in INITIAL_EVAL_FILES]},
        "recipe_schema_migrations": [
            "Remove the unused tis_splice key from the frozen RL recipe",
            "Rename Daytona provisioning keys to image_build keys; preserve snapshot/declarative values",
        ],
    }
    # The current trainer has no tis_splice option; this frozen key was unused.
    del resolved["rl"]["config"]["trainer"]["algorithm"]["tis_splice"]
    harbor = resolved["rl"]["config"]["terminal_bench"]["harbor"]
    harbor["daytona_image_build"] = harbor.pop("daytona_provisioning")
    harbor["eval_daytona_image_build"] = harbor.pop("eval_daytona_provisioning")
    os.environ.update(MARIN_PREFIX=spec.storage.output_prefix, MARIN_TEMP_PREFIX=spec.storage.temporary_prefix)
    pools, curriculum = exp5_round_pools(resolved)
    resolved["curriculum"] = {**asdict(curriculum), "identity": curriculum.identity}
    buffer = resolved["rl"]["config"]["trainer"]["rollout_buffer"]
    roles = draccus.decode(SkyRLRolePlan, resolved["rl"]["role_plan"])
    derived = consumed_route_quotas(curriculum, ROUTES, roles.train_batch_size)
    if roles.train_batch_size == 64 and derived != EXPECTED_ROUTE_QUOTAS:
        raise ValueError(
            f"The curriculum derives {derived!r} at batch {roles.train_batch_size}; "
            f"the recorded expectation is {EXPECTED_ROUTE_QUOTAS!r}"
        )
    buffer["consumed_route_quotas"] = derived
    buffer["source_routes"] = dict(ROUTES)
    require_archive_binds_final_sft_checkpoint(resolved)
    resolved["resolved_sha256"] = hashlib.sha256(
        json.dumps({key: value for key, value in resolved.items() if key != "resolved_sha256"}, sort_keys=True).encode()
    ).hexdigest()
    terminal = build_spec_graph(
        resolved,
        spec.version,
        through=stage,
        data_artifacts=pools if stage != "sft" else None,
        curriculum=curriculum,
        sft_release=handle,
        sft_step_builder=partial(native_sft_step, spec_builder=partial(round_sft_spec, spec)),
    )
    validation_output.mkdir(parents=True, exist_ok=True)
    for key in ("sft_capacity", "sft_exposure"):
        (validation_output / f"{key}.json").write_text(json.dumps(resolved["sft"][key], indent=2) + "\n")
    (validation_output / "resolved-plan.json").write_text(json.dumps(resolved, indent=2) + "\n")
    (validation_output / "optimizer-groups.json").write_text(
        json.dumps(optimizer_group_receipt(resolved["sft"]), indent=2)
    )
    (validation_output / "sft-lr-schedule.json").write_text(
        json.dumps(
            whole_run_lr_receipt(
                round_sft_spec(spec, f"{spec.name}/sft", spec.version, resolved["sft"]),
                schedule=draccus.decode(SftLrSchedule, resolved["sft"]["optimizer_schedule"]),
            ),
            indent=2,
        )
    )
    initializer = INITIALIZERS[spec.initializer]
    model = verified_initializer_model(initializer.receipt, initializer.receipt_sha256)
    validate_graph(
        terminal,
        validation_output,
        spec.storage.output_prefix,
        model_config=(StoragePath(model) / "config.json").read_bytes(),
    )
    return terminal


if __name__ == "__main__":
    main()
