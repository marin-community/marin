# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve the held Exp5 setup with R1 SFT parallelism and special-token rates.

The 2026-11-20 user decision restores R1's sqrt(32) token multiplier,
reversing the earlier 1.0 ruling. SFT uses a 4750-step safety cap, 30 warmup
steps and a peak plateau before its remaining-time-triggered cooldown.
OpenResearcher's reserved share still prevents submission.
"""

import hashlib
import json
import os
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import click
import draccus
from marin.execution.lazy import ArtifactStep
from marin.rl.cli import rl_build_options
from marin.rl.curriculum import RlCurriculum
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
from experiments.post_training.long_loop.exp5_data import (
    AGENTIC_MANIFEST_SHA256,
    OUTPUT_PREFIX,
    base_pool,
    pool_step,
)
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
from experiments.post_training.task_curation.exp4_curriculum import SOURCE_ROUTES
from experiments.post_training.task_curation.exp5_curriculum import (
    HARDMATH_LABEL,
    HARDMATH_REPO,
    IFBENCH_LABEL,
    IFBENCH_SOURCE,
    PENDING_SOURCES,
    TARGET_SHARES,
    exp5_rl_curriculum,
)

DATA_VERSION = "2026.11.20.1"
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
PREPARATION_ROUTE_QUOTAS = {"agentic": 14, "nonagentic": 50}
TERMINAL_STAGE = "archive"


def prepared_curriculum() -> RlCurriculum:
    """Keep the prepared sources' absolute shares; the missing five percent stays explicit."""
    complete = exp5_rl_curriculum()
    return replace(
        complete,
        name="long-loop/exp5-preparation-only",
        entries=tuple(entry for entry in complete.entries if entry.label not in PENDING_SOURCES),
    )


def prepared_pools(resolved: dict) -> dict[str, ArtifactStep]:
    """Pin the preserved SWE pool and the verified Exp5 nonagentic producer."""
    producer = pool_step(DATA_VERSION)
    paths = {"agentic": base_pool("agentic"), "nonagentic": producer.path(OUTPUT_PREFIX)}
    adopted = {}
    for route, path in paths.items():
        manifest_bytes = (StoragePath(path) / "manifest.json").read_bytes()
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        manifest = json.loads(manifest_bytes)
        record_bytes = (StoragePath(path) / ".artifact.json").read_bytes()
        if route == "agentic":
            if manifest_sha != AGENTIC_MANIFEST_SHA256:
                raise ValueError("Exp4 SWE pool manifest changed")
        elif (
            json.loads(record_bytes)["fingerprint"] != producer.fingerprint()
            or manifest["curriculum"] != exp5_rl_curriculum().identity
            or manifest["pending_sources"] != dict(PENDING_SOURCES)
        ):
            raise ValueError("Exp5 prepared pool does not match the held curriculum")
        files = [
            {"uri": prefix_join(path, "train.parquet"), "sha256": manifest["sha256"]},
            {"uri": prefix_join(path, "manifest.json"), "sha256": manifest_sha},
        ]
        adopted[route] = ArtifactStep.adopt(
            f"long-loop-exp5/prepared-{route}",
            DATA_VERSION,
            path,
            config={"files": files, "producer_record_sha256": hashlib.sha256(record_bytes).hexdigest()},
        )
        sources = manifest["source_counts"]
        provenance = {"nemotron_if": NEMOTRON_SOURCE, IFBENCH_LABEL: IFBENCH_SOURCE, HARDMATH_LABEL: HARDMATH_REPO}
        resolved["datasets"][route].update(
            uri=path,
            data_file="train.parquet",
            rows=manifest["rows"],
            files=files,
            source_names=[provenance.get(label, label) for label in sources],
            teacher_routes=sources,
        )
    return adopted


@click.command(help=__doc__)
@click.option("--round", "round_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--source-revision", required=True)
@click.option("--validation-output", type=click.Path(path_type=Path), required=True)
@rl_build_options
def main(round_path: Path, source_revision: str, validation_output: Path) -> ArtifactStep:
    context = click.get_current_context()
    if context.params["do_run"]:
        raise click.UsageError("Exp5 launch is on hold: OpenResearcher's 5% slot is not admitted; --run is disabled")
    configure_coreweave_s3()
    spec = load_round(round_path)
    if not isinstance(spec, RoundSpec) or spec.pools != "exp5-prepared":
        raise click.UsageError("Exp5 setup requires a training round with pools: exp5-prepared")
    if context.params["default_version"] != spec.version:
        raise click.UsageError("--version must equal the round's immutable version")
    workspace = Path(__file__).resolve().parents[3]
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
    curriculum = prepared_curriculum()
    resolved["curriculum"] = {**asdict(curriculum), "identity": curriculum.identity}
    resolved["exp5_admission"] = {
        "launch_ready": False,
        "terminal_stage": TERMINAL_STAGE,
        "launch_hold": "User review of Exp4 evaluation and unresolved OpenResearcher runtime",
        "target_shares": dict(TARGET_SHARES),
        "prepared_absolute_share": sum(curriculum.weights().values()),
        "reserved_share": sum(TARGET_SHARES[label] for label in PENDING_SOURCES),
        "pending_sources": dict(PENDING_SOURCES),
        "sampling_scope": (
            "Static preparation only. A sampler would normalize the prepared 0.95; "
            "--run is disabled so the reserved 0.05 cannot be redistributed."
        ),
        "science": "Reuse the exact qualified V7 science components; existing checked repairs are already present",
        "sft_ruling": {
            "reference": "9662fdba:experiments/post_training/long_loop/specs/sft.yaml",
            "replaces_parallelism": "CP32/EP8",
            "expert_axis_size": 1,
            "parameter_precision": "Retain FP32 master parameters and BF16 compute; R1 used BF16 parameters",
            "optimizer_grouping": (
                "Retain corrected hero-EP grouping (#11639): stacked expert matrices use MuonH, "
                "RMSNorm scales use Adam, lm_head/output_proj use AdamH; "
                "R1 instead put experts on Adam and stacked norm scales on MuonH"
            ),
            "remat_mode": "recompute_all",
            "remat_implementation": "Native Snowball applies jax.checkpoint to every scanned layer",
            "special_token_lr_reason": "Restore R1 sqrt(32); the 2026-11-20 user decision reverses the 1.0 ruling",
            "memory_fit": "Unmeasured; requires execution on the requested GPU mesh",
            "native_axis_adaptation": "Explicit expert: 1 preserves R1 parallelism and satisfies native partition specs",
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
    buffer = resolved["rl"]["config"]["trainer"]["rollout_buffer"]
    buffer["consumed_route_quotas"] = dict(PREPARATION_ROUTE_QUOTAS)
    buffer["source_routes"] = {
        **SOURCE_ROUTES,
        NEMOTRON_SOURCE: "nonagentic",
        IFBENCH_LABEL: "nonagentic",
        IFBENCH_SOURCE: "nonagentic",
        HARDMATH_LABEL: "nonagentic",
        HARDMATH_REPO: "nonagentic",
    }
    os.environ.update(MARIN_PREFIX=spec.storage.output_prefix, MARIN_TEMP_PREFIX=spec.storage.temporary_prefix)
    pools = prepared_pools(resolved)
    require_archive_binds_final_sft_checkpoint(resolved)
    resolved["resolved_sha256"] = hashlib.sha256(
        json.dumps({key: value for key, value in resolved.items() if key != "resolved_sha256"}, sort_keys=True).encode()
    ).hexdigest()
    terminal = build_spec_graph(
        resolved,
        spec.version,
        through=TERMINAL_STAGE,
        data_artifacts=pools,
        curriculum=curriculum,
        sft_release=handle,
        sft_step_builder=partial(native_sft_step, spec_builder=partial(round_sft_spec, spec)),
    )
    validation_output.mkdir(parents=True, exist_ok=True)
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
