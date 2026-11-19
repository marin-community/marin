# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build Exp5 from a qualified V7 artifact, the Exp4 curriculum and frozen Micro210.

Exp3 and Exp4 SFT multiplied special-token updates by sqrt(32) (~5.66x), and Exp3's
run showed submit/finish tool-call imprinting with a 0/10 Micro210 SWE-bench score.
Per R2, Exp5 copies Exp4 .76's resolved recipe with that multiplier held at 1.0,
asserted at construction and changeable only through an explicit override, while
keeping the annealing rule (optimizer horizon equals the updates actually run and
the final learning rate lands near min_lr).
"""

import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import click
from iris.cluster.client.job_info import get_job_info
from levanter.models.snowball_sft import SnowballSftConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.rl.cli import rl_build_options
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.evaluation.micro210_policy import Micro210Model, micro210_evaluation
from experiments.post_training.long_loop.__main__ import validate_graph
from experiments.post_training.long_loop.exp3_production import (
    INITIALIZER_RECEIPT,
    INITIALIZER_SHA256,
    DataRelease,
    baseline_files,
    production_plan,
)
from experiments.post_training.long_loop.exp3_recipe import (
    exp3_controls,
    verified_initializer_model,
    whole_run_lr_receipt,
)
from experiments.post_training.long_loop.exp3_rl_restart import restart_recipe
from experiments.post_training.long_loop.exp4 import (
    OUTPUT_PREFIX as EXP4_OUTPUT_PREFIX,
)
from experiments.post_training.long_loop.exp4 import REFERENCE_SHA256, REFERENCE_URI, SFT_UPDATES, release_artifact
from experiments.post_training.long_loop.exp4_data import NEMOTRON_SOURCE, pool_steps
from experiments.post_training.long_loop.optimizer_admission import optimizer_group_receipt
from experiments.post_training.long_loop.recipe_bundle import require_clean_revision, runtime_source_manifest
from experiments.post_training.long_loop.shared_sft import native_sft_step, shared_sft_spec
from experiments.post_training.long_loop.spec_pipeline import build_spec_graph
from experiments.post_training.task_curation.exp4_curriculum import SOURCE_ROUTES, exp4_rl_curriculum
from experiments.sft.launcher import SFTSpec

OUTPUT_PREFIX = "s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp5"
# R2 copies everything else forward from Exp4 .76, including its admitted RL pools;
# Exp5 binds those artifacts where they were produced instead of re-stamping them.
POOL_PREFIX = EXP4_OUTPUT_PREFIX

SPECIAL_TOKEN_LR_MULTIPLIER = 1.0


def exp5_sft_controls(*, special_token_lr_multiplier: float | None = None) -> SnowballSftConfig:
    """Exp5's SFT controls: R2 holds the special-token update rate at the base LR.

    The multiplier is 1.0 by construction; only an explicit argument may change it.
    """
    if special_token_lr_multiplier is None:
        special_token_lr_multiplier = SPECIAL_TOKEN_LR_MULTIPLIER
        assert special_token_lr_multiplier == 1.0, "R2: Exp5 special-token LR multiplier must be 1.0"
    return replace(exp3_controls(), special_token_lr_multiplier=special_token_lr_multiplier)


def exp5_sft_spec(name: str, version: str, recipe: dict) -> SFTSpec:
    """Bind the shared Exp4 SFT spec with Exp5's special-token learning-rate ruling."""
    spec = shared_sft_spec(name, version, recipe)
    multiplier = recipe.get("special_token_lr_multiplier", SPECIAL_TOKEN_LR_MULTIPLIER)
    spec = replace(spec, snowball_sft=exp5_sft_controls(special_token_lr_multiplier=multiplier))
    if spec.snowball_sft.special_token_lr_multiplier != multiplier:
        raise ValueError("Exp5 special-token LR multiplier disagrees with its resolved recipe")
    whole_run_lr_receipt(spec)
    return spec


def exp5_sft_step(name: str, version: str, recipe: dict) -> ArtifactStep:
    return native_sft_step(name, version, recipe, spec_builder=exp5_sft_spec)


def exp5_plan(release: DataRelease, source_revision: str, version: str) -> dict:
    resolved = production_plan(release, source_revision, baseline_files())
    sft = resolved["sft"]
    sft["stop_after_steps"] = SFT_UPDATES
    sft["trainer"]["num_train_steps"] = SFT_UPDATES
    sft["target_positions"] = SFT_UPDATES * sft["batch_size"] * sft["seq_len"]
    sft["optimizer_schedule"] = {
        "horizon": SFT_UPDATES,
        "warmup": 0,
        "decay": SFT_UPDATES,
        "min_lr_ratio": 0.1,
        "lr_schedule": "linear",
    }
    sft["special_token_lr_multiplier"] = SPECIAL_TOKEN_LR_MULTIPLIER
    raw = StoragePath(REFERENCE_URI).read_bytes()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA256:
        raise ValueError("Exp3 comparison reference changed")
    reference = json.loads(raw)
    rl = reference["plan"]["rl"]
    recipe = restart_recipe(reference)
    recipe["extra_env"]["NCCL_BUFFSIZE"] = "4194304"
    recipe["terminal_bench"]["harbor"].pop("eval_override_storage_mb", None)
    recipe["trainer"]["rollout_buffer"]["consumed_route_quotas"] = {"agentic": 22, "nonagentic": 42}
    recipe["trainer"]["rollout_buffer"]["source_routes"] = {
        **SOURCE_ROUTES,
        NEMOTRON_SOURCE: "nonagentic",
    }
    recipe["trainer"]["callbacks"] = [
        callback for callback in recipe["trainer"]["callbacks"] if callback["type"] != "wall_time_budget"
    ]
    resolved["rl"] = {
        **rl,
        "name": "mochi-exp5-rl",
        "config": recipe,
        "wall_clock_budget": {"seconds": 43200, "drain_margin": 1800},
    }
    resolved["name"] = "long-loop-exp5/round5"
    resolved["storage"]["output_prefix"] = OUTPUT_PREFIX
    resolved["execution"]["coordinator"]["coordinator_name"] = "mochi-exp5"
    canonical = micro210_evaluation(
        Micro210Model(
            "s3://unresolved/terminal",
            "sha256:" + "0" * 64,
            resolved["tokenizer"]["repo"],
            resolved["tokenizer"]["revision"],
        ),
        run_id=f"mochi-exp5@{version}",
        output_uri="<resolved-at-build>",
    )
    resolved["evaluation"] = {
        **canonical.policy()["evaluation"],
        "name": "users/benfeuer/checkpoints/mochi-exp5-micro210",
        "expected_rollouts": 440,
    }
    for route in ("agentic", "nonagentic"):
        resolved["datasets"][route]["uri"] = f"{POOL_PREFIX}/long-loop-exp4/rl-pool-{route}/{version}"
        resolved["datasets"][route]["data_file"] = "train.parquet"
    resolved["datasets"]["nonagentic"]["source_names"] = [
        *resolved["datasets"]["nonagentic"]["source_names"],
        NEMOTRON_SOURCE,
    ]
    resolved["archive"]["hf_upload"]["repo_prefix"] = "open-athena/Snowball-LongLoop-Round5"
    resolved["archive"]["evaldash"]["run_prefix"] = "long-loop-round5"
    resolved["curriculum"] = {**asdict(exp4_rl_curriculum()), "identity": exp4_rl_curriculum().identity}
    resolved["resolved_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in resolved.items() if key != "resolved_sha256"},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return resolved


@click.command(help=__doc__)
@click.option("--sft-artifact-record", required=True)
@click.option("--sft-artifact-record-sha256", required=True)
@click.option("--source-revision", required=True)
@click.option("--validation-output", type=click.Path(path_type=Path), required=True)
@rl_build_options
def main(
    sft_artifact_record: str, sft_artifact_record_sha256: str, source_revision: str, validation_output: Path
) -> ArtifactStep:
    configure_coreweave_s3()
    workspace = Path(__file__).resolve().parents[3]
    if get_job_info() is None:
        require_clean_revision(workspace, source_revision)
    release, handle = release_artifact(
        sft_artifact_record, sft_artifact_record_sha256, require_ready=click.get_current_context().params["do_run"]
    )
    version = resolve_version("mochi-exp5", None)
    resolved = exp5_plan(release, source_revision, version)
    resolved["runtime_source"] = runtime_source_manifest(workspace, source_revision)
    os.environ.update(MARIN_PREFIX=OUTPUT_PREFIX, MARIN_TEMP_PREFIX="s3://marin-us-east-08a/tmp")
    pool_version = "2026.11.18.71"
    produced = pool_steps(pool_version)
    pools = {}
    for route, producer in produced.items():
        path = producer.path(POOL_PREFIX)
        record_bytes = (StoragePath(path) / ".artifact.json").read_bytes()
        producer_record = json.loads(record_bytes)
        if producer_record["fingerprint"] != producer.fingerprint():
            raise ValueError(f"Regional Exp4 pool producer differs: {route}")
        manifest_bytes = (StoragePath(path) / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        if manifest["curriculum"] != exp4_rl_curriculum().identity:
            raise ValueError("Regional pool was prepared for another curriculum")
        pins = [
            {"uri": path + "/train.parquet", "sha256": manifest["sha256"]},
            {"uri": path + "/manifest.json", "sha256": hashlib.sha256(manifest_bytes).hexdigest()},
        ]
        pools[route] = ArtifactStep.adopt(
            producer.name,
            producer.version,
            path,
            config={
                "producer_fingerprint": producer.fingerprint(),
                "producer_record_sha256": hashlib.sha256(record_bytes).hexdigest(),
                "files": pins,
            },
        )
        resolved["datasets"][route].update(
            uri=path,
            rows=manifest["rows"],
            files=pins,
            source_names=[NEMOTRON_SOURCE if label == "nemotron_if" else label for label in manifest["source_counts"]],
            teacher_routes=manifest["source_counts"],
        )
    resolved["resolved_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in resolved.items() if key != "resolved_sha256"},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    terminal = build_spec_graph(
        resolved,
        version,
        through="archive",
        data_artifacts=pools,
        curriculum=exp4_rl_curriculum(),
        sft_release=handle,
        sft_step_builder=exp5_sft_step,
    )
    validation_output.mkdir(parents=True, exist_ok=True)
    (validation_output / "resolved-plan.json").write_text(json.dumps(resolved, indent=2))
    (validation_output / "optimizer-groups.json").write_text(
        json.dumps(optimizer_group_receipt(resolved["sft"]), indent=2)
    )
    (validation_output / "sft-lr-schedule.json").write_text(
        json.dumps(
            whole_run_lr_receipt(exp5_sft_spec("long-loop-exp5/round5/sft", version, resolved["sft"])), indent=2
        )
    )
    model = verified_initializer_model(INITIALIZER_RECEIPT, INITIALIZER_SHA256)
    validate_graph(
        terminal, validation_output, OUTPUT_PREFIX, model_config=(StoragePath(model) / "config.json").read_bytes()
    )
    return terminal


if __name__ == "__main__":
    main()
