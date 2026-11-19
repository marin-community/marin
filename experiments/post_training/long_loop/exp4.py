# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build Exp4 from a qualified V7 artifact, the Exp4 curriculum and frozen Micro210."""

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import click
from iris.cluster.client.job_info import get_job_info
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.rl.cli import rl_build_options
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.long_loop.__main__ import validate_graph
from experiments.post_training.long_loop.exp3_production import (
    GEOMETRY,
    INITIALIZER_RECEIPT,
    INITIALIZER_SHA256,
    DataRelease,
    StoragePin,
    baseline_files,
    production_plan,
)
from experiments.post_training.long_loop.exp3_recipe import verified_initializer_model
from experiments.post_training.long_loop.exp3_rl_restart import restart_recipe
from experiments.post_training.long_loop.exp4_data import NEMOTRON_SOURCE, pool_steps
from experiments.post_training.long_loop.optimizer_admission import optimizer_group_receipt
from experiments.post_training.long_loop.recipe_bundle import require_clean_revision, runtime_source_manifest
from experiments.post_training.long_loop.spec_pipeline import build_spec_graph
from experiments.post_training.task_curation.exp4_curriculum import SOURCE_ROUTES, exp4_rl_curriculum

OUTPUT_PREFIX = "s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp4"
REFERENCE_URI = "s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp3/evidence/restart53/frozen51-reference.json"
REFERENCE_SHA256 = "897f22035802380a63fb748739ae8a66ef208ea67f1fe1e5470bbec417b3e1bb"


def release_artifact(uri: str, sha256: str, *, require_ready: bool) -> tuple[DataRelease, ArtifactStep]:
    """Reconstruct the published handle only after its record and members verify."""
    raw = StoragePath(uri).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("V7 artifact record differs from the approved SHA256")
    record = json.loads(raw)
    configuration = record["config"]
    manifest = StoragePin(record["source"], configuration["sha256"])
    document = json.loads(manifest.read())
    geometry = document["geometry"]
    if geometry != {
        "updates": GEOMETRY.updates,
        "global_batch": GEOMETRY.global_batch,
        "sequence_length": GEOMETRY.sequence_length,
        "seed": GEOMETRY.seed,
        "packing_cap": GEOMETRY.packing_cap,
        "lookahead": GEOMETRY.lookahead,
    }:
        raise ValueError("V7 changes the controlled SFT geometry")
    if require_ready and document["training_ready"] is not True:
        raise ValueError(f"V7 is not admitted for training: {document.get('readiness')}")
    members = {name: StoragePin(**value) for name, value in configuration["members"].items()}
    for name, pin in members.items():
        if document["artifacts"][name] != {"uri": pin.uri, "sha256": pin.sha256}:
            raise ValueError(f"V7 artifact member disagrees with its release: {name}")
        pin.read()
    data = json.loads(members["data.json"].read())
    if data["cache_catalog"] != members["cache_catalog.json"].uri:
        raise ValueError("V7 data names another cache artifact")
    if data["train_weights"] != json.loads(members["train_weights.json"].read()):
        raise ValueError("V7 natural mixture weights differ from the pinned member")
    handle = ArtifactStep.adopt(record["name"], record["version"], record["source"], config=configuration)
    if handle.fingerprint() != record["fingerprint"]:
        raise ValueError("Reconstructed V7 handle differs from its published identity")
    return DataRelease(manifest, document["training_ready"], members, {}, data, document), handle


def exp4_plan(release: DataRelease, source_revision: str, version: str) -> dict:
    resolved = production_plan(release, source_revision, baseline_files())
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
        "name": "mochi-exp4-rl",
        "config": recipe,
        "wall_clock_budget": {"seconds": 43200, "drain_margin": 1800},
    }
    resolved["name"] = "long-loop-exp4/round4"
    resolved["storage"]["output_prefix"] = OUTPUT_PREFIX
    resolved["execution"]["coordinator"]["coordinator_name"] = "mochi-exp4"
    resolved["evaluation"]["name"] = "users/benfeuer/checkpoints/mochi-exp4-micro210"
    resolved["evaluation"]["expected_rollouts"] = 440
    for route in ("agentic", "nonagentic"):
        resolved["datasets"][route]["uri"] = f"{OUTPUT_PREFIX}/long-loop-exp4/rl-pool-{route}/{version}"
        resolved["datasets"][route]["data_file"] = "train.parquet"
    resolved["datasets"]["nonagentic"]["source_names"] = [
        *resolved["datasets"]["nonagentic"]["source_names"],
        NEMOTRON_SOURCE,
    ]
    resolved["archive"]["hf_upload"]["repo_prefix"] = "open-athena/Snowball-LongLoop-Round4"
    resolved["archive"]["evaldash"]["run_prefix"] = "long-loop-round4"
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
    version = resolve_version("mochi-exp4", None)
    resolved = exp4_plan(release, source_revision, version)
    resolved["runtime_source"] = runtime_source_manifest(workspace, source_revision)
    os.environ.update(MARIN_PREFIX=OUTPUT_PREFIX, MARIN_TEMP_PREFIX="s3://marin-us-east-08a/tmp")
    pool_version = "2026.11.18.71"
    produced = pool_steps(pool_version)
    pools = {}
    for route, producer in produced.items():
        record = StoragePath(producer.path(OUTPUT_PREFIX)) / ".artifact.json"
        if click.get_current_context().params["do_run"] and not record.exists():
            raise ValueError(f"Regional Exp4 pool not committed: {record}")
        pools[route] = ArtifactStep.adopt(
            producer.name,
            producer.version,
            producer.path(OUTPUT_PREFIX),
            config={"producer_fingerprint": producer.fingerprint()},
        )
        resolved["datasets"][route]["uri"] = producer.path(OUTPUT_PREFIX)
    terminal = build_spec_graph(
        resolved, version, through="archive", data_artifacts=pools, curriculum=exp4_rl_curriculum(), sft_release=handle
    )
    validation_output.mkdir(parents=True, exist_ok=True)
    (validation_output / "resolved-plan.json").write_text(json.dumps(resolved, indent=2))
    (validation_output / "optimizer-groups.json").write_text(
        json.dumps(optimizer_group_receipt(resolved["sft"]), indent=2)
    )
    model = verified_initializer_model(INITIALIZER_RECEIPT, INITIALIZER_SHA256)
    validate_graph(
        terminal, validation_output, OUTPUT_PREFIX, model_config=(StoragePath(model) / "config.json").read_bytes()
    )
    return terminal


if __name__ == "__main__":
    main()
