# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve and guard the isolated Snowball publication/loader baseline before launch."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

import marin.external_dependencies as dependency
import wandb
import yaml
from iris.cli.connect import open_iris_client
from iris.cli.job import add_standard_env_vars, build_resources, load_env_vars
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, Constraint, ConstraintOp
from iris.cluster.types import Entrypoint, EnvironmentSpec
from iris.rpc import job_pb2
from marin.execution.lazy import materialized_config, run
from marin.rl.skyrl import SkyRLCheckpoint, SkyRLTrainingResult, _launcher_command, sanitize_job_name
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from experiments.post_training import async_snowball as recipe

MSR_COMMIT = "cf4b0d8cf92d97f9354bd374a2df765555a9dd31"
PREFIX = "s3://marin-us-east-02a/marin"
JOB_NAME = "async-rl-v2-snowball-k6-e3-v1"
SOURCES = (
    "uv.lock",
    "pyproject.toml",
    "experiments/post_training/async_snowball.py",
    "experiments/post_training/async_rl.py",
    "experiments/post_training/launch_snowball_publication_gate.py",
    "lib/marin/src/marin/external_dependencies.py",
    "lib/marin/src/marin/rl/skyrl.py",
    "lib/marin/src/marin/execution/lazy.py",
)


def resolved():

    root = Path(__file__).resolve().parents[2]
    assert Path(dependency.__file__).resolve().is_relative_to(root)
    assert Path(recipe.__file__).resolve().is_relative_to(root)
    assert os.environ["MARIN_PREFIX"] == PREFIX
    recipe.validate_regional_storage(PREFIX, recipe.CLUSTER)
    training = recipe.build_experiment(
        version="2026.09.08.34",
        scale=recipe.Scale.CADENCE_GATE,
        completion="metrics",
        timeout_seconds=1350,
        inference_replicas=1,
        publication_stage_timing=True,
        dataloader_workers=0,
        train_rows=128,
        validation_rows=128,
        response_tokens=4096,
        eval_response_tokens=4096,
        context_tokens=8192,
        initial_eval_repeat_count=1,
        weight_sync_interval=1,
        max_staleness_steps=1,
        eval_interval=5,
        epoch_seeded_shuffle=True,
    )
    config = materialized_config(training, PREFIX)
    data = training.deps[1]
    data_config = materialized_config(data, PREFIX)
    request = config.request
    settings = yaml.safe_load(request.config_yaml)
    assert request.runtime.commit == MSR_COMMIT and request.runtime.profile == "megatron"
    assert request.completion_mode == "metrics" and request.seed == 17
    assert data_config.train_rows == data_config.validation_rows == 128
    assert settings["data"]["num_workers"] == 0 and settings["data"]["epoch_seeded_shuffle"]
    assert settings["trainer"]["max_steps"] == settings["trainer"]["eval_interval"] == 5
    assert settings["trainer"]["eval_before_train"]
    assert settings["trainer"].get("initial_eval_repeat_count", 1) == 1
    assert settings["trainer"]["train_batch_size"] == settings["trainer"]["policy_mini_batch_size"] == 32
    assert settings["trainer"]["fully_async"]["weight_sync_interval"] == 1
    assert settings["trainer"]["fully_async"]["max_staleness_steps"] == 1
    assert settings["generator"]["publication_stage_timing"]
    assert settings["context_budget"]["max_new_tokens_per_turn"] == 4096
    assert settings["context_budget"]["request_window_tokens"] == 8192
    assert settings["generator"]["eval_sampling_params"]["max_generate_length"] == 4096
    assert request.topology.num_nodes == 5 and request.topology.gpus_per_node == 8
    assert request.topology.role_plan.policy_num_nodes == 4 and not request.topology.role_plan.colocate_all
    assert config.execution.timeout_seconds == 1350 and config.execution.max_retries == 0
    assert config.execution.priority == "batch" and config.execution.cluster == "cw-us-east-02a"
    canonical = asdict(request)
    canonical.pop("attempt_id")
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    report = {
        "request_hash": digest,
        "request": asdict(request),
        "execution": asdict(config.execution),
        "training_fingerprint": training.fingerprint(),
        "data_fingerprint": data.fingerprint(),
        "data": asdict(data_config),
    }
    return training, config, report


def guard_absent(terminal_uri, receipt_glob, selection_uri):

    assert not StoragePath(terminal_uri).exists(), "Existing terminal result blocks duplicate gate"
    assert not StoragePath(receipt_glob).glob(), "Existing training receipts block duplicate gate"
    assert not StoragePath(selection_uri).exists(), "Existing data selection blocks duplicate gate"


def coordinate():

    for name, digest in json.loads(os.environ["SNOWBALL_SOURCE_HASHES"]).items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    print("SNOWBALL_SOURCE_VERIFIED", os.environ["SNOWBALL_MARIN_COMMIT"], MSR_COMMIT, flush=True)
    training, config, report = resolved()
    assert report["request_hash"] == os.environ["SNOWBALL_REQUEST_HASH"]
    root = training.path(PREFIX)
    selection_path = training.deps[1].path(PREFIX) + "/selection.json"
    guard_absent(config.request.output.terminal_manifest_uri, root + "/receipts/*.json", selection_path)
    print("SNOWBALL_ABSENCE_GUARD_PASS", root, selection_path, flush=True)
    envelope = {
        "schema_version": 2,
        "request": asdict(config.request),
        "execution": {
            **asdict(config.execution),
            "job_name": sanitize_job_name(f"{config.request.run_id}-{config.request.attempt_id}"),
        },
    }
    print("SNOWBALL_RESOLVED_REQUEST", json.dumps(report, sort_keys=True), flush=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as stream:
        json.dump(envelope, stream)
        stream.flush()
        subprocess.run([*_launcher_command(config.launcher_requirement, stream.name), "--dry-run"], check=True)
    print("SNOWBALL_NATIVE_LAUNCHER_DRY_RUN_PASS", flush=True)
    run(*training.deps, max_concurrent=2)
    selection = json.loads(StoragePath(selection_path).read_text())
    assert selection["rows"]["train"] == [f"train/{i}" for i in range(128)]
    assert selection["rows"]["test"] == [f"test/{i}" for i in range(128)]
    print("SNOWBALL_DATA_SELECTION_PASS", flush=True)
    with wandb.init(
        project="marin-async-non-agentic-rl",
        entity="dogml",
        job_type="launch-preflight",
        name=config.request.run_id + "-preflight",
    ) as preflight:
        assert preflight.entity == "dogml"
        print("SNOWBALL_WANDB_PREFLIGHT_PASS", preflight.url, flush=True)
    result = run(training, max_concurrent=2)[0]
    assert isinstance(result, SkyRLTrainingResult) and not isinstance(result, SkyRLCheckpoint)
    assert result.global_step == 5
    receipt = json.loads(StoragePath(result.receipt_uri).read_text())
    assert receipt["completion_mode"] == "metrics" and "checkpoint" not in receipt
    print("SNOWBALL_METRICS_RESULT", result.model_dump_json(), flush=True)
    print("SNOWBALL_METRICS_RECEIPT", json.dumps(receipt, sort_keys=True), flush=True)
    print("SNOWBALL_COMPLETED_TRACE_AUDIT_PENDING", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinate", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--marin-commit")
    args = parser.parse_args()
    if args.coordinate:
        assert not args.execute
        coordinate()
        return
    assert args.marin_commit and len(args.marin_commit) == 40
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == args.marin_commit
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True)
    _, _, report = resolved()
    hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in SOURCES}
    report.update(marin=args.marin_commit, msr=MSR_COMMIT, source_hashes=hashes, coordinator_job_name=JOB_NAME)
    print(json.dumps(report, indent=2), flush=True)
    if not args.execute:
        return

    assert os.environ["IRIS_USER"] == "atqamar"
    environment = add_standard_env_vars(load_env_vars([]))
    environment.update(
        MARIN_PREFIX=PREFIX,
        MARIN_CLUSTER="coreweave",
        IRIS_USER="atqamar",
        WANDB_ENTITY="dogml",
        SNOWBALL_SOURCE_HASHES=json.dumps(hashes),
        SNOWBALL_REQUEST_HASH=report["request_hash"],
        SNOWBALL_MARIN_COMMIT=args.marin_commit,
    )
    command = (
        'export PYTHONPATH="$PWD/lib/marin/src:$PWD/lib/iris/src:$PWD"; '
        "exec python -m experiments.post_training.launch_snowball_publication_gate --coordinate"
    )
    with open_iris_client(
        config_file=Path("lib/iris/config/marin.yaml"), cluster_name="marin", workspace=Path.cwd()
    ) as client:
        job = client.submit(
            entrypoint=Entrypoint.from_command("bash", "-c", command),
            name=JOB_NAME,
            resources=build_resources(None, None, cpu=4, memory="16GB", disk="32GB"),
            environment=EnvironmentSpec(env_vars=environment, extras=[], setup_scripts=None),
            constraints=[Constraint.create(key=CLUSTER_CONSTRAINT_KEY, op=ConstraintOp.EQ, value="cw-us-east-02a")],
            max_retries_failure=0,
            max_retries_preemption=0,
            max_task_failures=0,
            timeout=Duration.from_seconds(2100),
            scheduling_timeout=Duration.from_seconds(300),
            priority_band=job_pb2.PRIORITY_BAND_BATCH,
        )
        print("SNOWBALL_COORDINATOR_SUBMITTED", job.job_id, flush=True)


if __name__ == "__main__":
    main()
