# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve and guard the isolated matched Qwen weight-sync gate before launch."""

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

from experiments.post_training import async_rl as recipe

PREFIX = "s3://marin-us-east-02a/marin"
SOURCES = (
    "uv.lock",
    "pyproject.toml",
    "config/external/MarinSkyRL/pyproject.toml",
    "config/external/MarinSkyRL/uv.lock",
    "experiments/post_training/async_rl.py",
    "experiments/post_training/launch_qwen_weight_sync_gate.py",
    "lib/marin/src/marin/external_dependencies.py",
    "lib/marin/src/marin/rl/skyrl.py",
    "lib/marin/src/marin/execution/lazy.py",
)


def resolved(version, expected_msr_commit):
    root = Path(__file__).resolve().parents[2]
    assert Path(dependency.__file__).resolve().is_relative_to(root)
    assert Path(recipe.__file__).resolve().is_relative_to(root)
    assert os.environ["MARIN_PREFIX"] == PREFIX
    recipe.validate_regional_storage(PREFIX, "cw-us-east-02a")
    training, evaluation = recipe.build_experiment(
        version=version,
        runner=recipe.Runner.ASYNC,
        scale=recipe.Scale.SCREENING,
        cluster="cw-us-east-02a",
        completion="metrics",
        timeout_seconds=900,
        inference_replicas=8,
        publication_stage_timing=True,
        dataloader_workers=0,
        validation_rows=128,
        response_tokens=1024,
        eval_response_tokens=1024,
        context_tokens=2048,
        initial_eval_repeat_count=1,
        weight_sync_interval=1,
        staleness=1,
        eval_interval=20,
        screening_steps=20,
        epoch_seeded_shuffle=True,
        kl_loss=False,
        seed=17,
        correction=recipe.Correction.BEHAVIOR_CLIP,
    )
    assert evaluation is None
    config = materialized_config(training, PREFIX)
    data = training.deps[1]
    data_config = materialized_config(data, PREFIX)
    request = config.request
    settings = yaml.safe_load(request.config_yaml)
    assert request.runtime.commit == expected_msr_commit and request.runtime.profile == "megatron"
    assert request.completion_mode == "metrics" and request.seed == 17
    assert data_config.train_rows == 1024 and data_config.validation_rows == 128
    assert settings["data"]["num_workers"] == 0 and settings["data"]["epoch_seeded_shuffle"]
    assert settings["trainer"]["max_steps"] == settings["trainer"]["eval_interval"] == 20
    assert settings["trainer"]["eval_before_train"]
    assert settings["trainer"].get("initial_eval_repeat_count", 1) == 1
    assert settings["trainer"]["train_batch_size"] == settings["trainer"]["policy_mini_batch_size"] == 64
    assert settings["trainer"]["fully_async"].get("weight_sync_interval", 1) == 1
    assert settings["trainer"]["fully_async"]["max_staleness_steps"] == 1
    assert settings["generator"]["publication_stage_timing"]
    assert settings["generator"]["inference_stats_poll_seconds"] == 1.0
    assert settings["generator"]["num_inference_engines"] == 8
    assert settings["generator"]["n_samples_per_prompt"] == 4
    assert settings["trainer"]["fully_async"]["num_parallel_generation_workers"] == 64
    assert settings["trainer"]["algorithm"]["policy_loss_type"] == "behavior_clip"
    assert not settings["trainer"]["algorithm"]["use_tis"]
    assert not settings["trainer"]["algorithm"]["use_kl_loss"]
    assert not settings["trainer"]["algorithm"]["use_kl_in_reward"]
    assert settings["context_budget"]["max_new_tokens_per_turn"] == 1024
    assert settings["context_budget"]["request_window_tokens"] == 2048
    assert settings["generator"]["eval_sampling_params"]["max_generate_length"] == 1024
    assert request.topology.num_nodes == 2 and request.topology.gpus_per_node == 8
    assert request.topology.role_plan.policy_num_nodes == 1 and not request.topology.role_plan.colocate_all
    assert config.execution.timeout_seconds == 900 and config.execution.max_retries == 0
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
    for name, digest in json.loads(os.environ["QWEN_WEIGHT_SYNC_SOURCE_HASHES"]).items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    expected_msr = os.environ["QWEN_WEIGHT_SYNC_MSR_COMMIT"]
    print("QWEN_WEIGHT_SYNC_SOURCE_VERIFIED", os.environ["QWEN_WEIGHT_SYNC_MARIN_COMMIT"], expected_msr, flush=True)
    training, config, report = resolved(os.environ["QWEN_WEIGHT_SYNC_VERSION"], expected_msr)
    assert report["request_hash"] == os.environ["QWEN_WEIGHT_SYNC_REQUEST_HASH"]
    root = training.path(PREFIX)
    selection_path = training.deps[1].path(PREFIX) + "/selection.json"
    guard_absent(config.request.output.terminal_manifest_uri, root + "/receipts/*.json", selection_path)
    print("QWEN_WEIGHT_SYNC_ABSENCE_GUARD_PASS", root, selection_path, flush=True)
    envelope = {
        "schema_version": 2,
        "request": asdict(config.request),
        "execution": {
            **asdict(config.execution),
            "job_name": sanitize_job_name(f"{config.request.run_id}-{config.request.attempt_id}"),
        },
    }
    print("QWEN_WEIGHT_SYNC_RESOLVED_REQUEST", json.dumps(report, sort_keys=True), flush=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as stream:
        json.dump(envelope, stream)
        stream.flush()
        subprocess.run([*_launcher_command(config.launcher_requirement, stream.name), "--dry-run"], check=True)
    print("QWEN_WEIGHT_SYNC_NATIVE_LAUNCHER_DRY_RUN_PASS", flush=True)
    run(*training.deps, max_concurrent=2)
    selection = json.loads(StoragePath(selection_path).read_text())
    assert selection["rows"]["train"] == [f"train/{i}" for i in range(1024)]
    assert selection["rows"]["test"] == [f"test/{i}" for i in range(128)]
    print("QWEN_WEIGHT_SYNC_DATA_SELECTION_PASS", flush=True)
    with wandb.init(
        project="marin-async-non-agentic-rl",
        entity="dogml",
        job_type="launch-preflight",
        name=config.request.run_id + "-preflight",
    ) as preflight:
        assert preflight.entity == "dogml"
        print("QWEN_WEIGHT_SYNC_WANDB_PREFLIGHT_PASS", preflight.url, flush=True)
    result = run(training, max_concurrent=2)[0]
    assert isinstance(result, SkyRLTrainingResult) and not isinstance(result, SkyRLCheckpoint)
    assert result.global_step == 20
    receipt = json.loads(StoragePath(result.receipt_uri).read_text())
    assert receipt["completion_mode"] == "metrics" and "checkpoint" not in receipt
    print("QWEN_WEIGHT_SYNC_METRICS_RESULT", result.model_dump_json(), flush=True)
    print("QWEN_WEIGHT_SYNC_METRICS_RECEIPT", json.dumps(receipt, sort_keys=True), flush=True)
    print("QWEN_WEIGHT_SYNC_COMPLETED_TRACE_AUDIT_PENDING", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinate", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--marin-commit")
    parser.add_argument("--expected-msr-commit")
    parser.add_argument("--version")
    parser.add_argument("--job-name")
    args = parser.parse_args()
    if args.coordinate:
        assert not args.execute
        coordinate()
        return
    assert args.marin_commit and len(args.marin_commit) == 40
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == args.marin_commit
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True)
    assert args.expected_msr_commit and len(args.expected_msr_commit) == 40
    assert args.version and args.job_name
    _, _, report = resolved(args.version, args.expected_msr_commit)
    hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in SOURCES}
    report.update(
        marin=args.marin_commit, msr=args.expected_msr_commit, source_hashes=hashes, coordinator_job_name=args.job_name
    )
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
        QWEN_WEIGHT_SYNC_SOURCE_HASHES=json.dumps(hashes),
        QWEN_WEIGHT_SYNC_REQUEST_HASH=report["request_hash"],
        QWEN_WEIGHT_SYNC_MARIN_COMMIT=args.marin_commit,
        QWEN_WEIGHT_SYNC_MSR_COMMIT=args.expected_msr_commit,
        QWEN_WEIGHT_SYNC_VERSION=args.version,
    )
    command = (
        'export PYTHONPATH="$PWD/lib/marin/src:$PWD/lib/iris/src:$PWD"; '
        "exec python -m experiments.post_training.launch_qwen_weight_sync_gate --coordinate"
    )
    with open_iris_client(
        config_file=Path("lib/iris/config/marin.yaml"), cluster_name="marin", workspace=Path.cwd()
    ) as client:
        job = client.submit(
            entrypoint=Entrypoint.from_command("bash", "-c", command),
            name=args.job_name,
            resources=build_resources(None, None, cpu=4, memory="16GB", disk="32GB"),
            environment=EnvironmentSpec(env_vars=environment, extras=[], setup_scripts=None),
            constraints=[Constraint.create(key=CLUSTER_CONSTRAINT_KEY, op=ConstraintOp.EQ, value="cw-us-east-02a")],
            max_retries_failure=0,
            max_retries_preemption=0,
            max_task_failures=0,
            timeout=Duration.from_seconds(1800),
            scheduling_timeout=Duration.from_seconds(300),
            priority_band=job_pb2.PRIORITY_BAND_BATCH,
        )
        print("QWEN_WEIGHT_SYNC_COORDINATOR_SUBMITTED", job.job_id, flush=True)


if __name__ == "__main__":
    main()
