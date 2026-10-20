# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only preparation and result reading in the frozen SkyRL dependency environment."""

import argparse
import asyncio
import hashlib
import json
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from cloud.iris.artifacts import file_inventory, fs_and_path, terminal_checkpoint_step
from cloud.iris.export_hf_checkpoint import ExportJobSpec, checkpoint_export_launch_config
from cloud.iris.launch_config import load_launch_config
from cloud.iris.runtime_bundle import resolve_launcher_source
from marinskyrl.model_manifest import MODEL_MANIFEST_FILENAME, ModelManifest
from marinskyrl.remote_io import filesystem_and_path
from marinskyrl.task_machine_factories import task_machine_factories
from omegaconf import DictConfig, OmegaConf
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.token_prefix import TokenPrefixHarness
from rolloutengine.machines import _machine_spec
from rolloutengine.models import TaskRolloutSpec
from skyrl_gym.rollout_specs import is_shell_harness, shell_harness
from skyrl_gym.verification import VerificationResult, VerificationStatus, normalized_verifier_score
from skyrl_train.dataset.tasks import TaskDataset
from skyrl_train.hf_export_schema import HFExportRequest
from skyrl_train.tokenizer import create_tokenizer
from taskcompendium.grading_result import Outcome
from taskcompendium.models import ScriptGrader
from taskcompendium.runtime.grading import grade_empty_in_sandbox
from transformers import PreTrainedTokenizerBase

from experiments.post_training.snowball_inventory_probe import package_revisions

TRAINING_RUNTIME_COMMIT = "b23bc74e3207e0f25458b341209bd26eb4942d5d"


def read_json(uri: str) -> dict:
    with StoragePath(uri).open("rt") as source:
        return json.load(source)


def load_document(text: str, *, expected_commit: str | None = None):
    with tempfile.TemporaryDirectory(prefix="snowball-eval-config-") as directory:
        path = Path(directory) / "launch.yaml"
        path.write_text(text)
        config = load_launch_config(path)
    if config.runtime.launcher_commit != (expected_commit or resolve_launcher_source().commit):
        raise ValueError("Composed runtime identity differs from the installed frozen runtime")
    return config


def verify_model(uri: str) -> dict:
    manifest = ModelManifest.from_mapping(read_json(uri + "/" + MODEL_MANIFEST_FILENAME), uri)
    filesystem, root = filesystem_and_path(uri)
    inventory = {entry.path: entry.size for _, entry in file_inventory(filesystem, root)}
    for entry in manifest.files:
        if inventory.get(entry.path) != entry.size:
            raise ValueError(f"Missing or differently sized model object: {entry.path}")
    index_bytes = StoragePath(uri + "/model.safetensors.index.json").read_bytes()
    entry = next(entry for entry in manifest.files if entry.path == "model.safetensors.index.json")
    if hashlib.sha256(index_bytes).hexdigest() != entry.sha256:
        raise ValueError("HF weight index differs from the model manifest")
    weight_map = json.loads(index_bytes)["weight_map"]
    names = {entry.path for entry in manifest.files}
    if not weight_map or not set(weight_map.values()) <= names:
        raise ValueError("HF weight index references missing manifested shards")
    return {
        "model_manifest_identity": manifest.identity,
        "weight_shards": len(set(weight_map.values())),
        "weight_validation": "Manifest identity/index digest and remote object sizes; no full weight rehash",
    }


async def grader_canaries(specs: dict[str, TaskRolloutSpec], factories: dict) -> list[dict]:
    checks = []
    # These complete serially: CPU admission never opens more than one grading sandbox.
    for family, spec in sorted(specs.items()):
        assert isinstance(spec.task.grader, ScriptGrader) and spec.verifier_machine is not None
        grade = await grade_empty_in_sandbox(
            spec.task,
            factories[spec.verifier_machine.backend],
            _machine_spec(spec.task.grader.environment, spec.verifier_machine),
            timeout=spec.limits.verifier_timeout,
        )
        if grade.status is not Outcome.GRADED or grade.reward != 0:
            raise ValueError(f"The {family} grader did not reject an empty submission with a usable grade")
        checks.append(
            {"evaluation_family": family, "task_id": spec.task.id, "status": grade.status, "reward": grade.reward}
        )
    return checks


def preflight_tokenizer(config: DictConfig) -> PreTrainedTokenizerBase:
    """Load pinned tokenizer metadata before the GPU task stages the model directory."""
    return create_tokenizer(
        model_path=str(config.inputs.model.tokenizer_uri),
        revision=str(config.inputs.model.tokenizer_revision),
        disable_fast_tokenizer=bool(config.skyrl.trainer.disable_fast_tokenizer),
    )


def preflight(request: dict) -> dict:
    config = load_document(request["launch_config_yaml"])
    if config.runtime.entrypoint != "skyrl_train.entrypoints.main_generate" or config.run.export_hf:
        raise ValueError("Evaluation requires the registered generate entrypoint without model export")
    if config.inputs.train_data or len(config.inputs.validation_data) != 1:
        raise ValueError("Evaluation requires only one frozen validation panel and no training inputs")
    # The source composer retains an unused HOME-based train default for empty inputs.
    config.skyrl.data.train_data = []
    revisions = package_revisions()
    model = verify_model(str(config.inputs.model.uri))
    tokenizer = preflight_tokenizer(config)
    sessions = config.skyrl.environment.task_sessions
    factories = task_machine_factories(sessions)
    token_prefix = TokenPrefixHarness(**dict(sessions.token_prefix))
    source = config.inputs.validation_data[0]
    uri = str(source.uri).rstrip("/") + "/" + str(source.relative_path)
    tasks = []
    counts = {}
    engines = {}
    canaries = {}
    with tempfile.TemporaryDirectory(prefix="snowball-eval-panel-") as directory:
        path = Path(directory) / "validation.parquet"
        with StoragePath(uri).open("rb") as source_file, path.open("wb") as target:
            while chunk := source_file.read(8 * 1024 * 1024):
                target.write(chunk)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != request["panel_sha256"]:
            raise ValueError("Evaluation panel SHA-256 mismatch")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            for row in batch.to_pylist():
                spec = TaskRolloutSpec.model_validate_json(row["task_rollout_spec"])
                if spec.task.id != row["task_id"]:
                    raise ValueError("Panel pairing task ID differs from the runtime contract")
                if spec.harness not in engines:
                    if spec.harness == "token_prefix":
                        harness = token_prefix
                    elif is_shell_harness(spec.harness):
                        harness = shell_harness(spec.harness)
                    else:
                        raise ValueError("Evaluation panel has an unsupported harness")
                    engines[spec.harness] = RolloutEngine(factories, {spec.harness: harness})
                engines[spec.harness].check(spec)
                if isinstance(spec.task.grader, ScriptGrader):
                    if spec.verifier_machine is None or spec.verifier_machine.backend != "daytona":
                        raise ValueError("Original script graders require their Daytona verifier backend")
                    canaries.setdefault(row["evaluation_family"], spec)
                tasks.append(
                    {
                        "task_id": row["task_id"],
                        "evaluation_family": row["evaluation_family"],
                        "contract_id": row["contract_id"],
                    }
                )
                family = row["evaluation_family"]
                counts[family] = counts.get(family, 0) + 1
        if counts != request["family_counts"] or len({row["task_id"] for row in tasks}) != len(tasks):
            raise ValueError("Evaluation panel family counts or unique pairing IDs differ")
        # Exercise the same dataset's prompt filtering and token restoration before allocating GPUs.
        dataset = TaskDataset([str(path)], tokenizer, int(config.skyrl.trainer.max_prompt_length), num_workers=1)
        if len(dataset) != len(tasks):
            raise ValueError("Evaluation dataset prompt filtering dropped panel rows")
        admitted = dataset.collate_fn(list(dataset))
        if [row["input"].spec.task.id for row in admitted] != [task["task_id"] for task in tasks]:
            raise ValueError("Evaluation dataset changed panel task ordering")
        dataset_uids = {row["uid"]: row["input"].spec.task.id for row in admitted}
        if len(dataset_uids) != len(tasks):
            raise ValueError("Evaluation dataset pairing UIDs must be unique")
    if set(canaries) != set(request["family_counts"]) - {"structural"}:
        raise ValueError("Every original script-grading family requires a CPU canary")
    checks = asyncio.run(grader_canaries(canaries, factories))
    return {
        "status": "passed",
        "launch_config_yaml": OmegaConf.to_yaml(config, resolve=True),
        "panel_sha256": digest,
        "tasks": tasks,
        "family_counts": counts,
        "engine_checks": len(tasks),
        "dataset_uids": dataset_uids,
        "grader_canaries": checks,
        "factory_backends": sorted(factories),
        "package_revisions": revisions,
        **model,
    }


def results(request: dict) -> dict:
    config = load_document(request["launch_config_yaml"])
    root = StoragePath(str(config.artifacts.export_root) + "/dumped_evals/eval_only")
    outcomes = []
    for directory, _, files in root.walk():
        for filename in files:
            if not filename.endswith(".jsonl") or filename == "aggregated_results.jsonl":
                continue
            with (directory / filename).open("rt") as source:
                for line in source:
                    row = json.loads(line)
                    verification = row.get("verification")
                    score = None
                    if verification is not None:
                        verification["status"] = VerificationStatus(verification["status"])
                        verdict = VerificationResult(**verification)
                        if verdict.status is VerificationStatus.VERIFIED:
                            score = normalized_verifier_score(verdict)
                    outcomes.append(
                        {
                            "task_id": request["dataset_uids"][row["uid"]],
                            "score": score,
                            "stop_reason": row.get("stop_reason"),
                        }
                    )
    return {"outcomes": outcomes}


def export_config(request: dict) -> dict:
    source_text = StoragePath(request["training_config_uri"]).read_text()
    raw = OmegaConf.to_container(OmegaConf.create(source_text), resolve=False)
    if isinstance(raw, dict) and "config" in raw:
        raw = raw["config"]
    training = load_document(OmegaConf.to_yaml(OmegaConf.create(raw)), expected_commit=TRAINING_RUNTIME_COMMIT)
    step = request["step"]
    checkpoint_root = str(training.artifacts.checkpoint_root)
    checkpoint_path = checkpoint_root + f"/global_step_{step}"
    if terminal_checkpoint_step(checkpoint_root) < step:
        raise ValueError("The selected step has not been committed")
    filesystem, path = fs_and_path(checkpoint_path)
    inventory = {entry.path: entry.size for _, entry in file_inventory(filesystem, path)}
    required = {"trainer_state.pt", "policy/.metadata", *(f"policy/__{rank}_0.distcp" for rank in range(32))}
    if any(inventory.get(name, 0) <= 0 for name in required):
        raise ValueError("Selected checkpoint is incomplete or has already been removed by retention")
    placement = training.skyrl.trainer.placement
    if int(placement.policy_num_nodes) != 4 or int(placement.policy_num_gpus_per_node) != 8:
        raise ValueError("This exporter requires the approved four-by-eight saved policy geometry")
    model = training.inputs.model
    destination = request["output_path"].rstrip("/")
    export_root = destination + "/exports"
    export_request = HFExportRequest(
        step=step,
        checkpoint_base_path=checkpoint_root,
        checkpoint_path=checkpoint_path,
        export_path=export_root,
        model_path=str(model.local_path),
        num_nodes=4,
        gpus_per_node=8,
        model_source_uri=str(model.uri),
        model_source_identity=str(model.identity),
    )
    spec = ExportJobSpec(
        request=export_request,
        cluster=str(training.iris.cluster),
        priority=str(training.iris.priority),
        gpu_variant=str(training.iris.allocation.gpu_variant),
        job_name=None,
        timeout=7200,
        no_wait=False,
        cluster_config=str(training.iris.cluster_config),
        target_cluster=training.iris.target_cluster,
        parent_cluster_config=training.iris.parent_cluster_config,
        cpu=32,
        memory="1800GB",
        disk="2TB",
        allocation_gpus_per_node=8,
    )
    config = checkpoint_export_launch_config(training, export_request, spec)
    config.runtime.launcher_commit = resolve_launcher_source().commit
    config.run.id = "snowball-evaluation-export-" + hashlib.sha256(destination.encode()).hexdigest()[:12]
    config.run.attempt_id = f"step-{step}"
    config.iris.job_name = f"{config.run.id}-step-{step}"
    config.artifacts.attempts_root = destination + "/attempts"
    config.artifacts.resolved_config_uri = destination + "/resolved-launch.yaml"
    config.artifacts.terminal_manifest_uri = destination + "/terminal.json"
    config.ray.rendezvous_dir = destination + "/attempts/rendezvous"
    config.ray.log_dir = destination + "/attempts/ray-logs"
    document = OmegaConf.to_yaml(config, resolve=True)
    load_document(document)
    return {
        "launch_config_yaml": document,
        "policy_export_uri": export_root + f"/global_step_{step}/policy",
        "source_config_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "source_runtime_commit": TRAINING_RUNTIME_COMMIT,
        "checkpoint_step": step,
        "checkpoint_checks": "Committed step marker and nonempty control/metadata/32 shards; no tensor rehash or resume",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "results", "export-config", "verify-model"))
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    operation = {
        "preflight": preflight,
        "results": results,
        "export-config": export_config,
        "verify-model": lambda value: verify_model(value["model_uri"]),
    }[args.action]
    args.result.write_text(json.dumps(operation(request), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
