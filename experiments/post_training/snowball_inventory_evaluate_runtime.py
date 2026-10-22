# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only preparation and result reading in the frozen SkyRL dependency environment."""

import argparse
import asyncio
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from cloud.iris.artifacts import file_inventory, fs_and_path, terminal_checkpoint_step
from cloud.iris.export_hf_checkpoint import ExportJobSpec, checkpoint_export_launch_config
from cloud.iris.launch_config import load_launch_config
from marinskyrl.model_manifest import MODEL_MANIFEST_FILENAME, ModelManifest, sha256_file
from marinskyrl.remote_io import filesystem_and_path
from marinskyrl.task_machine_factories import task_machine_factories
from omegaconf import DictConfig, OmegaConf
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.prefill import PrefillHarness
from rolloutengine.machines import _machine_spec
from rolloutengine.models import TaskRolloutSpec
from skyrl_gym.rollout_specs import is_shell_harness, shell_harness
from skyrl_gym.verification import VerificationResult, VerificationStatus, normalized_verifier_score
from skyrl_train.dataset.tasks import TaskDataset
from skyrl_train.hf_export_schema import HFExportRequest
from taskcompendium.grading_result import Outcome
from taskcompendium.models import ScriptGrader
from taskcompendium.runtime.grading import grade_empty_in_sandbox

from experiments.post_training.snowball_curriculum.export import DerivedDeployment, derived_record
from experiments.post_training.snowball_inventory_panel import evaluation_dataset_uid
from experiments.post_training.snowball_inventory_probe import (
    package_revisions,
    preflight_tokenizer,
    runtime_source_evidence,
)

REFERENCE_RUNTIME_LIMITS = {"model_turn_timeout": 1800.0, "total_turn_timeout": 2400.0, "attempt_timeout": 3600.0}


def prepare_reference_panel(
    source: Path, destination: Path, *, source_sha256: str, family_counts: dict[str, int]
) -> dict:
    """Derive only the outer deadlines of a local panel; retain every task and pairing field."""
    if destination.exists():
        raise FileExistsError(destination)
    if sha256_file(source) != source_sha256:
        raise ValueError("Reference source panel differs from its frozen hash")
    parquet = pq.ParquetFile(source)
    rows = parquet.read().to_pylist()
    counts = Counter(row["evaluation_family"] for row in rows)
    if counts != family_counts or len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError("Reference panel family counts or unique pairing IDs differ")
    mappings = []
    derived = []
    change = DerivedDeployment(
        packages_locks={},
        environment_provenance_sha256="",
        deadlines=REFERENCE_RUNTIME_LIMITS,
        verifier_startup_timeout=None,
    )
    for original in rows:
        row, _ = derived_record(original, change)
        spec = TaskRolloutSpec.model_validate_json(row["task_rollout_spec"])
        if spec.task.id != original["task_id"]:
            raise ValueError("Reference pairing ID differs from its task")
        mappings.append(
            {
                "task_id": original["task_id"],
                "evaluation_family": original["evaluation_family"],
                "contract_id": original["contract_id"],
                "source_contract_id": original["source_contract_id"],
                "source_runtime_sha256": hashlib.sha256(original["task_rollout_spec"].encode()).hexdigest(),
                "runtime_sha256": hashlib.sha256(row["task_rollout_spec"].encode()).hexdigest(),
            }
        )
        derived.append(row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(derived, schema=parquet.schema_arrow), destination, compression="zstd")
    reread = pq.ParquetFile(destination)
    if reread.schema_arrow != parquet.schema_arrow:
        raise ValueError("Derived reference panel schema changed")
    for original, written in zip(rows, reread.read().to_pylist(), strict=True):
        before = json.loads(original["task_rollout_spec"])
        after = json.loads(written["task_rollout_spec"])
        for key, value in REFERENCE_RUNTIME_LIMITS.items():
            if after["limits"][key] != value:
                raise ValueError("Derived reference deadline differs")
            after["limits"][key] = before["limits"][key]
        restored = {**written, "task_rollout_spec": original["task_rollout_spec"]}
        if after != before or restored != original:
            raise ValueError("Derived reference panel changed fields outside the deadline whitelist")
    if sha256_file(source) != source_sha256:
        raise ValueError("Frozen reference source changed during preparation")
    return {
        "status": "passed",
        "source_sha256": source_sha256,
        "sha256": sha256_file(destination),
        "row_count": len(rows),
        "family_counts": dict(counts),
        "runtime_limits": dict(REFERENCE_RUNTIME_LIMITS),
        "allowed_json_paths": [f"limits.{key}" for key in REFERENCE_RUNTIME_LIMITS],
        "all_other_fields_preserved": True,
        "runtime_mappings": mappings,
    }


def require_reference_limits(spec: TaskRolloutSpec) -> None:
    """Reject an 8k reference panel that still carries the shorter serialized deadlines."""
    actual = spec.limits.model_dump()
    if any(actual[key] != value for key, value in REFERENCE_RUNTIME_LIMITS.items()):
        raise ValueError("An 8192-token evaluation requires a frozen panel with 1800/2400/3600-second deadlines")


def evaluation_protocol(config: DictConfig) -> dict:
    """Read the actual resolved decoding fields used by the TASK evaluator."""
    generator = config.skyrl.generator
    return {
        "max_new_tokens": int(generator.eval_sampling_params.max_generate_length),
        "request_window_tokens": int(generator.engine_init_kwargs.max_model_len),
        "temperature": float(generator.eval_sampling_params.temperature),
        "samples": int(generator.eval_n_samples_per_prompt),
    }


def read_json(uri: str) -> dict:
    with StoragePath(uri).open("rt") as source:
        return json.load(source)


def load_document(text: str):
    with tempfile.TemporaryDirectory(prefix="snowball-eval-config-") as directory:
        path = Path(directory) / "launch.yaml"
        path.write_text(text)
        config = load_launch_config(path)
    return config


def verify_model(uri: str) -> dict:
    manifest = ModelManifest.from_mapping(read_json(str(StoragePath(uri) / MODEL_MANIFEST_FILENAME)), uri)
    filesystem, root = filesystem_and_path(uri)
    inventory = {entry.path: entry.size for _, entry in file_inventory(filesystem, root)}
    for entry in manifest.files:
        if inventory.get(entry.path) != entry.size:
            raise ValueError(f"Missing or differently sized model object: {entry.path}")
    index_bytes = (StoragePath(uri) / "model.safetensors.index.json").read_bytes()
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


def preflight(request: dict) -> dict:
    config = load_document(request["launch_config_yaml"])
    if config.runtime.entrypoint != "skyrl_train.entrypoints.main_generate" or config.run.export_hf:
        raise ValueError("Evaluation requires the registered generate entrypoint without model export")
    if config.inputs.train_data or len(config.inputs.validation_data) != 1:
        raise ValueError("Evaluation requires only one frozen validation panel and no training inputs")
    max_new_tokens = request.get("max_new_tokens", 4096)
    protocol = evaluation_protocol(config)
    expected_protocol = {
        "max_new_tokens": max_new_tokens,
        "request_window_tokens": 16384 + max_new_tokens,
        "temperature": 0.0,
        "samples": 1,
    }
    if protocol != expected_protocol:
        raise ValueError("Evaluation decoding budget differs from the declared protocol")
    # The source composer retains an unused HOME-based train default for empty inputs.
    config.skyrl.data.train_data = []
    revisions = package_revisions()
    model = verify_model(str(config.inputs.model.uri))
    tokenizer = preflight_tokenizer(config)
    sessions = config.skyrl.environment.task_sessions
    factories = task_machine_factories(sessions)
    prefill = PrefillHarness()
    source = config.inputs.validation_data[0]
    uri = str(StoragePath(str(source.uri)) / str(source.relative_path))
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
                if max_new_tokens == 8192:
                    require_reference_limits(spec)
                if spec.task.id != row["task_id"]:
                    raise ValueError("Panel pairing task ID differs from the runtime contract")
                if spec.harness not in engines:
                    if spec.harness == "prefill":
                        harness = prefill
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
                        "contract_id": row["source_contract_id"] if max_new_tokens == 8192 else row["contract_id"],
                    }
                )
                family = row["evaluation_family"]
                counts[family] = counts.get(family, 0) + 1
        if counts != request["family_counts"] or len({row["task_id"] for row in tasks}) != len(tasks):
            raise ValueError("Evaluation panel family counts or unique pairing IDs differ")
        # Exercise the same dataset's prompt filtering and token restoration before allocating GPUs.
        dataset = TaskDataset(
            [str(path)],
            tokenizer,
            int(config.skyrl.trainer.max_prompt_length),
            num_workers=1,
            chat_template_kwargs=dict(config.skyrl.generator.chat_template_kwargs),
        )
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
        "protocol": protocol,
        "pairing_contract_field": "source_contract_id" if max_new_tokens == 8192 else "contract_id",
        "launch_config_yaml": OmegaConf.to_yaml(config, resolve=True),
        "panel_sha256": digest,
        "tasks": tasks,
        "family_counts": counts,
        "engine_checks": len(tasks),
        "dataset_uids": dataset_uids,
        "grader_canaries": checks,
        "factory_backends": sorted(factories),
        "package_revisions": revisions,
        **runtime_source_evidence(),
        **model,
    }


def results(request: dict) -> dict:
    config = load_document(request["launch_config_yaml"])
    step = request.get("evaluation_step")
    session = "eval_only" if step is None else f"global_step_{step}_evals"
    root = StoragePath(str(config.artifacts.export_root)) / "dumped_evals" / session
    outcomes = []
    dumps = []
    for directory, _, files in sorted(root.walk(), key=lambda entry: str(entry[0])):
        for filename in sorted(files):
            if not filename.endswith(".jsonl") or filename == "aggregated_results.jsonl":
                continue
            source = directory / filename
            raw = source.read_bytes()
            dumps.append({"uri": str(source), "sha256": hashlib.sha256(raw).hexdigest()})
            for line in raw.decode().splitlines():
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
                        "task_id": request["dataset_uids"][evaluation_dataset_uid(row["uid"], step)],
                        "score": score,
                        "stop_reason": row.get("stop_reason"),
                        "verification_status": verification["status"].value if verification is not None else None,
                        "failure_type": (
                            row.get("exception_type")
                            or (
                                (verification.get("diagnostics") or {}).get("cause_error_type") if verification else None
                            )
                        ),
                    }
                )
    return {"outcomes": outcomes, "source_dumps": dumps}


def export_config(request: dict) -> dict:
    source_text = StoragePath(request["training_config_uri"]).read_text()
    raw = OmegaConf.to_container(OmegaConf.create(source_text), resolve=False)
    if isinstance(raw, dict) and "config" in raw:
        raw = raw["config"]
    training = load_document(OmegaConf.to_yaml(OmegaConf.create(raw)))
    step = request["step"]
    checkpoint_root = str(training.artifacts.checkpoint_root)
    checkpoint_path = str(StoragePath(checkpoint_root) / f"global_step_{step}")
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
    destination = StoragePath(request["output_path"])
    export_root = str(destination / "exports")
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
    config.run.id = "snowball-evaluation-export-" + hashlib.sha256(str(destination).encode()).hexdigest()[:12]
    config.run.attempt_id = f"step-{step}"
    config.iris.job_name = f"{config.run.id}-step-{step}"
    config.artifacts.attempts_root = str(destination / "attempts")
    config.artifacts.resolved_config_uri = str(destination / "resolved-launch.yaml")
    config.artifacts.terminal_manifest_uri = str(destination / "terminal.json")
    config.ray.rendezvous_dir = str(destination / "attempts/rendezvous")
    config.ray.log_dir = str(destination / "attempts/ray-logs")
    document = OmegaConf.to_yaml(config, resolve=True)
    load_document(document)
    return {
        "launch_config_yaml": document,
        "policy_export_uri": str(StoragePath(export_root) / f"global_step_{step}" / "policy"),
        "source_config_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        **runtime_source_evidence(),
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
