#!/usr/bin/env python
"""Export a banked training checkpoint to HF safetensors as an Iris job.

WHY THIS EXISTS
---------------
Normal training records ``hf_export_request.json`` beside each checkpoint selected by
``hf_save_interval``. It never gathers live weights for Hugging Face serialization. Pass
that checkpoint directory with ``--request`` to run the conversion on a separate Iris
gang. The request remains pending after failure and becomes complete only after Iris
reports success, so an interrupted or partial export is explicitly rerunnable.

Converting those checkpoints on a laptop is not practical for the Megatron strategy. It
writes a ``torch.distributed.checkpoint`` set (``__N_0.distcp`` + ``.metadata``) whose
tensors are Megatron-native — layer-stacked keys, grouped experts — and the conversion
to HF layout runs through ``bridge.save_hf_weights`` (mbridge), which needs the Megatron
runtime and a live process group at the saved parallel geometry.

The task enters ``skyrl_train.entrypoints.checkpoint_export`` instead of an RL entrypoint.
It creates only the saved policy worker group, restores only model tensors, invokes the
strategy's existing HF converter, and exits. It does not create rollout engines, datasets,
generators, tracking, reference models, critics, optimizers, or schedulers. ``--timeout``
belongs to this export job and is independent of the source training gang's process-group
timeout.

The export lands in ``export_path`` on durable storage. When the request carries the
training run's Hub destination, the export-only job publishes the completed artifact.
"""

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from cloud.iris.artifacts import terminal_checkpoint_step
from cloud.iris.launch_config import RunMode, SubmissionMode, load_launch_config
from cloud.iris.runtime_environment import CHECKPOINT_EXPORT_ENTRYPOINT
from cloud.iris.runtime_environment import RuntimeMode, runtime_profile_for_strategy
from marinskyrl.checkpoint_paths import GLOBAL_STEP_PREFIX, policy_export_path
from marinskyrl.resource_locator import ModelLocatorError
from marinskyrl.resource_locator import join_resource_path
from omegaconf import DictConfig, OmegaConf
from skyrl_train.hf_export_schema import (
    DEFAULT_HF_EXPORT_TIMEOUT,
    DEFAULT_HF_HUB_REVISION,
    DEFAULT_HF_UPLOAD_MODE,
    HFExportRequest,
    HFExportStatus,
    HFUploadMode,
)


def _read_hf_export_request(checkpoint_path: str) -> HFExportRequest | None:
    from skyrl_train.hf_export import read_hf_export_request  # noqa: PLC0415 - keep launcher imports Torch-free

    return read_hf_export_request(checkpoint_path)


def _write_hf_export_request(request: HFExportRequest) -> None:
    from skyrl_train.hf_export import write_hf_export_request  # noqa: PLC0415 - keep launcher imports Torch-free

    write_hf_export_request(request)


def _verify_hf_model_export(export_path: str) -> None:
    from skyrl_train.hf_model_io import verify_hf_model_export  # noqa: PLC0415 - keep launcher imports Torch-free

    verify_hf_model_export(export_path)


@dataclass(frozen=True)
class ExportJobSpec:
    request: HFExportRequest
    cluster: str
    priority: str
    gpu_variant: str
    job_name: str | None
    timeout: int
    no_wait: bool
    cluster_config: str | None = None
    target_cluster: str | None = None
    parent_cluster_config: str | None = None
    cpu: float | None = None
    memory: str | None = None
    disk: str | None = None
    allocation_gpus_per_node: int | None = None
    launch_config_path: str | None = None

    @property
    def allocated_gpus_per_node(self) -> int:
        allocation = self.allocation_gpus_per_node
        if allocation is None:
            allocation = self.request.gpus_per_node
        if allocation < self.request.gpus_per_node:
            raise ValueError("Export allocation cannot have fewer GPUs than the saved policy geometry")
        return allocation


def checkpoint_export_launch_config(
    training_config: DictConfig,
    request: HFExportRequest,
    spec: ExportJobSpec,
) -> DictConfig:
    """Derive the config-only checkpoint-export document from a training launch."""
    config = OmegaConf.create(OmegaConf.to_container(training_config, resolve=False))
    OmegaConf.set_struct(config, False)

    strategy = str(config.skyrl.trainer.strategy)
    config.run.mode = RunMode.CHECKPOINT_EXPORT.value
    config.run.export_hf = False
    config.run.submission = SubmissionMode.DETACH.value if spec.no_wait else SubmissionMode.WAIT.value
    config.run.attempt_id = f"{config.run.attempt_id}-export-{request.step}"
    config.runtime.entrypoint = CHECKPOINT_EXPORT_ENTRYPOINT
    config.runtime.training_type = None
    config.runtime.profile = runtime_profile_for_strategy(strategy, mode=RuntimeMode.CHECKPOINT_EXPORT).value
    config.iris.job_name = spec.job_name or f"{config.iris.job_name}-export-step-{request.step}"
    config.iris.cluster = spec.cluster
    if spec.cluster_config is not None:
        config.iris.cluster_config = spec.cluster_config
    config.iris.priority = spec.priority
    config.iris.max_retries = 0
    config.iris.timeout = spec.timeout
    config.iris.target_cluster = spec.target_cluster
    config.iris.parent_cluster_config = spec.parent_cluster_config
    config.iris.allocation.num_nodes = request.num_nodes
    config.iris.allocation.gpus_per_node = spec.allocated_gpus_per_node
    config.iris.allocation.gpu_variant = spec.gpu_variant
    if spec.cpu is not None:
        config.iris.allocation.cpu = spec.cpu
    if spec.memory is not None:
        config.iris.allocation.memory = spec.memory
    if spec.disk is not None:
        config.iris.allocation.disk = spec.disk
    config.artifacts.checkpoint_root = request.checkpoint_base_path
    config.artifacts.export_root = request.export_path
    export_attempt_root = join_resource_path(str(config.artifacts.attempts_root), f"export-{request.step}")
    config.artifacts.attempts_root = export_attempt_root
    config.artifacts.resolved_config_uri = join_resource_path(export_attempt_root, "resolved.yaml")
    config.artifacts.terminal_manifest_uri = join_resource_path(export_attempt_root, "terminal.json")

    # Export is policy-only. Keep the role-plan inputs internally consistent so
    # launch validation derives exactly the saved policy gang, without reserving
    # rollout/reference/critic/teacher bundles that the export entrypoint never uses.
    config.skyrl.trainer.placement.colocate_all = True
    config.skyrl.trainer.placement.colocate_policy_ref = True
    config.skyrl.trainer.offload_optimizer_during_rollouts = False
    algorithm = config.skyrl.trainer.setdefault("algorithm", {})
    algorithm["use_kl_loss"] = False
    algorithm["use_kl_in_reward"] = False
    critic = config.skyrl.trainer.setdefault("critic", {})
    critic["model"] = critic.get("model", {})
    critic["model"]["path"] = None
    config.skyrl.pop("teachers", None)
    config.skyrl.pop("teacher_routing", None)
    if config.skyrl.generator.get("speculative_decoding") is not None:
        config.skyrl.generator.speculative_decoding.training = None

    config.skyrl.trainer.placement.policy_num_nodes = request.num_nodes
    config.skyrl.trainer.placement.policy_num_gpus_per_node = request.gpus_per_node
    tensor_parallel_size = int(config.skyrl.generator.get("inference_engine_tensor_parallel_size", 1))
    pipeline_parallel_size = int(config.skyrl.generator.get("inference_engine_pipeline_parallel_size", 1))
    data_parallel_size = int(config.skyrl.generator.get("inference_engine_data_parallel_size", 1))
    engine_width = tensor_parallel_size * pipeline_parallel_size * data_parallel_size
    total_policy_gpus = request.num_nodes * request.gpus_per_node
    if total_policy_gpus % engine_width:
        raise ValueError("checkpoint export policy geometry must divide rollout engine parallelism")
    config.skyrl.generator.run_engines_locally = True
    config.skyrl.generator.num_inference_engines = total_policy_gpus // engine_width
    config.skyrl.trainer.policy.model.path = request.model_path
    if request.model_source_uri is not None:
        config.inputs.model.uri = request.model_source_uri
        config.inputs.model.identity = request.model_source_identity
        config.skyrl.trainer.policy.model.source_uri = request.model_source_uri
    if request.model_source_identity is not None:
        config.skyrl.trainer.policy.model.source_identity = request.model_source_identity
    config.skyrl.checkpoint_export = {
        "step": request.step,
        "checkpoint_path": request.checkpoint_path,
        "export_root": request.export_path,
        "hf_hub_repo_id": request.hf_hub_repo_id,
        "hf_hub_private": request.hf_hub_private,
        "hf_hub_revision": request.hf_hub_revision,
        "hf_upload_mode": request.hf_upload_mode.value,
    }
    return config


def write_checkpoint_export_config(
    training_config_path: Path,
    request: HFExportRequest,
    spec: ExportJobSpec,
) -> Path:
    """Write one temporary config-only export document for the Iris CLI boundary."""
    training_config = load_launch_config(training_config_path)
    config = checkpoint_export_launch_config(training_config, request, spec)
    digest = hashlib.sha256(training_config_path.read_bytes()).hexdigest()[:16]
    destination = Path(tempfile.gettempdir()) / "marinskyrl" / f"checkpoint-export-{request.step}-{digest}.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, destination, resolve=True)
    return destination


def build_command(spec: ExportJobSpec) -> list[str]:
    """Return the config-native command for an export-only run."""
    if spec.launch_config_path is None:
        raise ValueError("checkpoint export requires a generated launch config")
    return [
        sys.executable,
        "-m",
        "cloud.iris.launch",
        "iris",
        "launch",
        "--config",
        spec.launch_config_path,
    ]


def _with_checkpoint_export_config(
    training_config_path: Path,
    spec: ExportJobSpec,
) -> ExportJobSpec:
    launch_config_path = write_checkpoint_export_config(training_config_path, spec.request, spec)
    return replace(spec, launch_config_path=str(launch_config_path))


def argument_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--request", help="global_step_N checkpoint directory containing hf_export_request.json")
    ap.add_argument("--ckpt_path", help="checkpoint root holding global_step_N/")
    ap.add_argument("--step", type=int, help="checkpoint step to export")
    ap.add_argument(
        "--launch-config",
        required=True,
        help="the complete resolved launch document for the training run",
    )
    ap.add_argument("--model_path", help="base model path, as at training time")
    ap.add_argument("--model-source-uri", help="object-store model source for a task-local --model_path")
    ap.add_argument("--model-source-identity", help="immutable identity for --model-source-uri")
    ap.add_argument("--cluster", default="cw-rno2a")
    ap.add_argument("--cluster-config")
    ap.add_argument("--target-cluster")
    ap.add_argument("--parent-cluster-config")
    ap.add_argument("--cpu", type=float)
    ap.add_argument("--memory")
    ap.add_argument("--disk")
    ap.add_argument("--num-nodes", type=int)
    ap.add_argument("--gpus-per-node", type=int)
    ap.add_argument("--gpu-variant", required=True)
    ap.add_argument(
        "--allocation-gpus-per-node",
        type=int,
        help="reserve a whole-node GPU slice while retaining the policy rank count stored in --request",
    )
    ap.add_argument("--priority", default="batch")
    ap.add_argument("--export_path", help="defaults to <ckpt_path parent>/exports")
    ap.add_argument("--job-name", dest="job_name")
    ap.add_argument("--hf-hub-repo-id")
    ap.add_argument("--hf-hub-private", action="store_true", default=None)
    ap.add_argument("--hf-hub-revision")
    ap.add_argument("--hf-upload-mode", choices=tuple(HFUploadMode))
    ap.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_HF_EXPORT_TIMEOUT,
        help=(
            "export-job timeout in seconds, independent of the training process group "
            f"(default: {DEFAULT_HF_EXPORT_TIMEOUT})"
        ),
    )
    ap.add_argument(
        "--no-wait",
        action="store_true",
        help="submit a manual export and return immediately; request-driven exports must record completion",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the command and exit")
    return ap


def request_spec(args: argparse.Namespace, parser: argparse.ArgumentParser) -> ExportJobSpec:
    hydrated_options = {
        "--ckpt_path": args.ckpt_path,
        "--step": args.step,
        "--export_path": args.export_path,
        "--model_path": args.model_path,
        "--model-source-uri": args.model_source_uri,
        "--model-source-identity": args.model_source_identity,
        "--num-nodes": args.num_nodes,
        "--gpus-per-node": args.gpus_per_node,
        "--hf-hub-repo-id": args.hf_hub_repo_id,
        "--hf-hub-private": args.hf_hub_private,
        "--hf-hub-revision": args.hf_hub_revision,
        "--hf-upload-mode": args.hf_upload_mode,
    }
    conflicts = [name for name, value in hydrated_options.items() if value is not None]
    if conflicts:
        parser.error(f"--request cannot be combined with request-owned options: {', '.join(conflicts)}")
    if args.no_wait:
        parser.error("--no-wait cannot be used with --request because completion must be recorded")
    try:
        request = _read_hf_export_request(args.request)
    except ValueError as error:
        parser.error(f"invalid HF export request: {error}")
    if request is None:
        parser.error(f"no hf_export_request.json found under {args.request}")
    spec = operational_spec(args, request, no_wait=False)
    return _with_checkpoint_export_config(Path(args.launch_config), spec)


def operational_spec(args: argparse.Namespace, request: HFExportRequest, *, no_wait: bool) -> ExportJobSpec:
    return ExportJobSpec(
        request=request,
        cluster=args.cluster,
        priority=args.priority,
        gpu_variant=args.gpu_variant,
        job_name=args.job_name,
        timeout=args.timeout,
        no_wait=no_wait,
        cluster_config=args.cluster_config,
        target_cluster=args.target_cluster,
        parent_cluster_config=args.parent_cluster_config,
        cpu=args.cpu,
        memory=args.memory,
        disk=args.disk,
        allocation_gpus_per_node=args.allocation_gpus_per_node,
    )


def manual_spec(args: argparse.Namespace, parser: argparse.ArgumentParser) -> ExportJobSpec:
    required = {
        "--ckpt_path": args.ckpt_path,
        "--step": args.step,
        "--model_path": args.model_path,
        "--num-nodes": args.num_nodes,
        "--gpus-per-node": args.gpus_per_node,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"provide --request or all manual export options; missing {', '.join(missing)}")
    checkpoint_base_path = args.ckpt_path.rstrip("/")
    export_path = args.export_path or os.path.join(os.path.dirname(checkpoint_base_path), "exports")
    try:
        request = HFExportRequest(
            step=args.step,
            checkpoint_base_path=checkpoint_base_path,
            checkpoint_path=os.path.join(checkpoint_base_path, f"{GLOBAL_STEP_PREFIX}{args.step}"),
            export_path=export_path,
            model_path=args.model_path,
            num_nodes=args.num_nodes,
            gpus_per_node=args.gpus_per_node,
            model_source_uri=args.model_source_uri,
            model_source_identity=args.model_source_identity,
            hf_hub_repo_id=args.hf_hub_repo_id,
            hf_hub_private=bool(args.hf_hub_private),
            hf_hub_revision=args.hf_hub_revision or DEFAULT_HF_HUB_REVISION,
            hf_upload_mode=HFUploadMode(args.hf_upload_mode or DEFAULT_HF_UPLOAD_MODE),
        )
    except ModelLocatorError as error:
        parser.error(str(error))
    spec = operational_spec(args, request, no_wait=args.no_wait)
    return _with_checkpoint_export_config(Path(args.launch_config), spec)


def _run_export(spec: ExportJobSpec, command: list[str]) -> None:
    print(
        f"[export-hf] policy geometry {spec.request.num_nodes}x{spec.request.gpus_per_node} GPU ranks, "
        f"allocation {spec.request.num_nodes}x{spec.allocated_gpus_per_node} GPUs"
    )
    # The parent launcher's stdout is a single JSON response. A nested export
    # launcher has the same response format, so keep its output with the live
    # diagnostics instead of corrupting the parent's machine-readable stream.
    # Relative cluster paths must retain the parent launch's working directory.
    exit_code = subprocess.call(command, stdout=sys.stderr)
    if exit_code != 0:
        raise subprocess.CalledProcessError(exit_code, command)
    if not spec.no_wait:
        export_path = policy_export_path(spec.request.export_path, spec.request.step)
        _verify_hf_model_export(export_path)


def submit_requested_export(spec: ExportJobSpec, command: list[str]) -> None:
    """Submit one export job, verify synchronous output, and persist request state."""
    request = spec.request.with_status(HFExportStatus.IN_PROGRESS, timeout=spec.timeout, increment_attempts=True)
    _write_hf_export_request(request)
    try:
        _run_export(spec, command)
    except BaseException as error:
        exit_code = error.returncode if isinstance(error, subprocess.CalledProcessError) else 1
        _write_hf_export_request(request.with_status(HFExportStatus.PENDING, last_exit_code=exit_code))
        raise
    _write_hf_export_request(request.with_status(HFExportStatus.COMPLETE, last_exit_code=0))


def export_terminal_policy(training_config_path: Path) -> None:
    """Derive and run the terminal policy export from one training launch document."""
    training_config = load_launch_config(training_config_path)
    checkpoint_root = str(training_config.artifacts.checkpoint_root)
    step = terminal_checkpoint_step(checkpoint_root)
    checkpoint_path = join_resource_path(checkpoint_root, f"{GLOBAL_STEP_PREFIX}{step}")
    request = _read_hf_export_request(checkpoint_path)
    if request is None:
        raise ValueError(f"checkpoint {checkpoint_path} has no hf_export_request.json")
    if request.status is HFExportStatus.COMPLETE:
        return
    allocation = training_config.iris.allocation
    spec = ExportJobSpec(
        request=request,
        cluster=str(training_config.iris.cluster),
        priority=str(training_config.iris.priority),
        gpu_variant=str(allocation.gpu_variant),
        job_name=f"{training_config.iris.job_name}-export-{step}",
        timeout=DEFAULT_HF_EXPORT_TIMEOUT,
        no_wait=False,
        cluster_config=str(training_config.iris.cluster_config),
        target_cluster=training_config.iris.target_cluster,
        parent_cluster_config=training_config.iris.parent_cluster_config,
        cpu=float(allocation.cpu),
        memory=str(allocation.memory),
        disk=str(allocation.disk),
        allocation_gpus_per_node=int(allocation.gpus_per_node),
    )
    resolved_spec = _with_checkpoint_export_config(training_config_path, spec)
    submit_requested_export(resolved_spec, build_command(resolved_spec))


def main() -> None:
    parser = argument_parser()
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive for an export job")
    if args.request:
        spec = request_spec(args, parser)
        request = spec.request
    else:
        request = None
        spec = manual_spec(args, parser)

    if request is not None and request.status is HFExportStatus.COMPLETE:
        print(f"[export-hf] global_step_{request.step} is already complete")
        return

    cmd = build_command(spec)

    print("[export-hf] converting checkpoint step", spec.request.step)
    print("[export-hf]", " ".join(cmd))
    if args.dry_run:
        return
    if request is None:
        _run_export(spec, cmd)
    else:
        submit_requested_export(spec, cmd)


if __name__ == "__main__":
    main()
