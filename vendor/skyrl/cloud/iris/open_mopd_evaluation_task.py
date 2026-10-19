"""Run the pinned Open-MOPD evaluation against a released model or training checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from fsspec.spec import AbstractFileSystem

from cloud.iris.artifacts import FileEntry, copy_file_inventory, copy_tree, file_inventory, fs_and_path
from cloud.iris.open_mopd_evaluation import (
    GATES,
    SMOKE_MAX_TOKENS,
    EvaluationBenchmark,
    EvaluationConfig,
    benchmark_coverage,
    evaluation_scale,
    load_evaluation_config,
    validate_checkpoint_source,
    validate_model_source,
)
from cloud.iris.open_mopd_fidelity import DatasetArtifact, FidelityConfig, gpu_count, load_config, validate_output_uri
from cloud.iris.open_mopd_fidelity_task import (
    ArtifactVerification,
    FileVerification,
    LATEST_CHECKPOINT_NAME,
    checkout_source,
    periodic_sync,
    reject_existing_output,
    runtime_inventory,
    snapshot_model,
    sync_tree,
    validate_runtime,
)
from cloud.iris.open_mopd_vllm_rollout import evaluation_port_seed
from marinskyrl.hf_model import (
    TOKENIZER_CONFIG_NAME,
    normalize_fast_tokenizer_metadata,
    validate_portable_hf_model_files,
)
from skyrl_train.hf_model_io import verify_hf_model_export

ROLLOUT_WRAPPER = Path(__file__).with_name("open_mopd_vllm_rollout.py")
FSDP_CONFIG_NAME = "fsdp_config.json"
HUGGINGFACE_METADATA_DIRECTORY = "huggingface"
FSDP_MODEL_SHARD_PATTERN = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt")


@dataclass(frozen=True)
class StagedBenchmark:
    benchmark: EvaluationBenchmark
    path: Path
    verification: ArtifactVerification


@dataclass(frozen=True)
class CheckpointFile:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class CheckpointVerification:
    checkpoint_uri: str
    step: int
    committed_through_step: int
    files: tuple[CheckpointFile, ...]


@dataclass(frozen=True)
class NativeModelVerification:
    source_uri: str
    source_identity: str
    files: tuple[CheckpointFile, ...]
    staged_tokenizer_config: CheckpointFile | None = None


@dataclass(frozen=True)
class EvaluationInputs:
    source: Path
    model: Path
    benchmarks: tuple[StagedBenchmark, ...]
    model_verification: ArtifactVerification | CheckpointVerification | NativeModelVerification


@dataclass(frozen=True)
class RolloutCommand:
    domain: str
    argv: tuple[str, ...]
    log_path: Path


class EvaluationCommandError(RuntimeError):
    """A child evaluation command failed after writing its durable log."""

    def __init__(self, command: list[str], returncode: int, log_path: Path):
        self.command = tuple(command)
        self.returncode = returncode
        self.log_path = log_path
        super().__init__(f"Evaluation command exited with code {returncode}; see {log_path}")


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def run_logged_command(command: list[str], *, cwd: Path, log_path: Path) -> None:
    """Run a child process while preserving its merged stdout and stderr."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        with subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process:
            if process.stdout is None:
                raise RuntimeError("Failed to capture evaluation command output")
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            returncode = process.wait()
    if returncode:
        raise EvaluationCommandError(command, returncode, log_path)


def verify_benchmark_file(benchmark: EvaluationBenchmark, path: Path) -> FileVerification:
    """Verify one downloaded benchmark against its pinned LFS object."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Evaluation data for {benchmark.name} is missing or not a regular file")
    observed_size = path.stat().st_size
    if observed_size != benchmark.size:
        raise ValueError(
            f"Evaluation data integrity mismatch for {benchmark.name}: "
            f"expected {benchmark.size} bytes, found {observed_size}"
        )
    with path.open("rb") as source:
        observed_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    if observed_sha256 != benchmark.sha256:
        raise ValueError(
            f"Evaluation data integrity mismatch for {benchmark.name}: "
            f"expected SHA-256 {benchmark.sha256}, found {observed_sha256}"
        )
    return FileVerification(
        path=benchmark.path,
        expected_size=benchmark.size,
        observed_size=observed_size,
        expected_sha256=benchmark.sha256,
        observed_sha256=observed_sha256,
    )


def _download_benchmark(config: EvaluationConfig, benchmark: EvaluationBenchmark, destination: Path) -> StagedBenchmark:
    artifact = DatasetArtifact(
        repository=config.data.repository,
        revision=config.data.revision,
        path=benchmark.path,
        size=benchmark.size,
        sha256=benchmark.sha256,
    )
    code = (
        "from huggingface_hub import hf_hub_download; import sys; "
        "print(hf_hub_download(sys.argv[1], sys.argv[3], repo_type='dataset', revision=sys.argv[2], "
        "local_dir=sys.argv[4]))"
    )
    _run([sys.executable, "-c", code, artifact.repository, artifact.revision, artifact.path, str(destination)])
    path = destination / artifact.path
    file_verification = verify_benchmark_file(benchmark, path)
    verification = ArtifactVerification(
        repository=artifact.repository,
        revision=artifact.revision,
        files=(file_verification,),
    )
    return StagedBenchmark(benchmark=benchmark, path=path, verification=verification)


def _checkpoint_file(path: Path, relative: str) -> CheckpointFile:
    with path.open("rb") as source:
        sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    return CheckpointFile(path=relative, size=path.stat().st_size, sha256=sha256)


def _committed_checkpoint_step(
    filesystem: AbstractFileSystem,
    target: str,
    checkpoint_uri: str,
    requested_step: int,
) -> int:
    checkpoint_root = posixpath.dirname(posixpath.dirname(target))
    pointer = posixpath.join(checkpoint_root, LATEST_CHECKPOINT_NAME)
    if not filesystem.exists(pointer):
        raise ValueError(f"Checkpoint commit pointer is missing for {checkpoint_uri}")
    with filesystem.open(pointer, encoding="utf-8") as pointer_file:
        committed_value = pointer_file.read().strip()
    if not committed_value.isdigit() or int(committed_value) < requested_step:
        raise ValueError(f"Checkpoint step {requested_step} is not durably committed under {checkpoint_uri}")
    return int(committed_value)


def _checkpoint_model_inventory(filesystem: AbstractFileSystem, target: str) -> tuple[tuple[str, FileEntry], ...]:
    return tuple(
        (remote_path, entry)
        for remote_path, entry in file_inventory(filesystem, target)
        if entry.path == FSDP_CONFIG_NAME
        or entry.path.startswith(f"{HUGGINGFACE_METADATA_DIRECTORY}/")
        or FSDP_MODEL_SHARD_PATTERN.fullmatch(entry.path)
    )


def _validate_checkpoint_files(
    checkpoint_dir: Path,
    inventory: tuple[FileEntry, ...],
    checkpoint_uri: str,
) -> None:
    fsdp_config_path = checkpoint_dir / FSDP_CONFIG_NAME
    if not fsdp_config_path.is_file():
        raise ValueError(f"Checkpoint is missing {FSDP_CONFIG_NAME}: {checkpoint_uri}")
    fsdp_config = json.loads(fsdp_config_path.read_text())
    world_size = fsdp_config.get("world_size")
    if not isinstance(world_size, int) or world_size <= 0:
        raise ValueError(f"Checkpoint has an invalid FSDP world size: {checkpoint_uri}")
    expected_shards = {f"model_world_size_{world_size}_rank_{rank}.pt" for rank in range(world_size)}
    observed_shards = {item.path for item in inventory if FSDP_MODEL_SHARD_PATTERN.fullmatch(item.path)}
    if observed_shards != expected_shards:
        missing = sorted(expected_shards - observed_shards)
        extra = sorted(observed_shards - expected_shards)
        raise ValueError(
            f"Checkpoint model shards do not match world size {world_size}: missing={missing}, extra={extra}"
        )
    if not (checkpoint_dir / HUGGINGFACE_METADATA_DIRECTORY / "config.json").is_file():
        raise ValueError(f"Checkpoint is missing Hugging Face model metadata: {checkpoint_uri}")


def _merge_checkpoint(source: Path, checkpoint_dir: Path, model_dir: Path, checkpoint_uri: str) -> None:
    merger = source / "training" / "verl" / "scripts" / "legacy_model_merger.py"
    _run(
        [
            sys.executable,
            str(merger),
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            str(checkpoint_dir),
            "--target_dir",
            str(model_dir),
        ],
        cwd=source / "training" / "verl",
    )
    if not (model_dir / "config.json").is_file() or not tuple(model_dir.glob("*.safetensors")):
        raise ValueError(f"FSDP merger did not produce a loadable Hugging Face model for {checkpoint_uri}")


def stage_checkpoint_model(
    source: Path,
    checkpoint_uri: str,
    checkpoint_step: int,
    checkpoint_dir: Path,
    model_dir: Path,
) -> tuple[Path, CheckpointVerification]:
    """Download and merge one committed FSDP actor checkpoint."""
    checkpoint_uri = checkpoint_uri.rstrip("/")
    validate_checkpoint_source(checkpoint_uri, checkpoint_step)
    filesystem, target = fs_and_path(checkpoint_uri)
    committed_step = _committed_checkpoint_step(filesystem, target, checkpoint_uri, checkpoint_step)
    inventory = _checkpoint_model_inventory(filesystem, target)
    if not inventory:
        raise ValueError(f"No actor model files found under {checkpoint_uri}")
    copied = copy_file_inventory(filesystem, inventory, checkpoint_dir)
    _validate_checkpoint_files(checkpoint_dir, copied, checkpoint_uri)
    _merge_checkpoint(source, checkpoint_dir, model_dir, checkpoint_uri)
    downloaded = tuple(_checkpoint_file(checkpoint_dir / item.path, item.path) for item in copied)
    verification = CheckpointVerification(
        checkpoint_uri=checkpoint_uri,
        step=checkpoint_step,
        committed_through_step=committed_step,
        files=downloaded,
    )
    return model_dir, verification


def stage_native_model(source_uri: str, source_identity: str, model_dir: Path) -> tuple[Path, NativeModelVerification]:
    """Stage an exported native HF model and record exact downloaded file hashes."""
    copied = copy_tree(source_uri, model_dir)
    validate_portable_hf_model_files({item.path for item in copied}, source_uri)
    verify_hf_model_export(str(model_dir))
    downloaded = tuple(_checkpoint_file(model_dir / item.path, item.path) for item in copied)
    normalized = normalize_fast_tokenizer_metadata(model_dir)
    staged_tokenizer_config = (
        _checkpoint_file(model_dir / TOKENIZER_CONFIG_NAME, TOKENIZER_CONFIG_NAME) if normalized else None
    )
    return model_dir, NativeModelVerification(
        source_uri=source_uri,
        source_identity=source_identity,
        files=downloaded,
        staged_tokenizer_config=staged_tokenizer_config,
    )


def stage_evaluation_inputs(
    evaluation: EvaluationConfig,
    fidelity: FidelityConfig,
    root: Path,
    *,
    checkpoint_uri: str | None = None,
    checkpoint_step: int | None = None,
    model_export_uri: str | None = None,
    model_export_identity: str | None = None,
) -> EvaluationInputs:
    source = checkout_source(fidelity, root / "source")
    validate_checkpoint_source(checkpoint_uri, checkpoint_step)
    validate_model_source(checkpoint_uri, model_export_uri, model_export_identity)
    if model_export_uri is not None and model_export_identity is not None:
        model, model_verification = stage_native_model(model_export_uri, model_export_identity, root / "model")
    elif checkpoint_uri is None or checkpoint_step is None:
        model, model_verification = snapshot_model(
            fidelity.evaluation_reference.repository,
            fidelity.evaluation_reference.revision,
            fidelity.evaluation_reference.lfs_files,
            root / "model",
        )
    else:
        model, model_verification = stage_checkpoint_model(
            source,
            checkpoint_uri,
            checkpoint_step,
            root / "checkpoint",
            root / "model",
        )
    benchmarks = tuple(_download_benchmark(evaluation, benchmark, root / "data") for benchmark in evaluation.benchmarks)
    return EvaluationInputs(
        source=source,
        model=model,
        benchmarks=benchmarks,
        model_verification=model_verification,
    )


def rollout_commands(
    config: EvaluationConfig,
    inputs: EvaluationInputs,
    gate: str,
    output: Path,
    *,
    world_size: int,
    vllm_port_seed: int,
) -> tuple[RolloutCommand, ...]:
    if gate not in GATES:
        raise ValueError(f"Unknown evaluation gate: {gate}")
    commands = []
    for domain in ("math", "code", "if"):
        selected = [item for item in inputs.benchmarks if item.benchmark.domain == domain]
        protocol = selected[0].benchmark
        samples = 1 if gate == "smoke" else protocol.samples
        max_tokens = min(protocol.max_tokens, SMOKE_MAX_TOKENS) if gate == "smoke" else protocol.max_tokens
        command = [
            sys.executable,
            str(ROLLOUT_WRAPPER),
            "--model",
            str(inputs.model),
            "--input",
            *(str(item.path) for item in selected),
            "--output-dir",
            str(output / "rollouts"),
            "--vllm-port-seed",
            str(vllm_port_seed),
            "--tensor-parallel-size",
            "1",
            "--data-parallel-size",
            str(world_size),
            "--max-model-len",
            str(config.protocol.max_model_len),
            "--dtype",
            config.protocol.dtype,
            "--gpu-memory-utilization",
            str(protocol.gpu_memory_utilization),
            "--max-num-seqs",
            str(config.protocol.max_num_seqs),
            "--temperature",
            str(protocol.temperature),
            "--top-p",
            str(config.protocol.top_p),
            "--top-k",
            str(config.protocol.top_k),
            "--n",
            str(samples),
            "--max-tokens",
            str(max_tokens),
            "--stop-token-ids",
            ",".join(str(token_id) for token_id in config.protocol.stop_token_ids),
            "--trust-remote-code",
        ]
        if protocol.enable_thinking is not None:
            command.extend(["--enable-thinking", str(protocol.enable_thinking).lower()])
        if gate == "smoke":
            command.extend(["--base", "0", "--offset", "1"])
        commands.append(
            RolloutCommand(
                domain=domain,
                argv=tuple(command),
                log_path=output / "logs" / f"rollout-{domain}.log",
            )
        )
    return tuple(commands)


def score_released_benchmarks(config: EvaluationConfig, inputs: EvaluationInputs, output: Path) -> None:
    rollout_dir = output / "rollouts"
    data_dir = inputs.benchmarks[0].path.parents[2]
    for benchmark in config.benchmarks:
        if benchmark.score_mode != "released":
            continue
        rollouts = sorted(rollout_dir.glob(f"{benchmark.name}_rollouts_*.parquet"))
        if not rollouts:
            raise ValueError(f"No rollout files found for released scorer {benchmark.name}")
        command = [
            sys.executable,
            "-m",
            "evals.verifier.score",
            "--rollout",
            *(str(path) for path in rollouts),
            "--output",
            str(output / "scores" / f"{benchmark.name}.json"),
            "--data-dir",
            str(data_dir),
        ]
        run_logged_command(command, cwd=inputs.source, log_path=scorer_log_path(output, benchmark.name))


def scorer_log_path(output: Path, benchmark_name: str) -> Path:
    return output / "logs" / f"score-{benchmark_name}.log"


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fidelity-config", type=Path, required=True)
    parser.add_argument("--gate", choices=GATES, required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--gpu-slice", required=True)
    parser.add_argument("--task-image", required=True)
    parser.add_argument("--launcher-commit", required=True)
    parser.add_argument("--checkpoint-uri")
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--model-export-uri")
    parser.add_argument("--model-export-identity")
    parser.add_argument("--work-root", type=Path, default=Path("/tmp/open-mopd-final-eval"))
    parser.add_argument("--sync-interval", type=int, default=300)
    return parser


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    evaluation = load_evaluation_config(args.config)
    fidelity = load_config(args.fidelity_config)
    validate_output_uri(args.output_uri)
    validate_checkpoint_source(args.checkpoint_uri, args.checkpoint_step)
    validate_model_source(args.checkpoint_uri, args.model_export_uri, args.model_export_identity)
    validate_runtime(fidelity)
    world_size = gpu_count(args.gpu_slice)
    reject_existing_output(args.output_uri, "evaluation-manifest.json")
    if args.work_root.exists():
        raise ValueError(f"Work root already exists: {args.work_root}")
    output = args.work_root / "output"
    output.mkdir(parents=True)
    manifest_path = output / "evaluation-manifest.json"
    planned_completions, maximum_output_tokens = evaluation_scale(evaluation, args.gate)
    vllm_port_seed = evaluation_port_seed(args.output_uri)
    manifest: dict[str, object] = {
        "status": "staging",
        "gate": args.gate,
        "evaluation_config": asdict(evaluation),
        "fidelity_config": asdict(fidelity),
        "coverage": [asdict(item) for item in benchmark_coverage(evaluation, args.gate)],
        "planned_completions": planned_completions,
        "maximum_output_tokens": maximum_output_tokens,
        "task_image": args.task_image,
        "launcher_commit": args.launcher_commit,
        "gpu_slice": args.gpu_slice,
        "vllm_port_seed": vllm_port_seed,
        "checkpoint_uri": args.checkpoint_uri,
        "checkpoint_step": args.checkpoint_step,
        "model_export_uri": args.model_export_uri,
        "model_export_identity": args.model_export_identity,
    }
    _write_manifest(manifest_path, manifest)
    sync_tree(output, args.output_uri)
    stop = threading.Event()
    uploader = threading.Thread(
        target=periodic_sync,
        args=(output, args.output_uri, stop, args.sync_interval),
        daemon=True,
        name="open-mopd-evaluation-output-sync",
    )
    uploader.start()
    try:
        inputs = stage_evaluation_inputs(
            evaluation,
            fidelity,
            args.work_root,
            checkpoint_uri=args.checkpoint_uri,
            checkpoint_step=args.checkpoint_step,
            model_export_uri=args.model_export_uri,
            model_export_identity=args.model_export_identity,
        )
        commands = rollout_commands(
            evaluation,
            inputs,
            args.gate,
            output,
            world_size=world_size,
            vllm_port_seed=vllm_port_seed,
        )
        manifest.update(
            {
                "status": "running",
                "runtime": runtime_inventory(inputs.source),
                "model_verification": asdict(inputs.model_verification),
                "data_verifications": [asdict(item.verification) for item in inputs.benchmarks],
                "rollout_commands": [command.argv for command in commands],
                "command_logs": {
                    "rollouts": {command.domain: str(command.log_path.relative_to(output)) for command in commands},
                    "scorers": {
                        benchmark.name: str(scorer_log_path(output, benchmark.name).relative_to(output))
                        for benchmark in evaluation.benchmarks
                        if benchmark.score_mode == "released"
                    },
                },
            }
        )
        _write_manifest(manifest_path, manifest)
        sync_tree(output, args.output_uri)
        for command in commands:
            run_logged_command(
                list(command.argv),
                cwd=inputs.source,
                log_path=command.log_path,
            )
        score_released_benchmarks(evaluation, inputs, output)
        manifest["status"] = "complete"
        _write_manifest(manifest_path, manifest)
        return 0
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        _write_manifest(manifest_path, manifest)
        raise
    finally:
        stop.set()
        uploader.join(timeout=10)
        sync_tree(output, args.output_uri)


if __name__ == "__main__":
    sys.exit(main())
