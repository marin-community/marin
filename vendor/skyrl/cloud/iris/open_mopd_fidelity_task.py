"""Run the pinned authors' Open-MOPD control inside one Iris GPU task."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import posixpath
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from cloud.iris.artifacts import fs_and_path, read_json
from marinskyrl.resource_locator import relative_resource_path
from cloud.iris.open_mopd_fidelity import (
    CHECKPOINT_DIRECTORY_NAME,
    DOMAINS,
    GATES,
    GLOBAL_STEP_PREFIX,
    DatasetArtifact,
    FidelityConfig,
    LfsFile,
    gpu_count,
    load_config,
    validate_output_uri,
)
from open_mopd_versions import versions_match
from marinskyrl.resource_locator import join_resource_path

CONTROL_MANIFEST_NAME = "control-manifest.json"
LATEST_CHECKPOINT_NAME = "latest_checkpointed_iteration.txt"
STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"
HYDRA_REWARD_MODE_PATCH = "scripts/local/mt_opd.sh: declare the release-only rollout.reward_mode key"
RAW_PROMPT_RETENTION_PATCH = "verl/trainer/ppo/ray_trainer.py: retain raw_prompt for teacher retokenization"
DOMAIN_RESPONSE_LIMIT_PATCH = "verl rollout: apply paper per-domain response limits to each training request"


@dataclass(frozen=True)
class FileVerification:
    path: str
    expected_size: int
    observed_size: int
    expected_sha256: str
    observed_sha256: str


@dataclass(frozen=True)
class ArtifactVerification:
    repository: str
    revision: str
    files: tuple[FileVerification, ...]


@dataclass(frozen=True)
class StagedInputs:
    source: Path
    student: Path
    teachers: tuple[Path, ...]
    dataset: Path
    validation_dataset: Path
    artifact_verifications: tuple[ArtifactVerification, ...] = ()


def _run(
    command: list[str], *, cwd: Path | None = None, capture_output: bool = False
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=capture_output)


def validate_runtime(config: FidelityConfig) -> None:
    expected_python = config.environment.python
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != expected_python:
        raise ValueError(f"Python version mismatch: expected {expected_python}, found {actual_python}")
    for distribution, expected in config.environment.packages.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise ValueError(f"Required distribution is missing from the task image: {distribution}") from error
        if not versions_match(expected, actual):
            raise ValueError(f"{distribution} version mismatch: expected {expected}, found {actual}")


def checkout_source(config: FidelityConfig, destination: Path) -> Path:
    source = destination / "Open-MOPD"
    source.mkdir(parents=True)
    _run(["git", "init", "-q"], cwd=source)
    _run(["git", "remote", "add", "origin", config.source.repository], cwd=source)
    _run(["git", "fetch", "--depth=1", "origin", config.source.commit], cwd=source)
    _run(["git", "checkout", "-q", "--detach", "FETCH_HEAD"], cwd=source)
    actual = _run(["git", "rev-parse", "HEAD"], cwd=source, capture_output=True).stdout.strip()
    if actual != config.source.commit:
        raise ValueError(f"Open-MOPD checkout mismatch: expected {config.source.commit}, found {actual}")
    return source


def replace_source_contract(path: Path, old: str, new: str, description: str) -> None:
    text = path.read_text()
    if text.count(old) != 1 or new in text:
        raise ValueError(f"Pinned Open-MOPD source no longer matches the expected {description}")
    path.write_text(text.replace(old, new))


def patch_source_for_reference(source: Path) -> tuple[str, ...]:
    """Apply fail-closed compatibility and paper-protocol fixes to the pinned source."""
    replace_source_contract(
        source / "scripts" / "local" / "mt_opd.sh",
        '"actor_rollout_ref.rollout.reward_mode=mt_opd"',
        '"+actor_rollout_ref.rollout.reward_mode=mt_opd"',
        "reward_mode assignment",
    )
    replace_source_contract(
        source / "training" / "verl" / "verl" / "trainer" / "ppo" / "ray_trainer.py",
        '{"data_source", "reward_model", "extra_info", "uid", "domain"} & batch.non_tensor_batch.keys()',
        '{"data_source", "reward_model", "extra_info", "uid", "domain", "raw_prompt"}\n'
        "            & batch.non_tensor_batch.keys()",
        "reward-model metadata retention set",
    )
    replace_source_contract(
        source / "training" / "verl" / "verl" / "trainer" / "ppo" / "ray_trainer.py",
        "        if self.async_rollout_mode:\n            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)",
        "        if self.async_rollout_mode:\n"
        "            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)\n"
        '        elif "domain" in batch.non_tensor_batch:\n'
        '            gen_batch.non_tensor_batch["domain"] = batch.non_tensor_batch["domain"].copy()',
        "synchronous rollout domain metadata",
    )
    replace_source_contract(
        source / "training" / "verl" / "verl" / "workers" / "config" / "rollout.py",
        "    response_length: int = 512\n",
        "    response_length: int = 512\n    domain_response_limits: dict[str, int] = field(default_factory=dict)\n",
        "typed rollout per-domain response limits",
    )
    replace_source_contract(
        source / "training" / "verl" / "verl" / "workers" / "rollout" / "vllm_rollout" / "vllm_rollout_spmd.py",
        "        with self.update_sampling_params(**kwargs):\n            outputs = self.inference_engine.generate(",
        "        with self.update_sampling_params(**kwargs):\n"
        "            sampling_params = self.sampling_params\n"
        "            if not is_validate:\n"
        '                domains = non_tensor_batch.get("domain")\n'
        "                if domains is None or len(domains) != batch_size:\n"
        '                    raise ValueError("Per-domain response limits require one domain per training request")\n'
        "                limits = self.config.domain_response_limits\n"
        "                sampling_params = []\n"
        "                for domain in domains:\n"
        "                    if domain not in limits:\n"
        '                        raise ValueError(f"Unknown domain for response limit: {domain}")\n'
        "                    params = self.sampling_params.clone()\n"
        "                    params.max_tokens = int(limits[domain])\n"
        "                    sampling_params.append(params)\n"
        "            outputs = self.inference_engine.generate(",
        "training rollout per-request sampling parameters",
    )
    replace_source_contract(
        source / "training" / "verl" / "verl" / "workers" / "rollout" / "vllm_rollout" / "vllm_rollout_spmd.py",
        "                sampling_params=self.sampling_params,",
        "                sampling_params=sampling_params,",
        "training rollout sampling parameters",
    )
    return HYDRA_REWARD_MODE_PATCH, RAW_PROMPT_RETENTION_PATCH, DOMAIN_RESPONSE_LIMIT_PATCH


def verify_lfs_files(destination: Path, expected_files: tuple[LfsFile, ...]) -> tuple[FileVerification, ...]:
    """Verify downloaded model files against the pinned Hugging Face LFS manifest."""
    root = destination.resolve()
    failures = []
    verified = []
    for expected in expected_files:
        candidate = root / expected.path
        path = candidate.resolve()
        if not path.is_relative_to(root):
            failures.append(f"{expected.path}: resolves outside the model directory")
            continue
        if candidate.is_symlink() or not path.is_file():
            failures.append(f"{expected.path}: missing or not a regular file")
            continue
        observed_size = path.stat().st_size
        if observed_size != expected.size:
            failures.append(f"{expected.path}: expected {expected.size} bytes, found {observed_size}")
            continue
        with path.open("rb") as source:
            observed_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
        if observed_sha256 != expected.sha256:
            failures.append(f"{expected.path}: expected SHA-256 {expected.sha256}, found {observed_sha256}")
            continue
        verified.append(
            FileVerification(
                path=expected.path,
                expected_size=expected.size,
                observed_size=observed_size,
                expected_sha256=expected.sha256,
                observed_sha256=observed_sha256,
            )
        )
    if failures:
        raise ValueError("Model artifact integrity verification failed: " + "; ".join(failures))
    return tuple(verified)


def snapshot_model(
    repository: str, revision: str, expected_files: tuple[LfsFile, ...], destination: Path
) -> tuple[Path, ArtifactVerification]:
    code = (
        "from huggingface_hub import snapshot_download; import sys; "
        "print(snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3]))"
    )
    _run([sys.executable, "-c", code, repository, revision, str(destination)])
    files = verify_lfs_files(destination, expected_files)
    return destination, ArtifactVerification(repository=repository, revision=revision, files=files)


def verify_shared_tokenizer(student: Path, teachers: tuple[Path, ...]) -> None:
    """Fail closed before using student token IDs to query a teacher model."""
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        fingerprints = []
        for model in (student, *teachers):
            path = model / filename
            if not path.is_file():
                raise ValueError(f"missing tokenizer file: {path}")
            with path.open("rb") as source:
                fingerprints.append(hashlib.file_digest(source, "sha256").hexdigest())
        if len(set(fingerprints)) != 1:
            raise ValueError(f"student/teacher tokenizer mismatch in {filename}: {fingerprints}")


def _dataset(artifact: DatasetArtifact, destination: Path) -> tuple[Path, ArtifactVerification]:
    code = (
        "from huggingface_hub import hf_hub_download; import sys; "
        "print(hf_hub_download(sys.argv[1], sys.argv[3], repo_type='dataset', revision=sys.argv[2], "
        "local_dir=sys.argv[4]))"
    )
    _run(
        [
            sys.executable,
            "-c",
            code,
            artifact.repository,
            artifact.revision,
            artifact.path,
            str(destination),
        ]
    )
    path = destination / artifact.path
    observed_size = path.stat().st_size
    if observed_size != artifact.size:
        raise ValueError(f"Dataset size mismatch: expected {artifact.size} bytes, found {observed_size}")
    with path.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != artifact.sha256:
        raise ValueError(f"Dataset digest mismatch: expected {artifact.sha256}, found {actual}")
    verification = FileVerification(
        path=artifact.path,
        expected_size=artifact.size,
        observed_size=observed_size,
        expected_sha256=artifact.sha256,
        observed_sha256=actual,
    )
    return path, ArtifactVerification(
        repository=artifact.repository,
        revision=artifact.revision,
        files=(verification,),
    )


def stage_inputs(config: FidelityConfig, root: Path) -> StagedInputs:
    source = checkout_source(config, root / "source")
    student, student_verification = snapshot_model(
        config.student.repository,
        config.student.revision,
        config.student.lfs_files,
        root / "models" / "student",
    )
    teacher_snapshots = tuple(
        snapshot_model(
            teacher.repository,
            teacher.revision,
            teacher.lfs_files,
            root / "models" / teacher.domain,
        )
        for teacher in config.teachers
    )
    verify_shared_tokenizer(student, tuple(snapshot for snapshot, _ in teacher_snapshots))
    dataset, dataset_verification = _dataset(config.dataset, root / "data")
    validation_dataset, validation_verification = _dataset(config.validation_dataset, root / "validation")
    return StagedInputs(
        source=source,
        student=student,
        teachers=tuple(snapshot for snapshot, _ in teacher_snapshots),
        dataset=dataset,
        validation_dataset=validation_dataset,
        artifact_verifications=(
            student_verification,
            *(verification for _, verification in teacher_snapshots),
            dataset_verification,
            validation_verification,
        ),
    )


def training_command(
    config: FidelityConfig,
    inputs: StagedInputs,
    gate: str,
    output: Path,
    *,
    world_size: int,
) -> list[str]:
    training = config.training
    steps = GATES[gate]
    overrides = [
        "algorithm.adv_estimator=token_reward_direct",
        "algorithm.use_kl_in_reward=False",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={training.mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_epochs={training.ppo_epochs}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={training.micro_batch_size_per_gpu}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={training.micro_batch_size_per_gpu}",
        f"actor_rollout_ref.actor.optim.lr={training.learning_rate}",
        "actor_rollout_ref.actor.optim.lr_scheduler_type=constant",
        f"actor_rollout_ref.actor.optim.clip_grad={training.gradient_clip}",
        "actor_rollout_ref.actor.use_kl_loss=False",
        f"actor_rollout_ref.actor.entropy_coeff={training.entropy_coefficient}",
        f"actor_rollout_ref.actor.loss_agg_mode={training.loss_aggregation}",
        f"actor_rollout_ref.actor.clip_ratio_low={training.clip_low}",
        f"actor_rollout_ref.actor.clip_ratio_high={training.clip_high}",
        "+actor_rollout_ref.actor.opd_refresh_advantage=True",
        "+actor_rollout_ref.actor.opd_reward_weight_mode=student_p",
        f"+actor_rollout_ref.rollout.log_prob_top_k={training.top_k}",
        "+actor_rollout_ref.rollout.top_k_strategy=only_stu",
        "+actor_rollout_ref.rollout.reward_weight_mode=student_p",
        "+actor_rollout_ref.rollout.domain_response_limits={"
        + ",".join(f"{domain}:{limit}" for domain, limit in zip(DOMAINS, training.domain_response_limits, strict=True))
        + "}",
        "actor_rollout_ref.rollout.mode=sync",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={training.prompt_limit + training.response_limit}",
        f"actor_rollout_ref.rollout.temperature={training.temperature}",
        f"actor_rollout_ref.rollout.top_p={training.nucleus_p}",
        f"reward_model.micro_batch_size_per_gpu={training.micro_batch_size_per_gpu}",
        "reward_model.model.input_tokenizer=null",
        f"+reward_model.teacher_temperature={training.teacher_temperature}",
        "+reward_model.reward_kwargs.compute_true_reward=False",
        "+data.sampler.class_path=pkg://verl.utils.dataset.domain_weighted_sampler",
        "+data.sampler.class_name=DomainWeightedSampler",
        *(f"+data.domain_weights.{domain}={weight}" for domain, weight in zip(DOMAINS, training.domain_weights)),
        "data.return_raw_chat=True",
        "data.dataloader_num_workers=0",
        "+mt_opd.domain_weighting=domain_routing",
        "+mt_opd.target_share_domains=[math,code,if]",
        f"+mt_opd.target_share_values=[{','.join(str(value) for value in training.target_gradient_shares)}]",
        f"+mt_opd.normalize_reward_scale={training.gap_following_alpha}",
        "+mt_opd.reward_scale_stat=mean",
        "+mt_opd.reward_scale_direction=multiply",
        "+mt_opd.reward_scale_anchored=True",
        "+mt_opd.conflict_policy=none",
        f"trainer.total_training_steps={steps}",
        f"trainer.save_freq={min(training.save_every, steps)}",
        f"trainer.test_freq={min(training.eval_every, steps)}",
        "trainer.val_before_train=False",
        f"trainer.validation_data_dir={output / 'validation'}",
        f"actor_rollout_ref.rollout.val_kwargs.temperature={training.eval_temperature}",
        f"actor_rollout_ref.rollout.val_kwargs.top_p={training.eval_nucleus_p}",
        f"+actor_rollout_ref.rollout.val_kwargs.max_tokens={training.response_limit}",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
        f"actor_rollout_ref.rollout.val_kwargs.n={training.eval_samples}",
        "trainer.logger=['console']",
        "trainer.resume_mode=auto",
    ]
    command = [
        "bash",
        "scripts/local/mt_opd.sh",
        "--run",
        "--model",
        str(inputs.student),
        "--teacher",
        str(inputs.teachers[0]),
        "--teacher",
        str(inputs.teachers[1]),
        "--teacher",
        str(inputs.teachers[2]),
        "--domains",
        ",".join(DOMAINS),
        "--train",
        str(inputs.dataset),
        "--val",
        str(inputs.validation_dataset),
        "--output",
        str(output),
        "--checkpoint",
        str(output / CHECKPOINT_DIRECTORY_NAME),
        "--gpus",
        str(world_size),
    ]
    training_env = {
        "TRAIN_BATCH_SIZE": str(training.train_batch_size),
        "MAX_PROMPT_LENGTH": str(training.prompt_limit),
        "MAX_RESPONSE_LENGTH": str(training.response_limit),
        "N_RESPONSES": "1",
        "TOTAL_EPOCHS": "1",
    }
    for override in overrides:
        command.extend(["--extra", override])
    return ["env", *(f"{key}={value}" for key, value in training_env.items()), *command]


def _checkpoint_step(relative: Path) -> int | None:
    """Return the checkpoint step encoded by a payload path, if present."""
    if len(relative.parts) < 3 or relative.parts[0] != CHECKPOINT_DIRECTORY_NAME:
        return None
    step_directory = relative.parts[1]
    if not step_directory.startswith(GLOBAL_STEP_PREFIX):
        return None
    step = step_directory[len(GLOBAL_STEP_PREFIX) :]
    return int(step) if step.isdigit() else None


def sync_tree(local: Path, output_uri: str) -> None:
    filesystem, target = fs_and_path(output_uri)
    pointer = local / CHECKPOINT_DIRECTORY_NAME / LATEST_CHECKPOINT_NAME
    pointer_payload = None
    committed_step = None
    if pointer.is_file():
        pointer_payload = pointer.read_bytes()
        pointer_value = pointer_payload.decode().strip()
        if not pointer_value.isdigit():
            raise ValueError(f"Invalid local checkpoint iteration {pointer_value!r}")
        committed_step = int(pointer_value)
    sources = sorted((source for source in local.rglob("*") if source.is_file()), key=lambda path: path.as_posix())
    sources.sort(key=lambda path: path.name == LATEST_CHECKPOINT_NAME)
    source_entries = [
        (source, source.relative_to(local), _checkpoint_step(source.relative_to(local))) for source in sources
    ]
    upload_entries = [
        (source, relative, step)
        for source, relative, step in source_entries
        if (source != pointer or pointer_payload is not None)
        and (step is None or (committed_step is not None and step <= committed_step))
    ]
    checkpoint_sources = {source for source, _relative, step in upload_entries if step is not None}
    remote_sizes: dict[str, int] = {}
    if checkpoint_sources:
        remote_details = filesystem.find(target, detail=True, withdirs=False)
        remote_sizes = {
            relative_resource_path(target, remote_path): int(details["size"])
            for remote_path, details in remote_details.items()
        }
    for source, relative, _step in upload_entries:
        destination = posixpath.join(target, relative.as_posix())
        if source in checkpoint_sources and remote_sizes.get(relative.as_posix()) == source.stat().st_size:
            continue
        filesystem.makedirs(posixpath.dirname(destination), exist_ok=True)
        if source == pointer:
            assert pointer_payload is not None
            filesystem.pipe_file(destination, pointer_payload)
        else:
            filesystem.put_file(str(source), destination)


def reject_existing_output(output_uri: str, manifest_name: str = CONTROL_MANIFEST_NAME) -> None:
    """Reject an output prefix that already contains the named terminal artifact."""
    filesystem, target = fs_and_path(output_uri)
    if filesystem.exists(posixpath.join(target, manifest_name)):
        raise ValueError(f"Durable output already contains {manifest_name}: {output_uri}")


def validate_resume_manifest(existing: dict[str, object], expected: dict[str, object], output_uri: str) -> None:
    if existing.get("returncode") == 0:
        raise ValueError(f"Durable output already contains a completed control: {output_uri}")
    mismatches = [key for key, value in expected.items() if existing.get(key) != value]
    if mismatches:
        raise ValueError(f"Durable output is incompatible with this retry ({', '.join(mismatches)}): {output_uri}")


def restore_latest_checkpoint(output_uri: str, output: Path) -> int | None:
    """Restore the newest remotely committed checkpoint and return its step, or ``None`` when none exists."""
    filesystem, target = fs_and_path(output_uri)
    remote_files = tuple(filesystem.find(target))
    pointers = [path for path in remote_files if posixpath.basename(path) == LATEST_CHECKPOINT_NAME]
    if not pointers:
        return None
    if len(pointers) != 1:
        raise ValueError(f"Expected one {LATEST_CHECKPOINT_NAME} under {output_uri}, found {len(pointers)}")
    pointer = pointers[0]
    with filesystem.open(pointer, encoding="utf-8") as source:
        value = source.read().strip()
    if not value.isdigit():
        raise ValueError(f"Invalid checkpoint iteration {value!r} under {output_uri}")
    step = int(value)
    checkpoint_root = posixpath.dirname(pointer)
    step_root = posixpath.join(checkpoint_root, f"{GLOBAL_STEP_PREFIX}{step}")
    checkpoint_files = [path for path in remote_files if path.startswith(f"{step_root}/")]
    if not checkpoint_files:
        raise ValueError(f"Checkpoint pointer selects missing {GLOBAL_STEP_PREFIX}{step} under {output_uri}")
    for remote_path in [*sorted(checkpoint_files), pointer]:
        relative = relative_resource_path(target, remote_path)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        filesystem.get_file(remote_path, str(destination))
    return step


def periodic_sync(local: Path, output_uri: str, stop: threading.Event, interval: int) -> None:
    while not stop.wait(interval):
        try:
            sync_tree(local, output_uri)
        except Exception as error:
            print(f"Periodic output sync failed; retrying in {interval} seconds: {error}", file=sys.stderr)


def runtime_inventory(source: Path) -> dict[str, object]:
    return {
        "open_mopd_git_status": _run(["git", "status", "--porcelain"], cwd=source, capture_output=True).stdout,
        "python": sys.version,
        "pip_freeze": _run([sys.executable, "-m", "pip", "freeze"], capture_output=True).stdout.splitlines(),
        "nvidia_smi": _run(["nvidia-smi", "-q"], capture_output=True).stdout,
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gate", choices=tuple(GATES), required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--gpu-slice", required=True)
    parser.add_argument("--task-image", required=True)
    parser.add_argument("--launcher-commit", required=True)
    parser.add_argument("--work-root", type=Path, default=Path("/tmp/open-mopd-fidelity"))
    parser.add_argument("--sync-interval", type=int, default=300)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    config = load_config(args.config)
    validate_output_uri(args.output_uri)
    validate_runtime(config)
    world_size = gpu_count(args.gpu_slice)
    identity = {
        "config": json.loads(json.dumps(asdict(config))),
        "gate": args.gate,
        "steps": GATES[args.gate],
        "task_image": args.task_image,
        "launcher_commit": args.launcher_commit,
        "gpu_slice": args.gpu_slice,
    }
    existing_manifest = read_json(join_resource_path(args.output_uri, CONTROL_MANIFEST_NAME))
    if existing_manifest is not None:
        validate_resume_manifest(existing_manifest, identity, args.output_uri)
    if args.work_root.exists():
        raise ValueError(f"Work root already exists: {args.work_root}")
    args.work_root.mkdir(parents=True)
    output = args.work_root / "output"
    output.mkdir()
    resumed_from_step = restore_latest_checkpoint(args.output_uri, output) if existing_manifest is not None else None
    inputs = stage_inputs(config, args.work_root)
    source_patches = patch_source_for_reference(inputs.source)
    command = training_command(config, inputs, args.gate, output, world_size=world_size)
    manifest = {
        **identity,
        "command": command,
        "runtime": runtime_inventory(inputs.source),
        "source_patches": source_patches,
        "artifact_verifications": [asdict(verification) for verification in inputs.artifact_verifications],
        "attempt": int(existing_manifest.get("attempt", 1)) + 1 if existing_manifest is not None else 1,
        "resumed_from_step": resumed_from_step,
        "status": STATUS_RUNNING,
    }
    manifest_path = output / CONTROL_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    sync_tree(output, args.output_uri)
    stop = threading.Event()
    uploader = threading.Thread(
        target=periodic_sync,
        args=(output, args.output_uri, stop, args.sync_interval),
        daemon=True,
        name="open-mopd-output-sync",
    )
    uploader.start()
    try:
        result = subprocess.run(command, cwd=inputs.source, check=False)
        manifest["returncode"] = result.returncode
        manifest["status"] = STATUS_COMPLETE if result.returncode == 0 else STATUS_FAILED
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    finally:
        stop.set()
        uploader.join(timeout=10)
        sync_tree(output, args.output_uri)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
