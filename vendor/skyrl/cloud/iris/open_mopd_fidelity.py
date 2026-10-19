"""Build an inspectable Iris command for the released Open-MOPD control."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from cloud.iris.runtime_bundle import LauncherSource, resolve_launcher_source

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "open_mopd_fidelity.json"
TASK_MODULE = "cloud.iris.open_mopd_fidelity_task"
DOMAINS = ("math", "code", "if")
PAPER_DOMAIN_RESPONSE_LIMITS = (16384, 16384, 2048)
GATES = {"one_step": 1, "paper_checkpoint": 200, "paper_schedule": 600}
CHECKPOINT_DIRECTORY_NAME = "checkpoints"
GLOBAL_STEP_PREFIX = "global_step_"
ACTOR_DIRECTORY_NAME = "actor"
_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
_GPU_SLICE_PATTERN = re.compile(r"[A-Za-z0-9_-]+x([1-9][0-9]*)")


def actor_checkpoint_relative_path(step: int) -> str:
    """Return the released trainer's actor-checkpoint path for one positive step."""
    if step <= 0:
        raise ValueError("Checkpoint step must be positive")
    return f"{CHECKPOINT_DIRECTORY_NAME}/{GLOBAL_STEP_PREFIX}{step}/{ACTOR_DIRECTORY_NAME}"


@dataclass(frozen=True)
class Source:
    repository: str
    commit: str


@dataclass(frozen=True)
class LfsFile:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ModelArtifact:
    repository: str
    revision: str
    lfs_files: tuple[LfsFile, ...]


@dataclass(frozen=True)
class DatasetArtifact:
    repository: str
    revision: str
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class Teacher:
    domain: str
    repository: str
    revision: str
    lfs_files: tuple[LfsFile, ...]


@dataclass(frozen=True)
class Environment:
    python: str
    packages: dict[str, str]


@dataclass(frozen=True)
class Hardware:
    gpu: str
    cpu: int
    memory: str
    disk: str


@dataclass(frozen=True)
class Training:
    train_batch_size: int
    mini_batch_size: int
    ppo_epochs: int
    micro_batch_size_per_gpu: int
    learning_rate: float
    gradient_clip: float
    entropy_coefficient: float
    loss_aggregation: str
    prompt_limit: int
    response_limit: int
    domain_response_limits: tuple[int, int, int]
    top_k: int
    nucleus_p: float
    temperature: float
    teacher_temperature: float
    clip_low: float
    clip_high: float
    domain_weights: tuple[int, int, int]
    target_gradient_shares: tuple[float, float, float]
    gap_following_alpha: float
    save_every: int
    eval_every: int
    eval_temperature: float
    eval_nucleus_p: float
    eval_samples: int


@dataclass(frozen=True)
class FidelityConfig:
    schema_version: int
    source: Source
    student: ModelArtifact
    teachers: tuple[Teacher, Teacher, Teacher]
    evaluation_reference: ModelArtifact
    dataset: DatasetArtifact
    validation_dataset: DatasetArtifact
    hardware: Hardware
    environment: Environment
    training: Training
    known_deviations: tuple[str, ...]


@dataclass(frozen=True)
class FidelityLaunchPlan:
    gate: str
    steps: int
    gpu_slice: str
    source_commit: str
    launcher_commit: str
    task_image: str
    output_uri: str
    evaluation_reference: ModelArtifact
    prompt_limit: int
    iris_command: tuple[str, ...]
    known_deviations: tuple[str, ...]

    def json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _object(value: Any, name: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    actual = set(value)
    if actual != keys:
        raise ValueError(f"{name} keys must be {sorted(keys)}; found {sorted(actual)}")
    return value


def _revision(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a full 40-character commit")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def _number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    return float(value)


def _triple(value: Any, name: str, item_parser: Any) -> tuple[Any, Any, Any]:
    if not isinstance(value, list) or len(value) != len(DOMAINS):
        raise TypeError(f"{name} must contain exactly three values")
    return tuple(item_parser(item, f"{name}[{index}]") for index, item in enumerate(value))


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def _lfs_files(value: Any, name: str) -> tuple[LfsFile, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    files = []
    for index, raw in enumerate(value):
        item_name = f"{name}[{index}]"
        item = _object(raw, item_name, {"path", "size", "sha256"})
        path = _string(item["path"], f"{item_name}.path")
        parsed_path = PurePosixPath(path)
        if (
            parsed_path.is_absolute()
            or not parsed_path.parts
            or path != parsed_path.as_posix()
            or any(part in {".", ".."} for part in parsed_path.parts)
        ):
            raise ValueError(f"{item_name}.path must be a normalized safe relative path")
        size = _integer(item["size"], f"{item_name}.size")
        if size <= 0:
            raise ValueError(f"{item_name}.size must be positive")
        files.append(LfsFile(path=path, size=size, sha256=_sha256(item["sha256"], f"{item_name}.sha256")))
    paths = [file.path for file in files]
    if len(paths) != len(set(paths)):
        raise ValueError(f"{name} paths must be unique")
    return tuple(files)


def _model_artifact(value: Any, name: str) -> ModelArtifact:
    item = _object(value, name, {"repository", "revision", "lfs_files"})
    return ModelArtifact(
        repository=_string(item["repository"], f"{name}.repository"),
        revision=_revision(item["revision"], f"{name}.revision"),
        lfs_files=_lfs_files(item["lfs_files"], f"{name}.lfs_files"),
    )


def _dataset_artifact(value: Any, name: str) -> DatasetArtifact:
    item = _object(value, name, {"repository", "revision", "path", "size", "sha256"})
    size = _integer(item["size"], f"{name}.size")
    if size <= 0:
        raise ValueError(f"{name}.size must be positive")
    return DatasetArtifact(
        repository=_string(item["repository"], f"{name}.repository"),
        revision=_revision(item["revision"], f"{name}.revision"),
        path=_string(item["path"], f"{name}.path"),
        size=size,
        sha256=_sha256(item["sha256"], f"{name}.sha256"),
    )


def load_config(path: Path) -> FidelityConfig:
    root = _object(
        json.loads(path.read_text()),
        "config",
        {"schema_version", "source", "artifacts", "hardware", "environment", "training", "gates", "known_deviations"},
    )
    if _integer(root["schema_version"], "schema_version") != 4:
        raise ValueError("Open-MOPD fidelity config schema_version must be 4")
    source_value = _object(root["source"], "source", {"repository", "commit"})
    source = Source(
        _string(source_value["repository"], "source.repository"),
        _revision(source_value["commit"], "source.commit"),
    )
    artifacts = _object(
        root["artifacts"],
        "artifacts",
        {"student", "teachers", "evaluation_reference", "dataset", "validation_dataset"},
    )
    teacher_values = artifacts["teachers"]
    if not isinstance(teacher_values, list) or len(teacher_values) != len(DOMAINS):
        raise ValueError("artifacts.teachers must contain exactly math, code, and if")
    teachers = tuple(
        Teacher(
            domain=_string(item["domain"], f"teachers[{index}].domain"),
            repository=_string(item["repository"], f"teachers[{index}].repository"),
            revision=_revision(item["revision"], f"teachers[{index}].revision"),
            lfs_files=_lfs_files(item["lfs_files"], f"teachers[{index}].lfs_files"),
        )
        for index, value in enumerate(teacher_values)
        for item in [_object(value, f"teachers[{index}]", {"domain", "repository", "revision", "lfs_files"})]
    )
    if tuple(teacher.domain for teacher in teachers) != DOMAINS:
        raise ValueError("Teacher order must be math, code, if because patched verl assigns teachers positionally")
    gates = _object(root["gates"], "gates", set(GATES))
    if any(_integer(gates[name], f"gates.{name}") != steps for name, steps in GATES.items()):
        raise ValueError("Open-MOPD fidelity gates must resolve to 1, 200, and 600 steps")
    hardware_value = _object(root["hardware"], "hardware", {"gpu", "cpu", "memory", "disk"})
    environment_value = _object(root["environment"], "environment", {"python", "packages"})
    packages_value = environment_value["packages"]
    if not isinstance(packages_value, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in packages_value.items()
    ):
        raise TypeError("environment.packages must map distribution names to versions")
    packages = dict(packages_value)
    training_value = _object(root["training"], "training", set(Training.__dataclass_fields__))
    training = Training(
        train_batch_size=_integer(training_value["train_batch_size"], "training.train_batch_size"),
        mini_batch_size=_integer(training_value["mini_batch_size"], "training.mini_batch_size"),
        ppo_epochs=_integer(training_value["ppo_epochs"], "training.ppo_epochs"),
        micro_batch_size_per_gpu=_integer(
            training_value["micro_batch_size_per_gpu"], "training.micro_batch_size_per_gpu"
        ),
        learning_rate=_number(training_value["learning_rate"], "training.learning_rate"),
        gradient_clip=_number(training_value["gradient_clip"], "training.gradient_clip"),
        entropy_coefficient=_number(training_value["entropy_coefficient"], "training.entropy_coefficient"),
        loss_aggregation=_string(training_value["loss_aggregation"], "training.loss_aggregation"),
        prompt_limit=_integer(training_value["prompt_limit"], "training.prompt_limit"),
        response_limit=_integer(training_value["response_limit"], "training.response_limit"),
        domain_response_limits=_triple(
            training_value["domain_response_limits"], "training.domain_response_limits", _integer
        ),
        top_k=_integer(training_value["top_k"], "training.top_k"),
        nucleus_p=_number(training_value["nucleus_p"], "training.nucleus_p"),
        temperature=_number(training_value["temperature"], "training.temperature"),
        teacher_temperature=_number(training_value["teacher_temperature"], "training.teacher_temperature"),
        clip_low=_number(training_value["clip_low"], "training.clip_low"),
        clip_high=_number(training_value["clip_high"], "training.clip_high"),
        domain_weights=_triple(training_value["domain_weights"], "training.domain_weights", _integer),
        target_gradient_shares=_triple(
            training_value["target_gradient_shares"], "training.target_gradient_shares", _number
        ),
        gap_following_alpha=_number(training_value["gap_following_alpha"], "training.gap_following_alpha"),
        save_every=_integer(training_value["save_every"], "training.save_every"),
        eval_every=_integer(training_value["eval_every"], "training.eval_every"),
        eval_temperature=_number(training_value["eval_temperature"], "training.eval_temperature"),
        eval_nucleus_p=_number(training_value["eval_nucleus_p"], "training.eval_nucleus_p"),
        eval_samples=_integer(training_value["eval_samples"], "training.eval_samples"),
    )
    if training.train_batch_size % training.mini_batch_size:
        raise ValueError("train_batch_size must be divisible by mini_batch_size")
    if (
        training.response_limit != max(PAPER_DOMAIN_RESPONSE_LIMITS)
        or training.domain_response_limits != PAPER_DOMAIN_RESPONSE_LIMITS
    ):
        raise ValueError("Open-MOPD paper response limits must be 16K for math/code and 2K for IF")
    if training.eval_every <= 0:
        raise ValueError("training.eval_every must be positive")
    if training.eval_temperature <= 0 or not 0 < training.eval_nucleus_p <= 1 or training.eval_samples <= 0:
        raise ValueError("training inline validation sampling values must be positive and top-p at most 1")
    deviations = root["known_deviations"]
    if not isinstance(deviations, list) or not deviations or not all(isinstance(value, str) for value in deviations):
        raise ValueError("known_deviations must be a non-empty list of strings")
    return FidelityConfig(
        schema_version=4,
        source=source,
        student=_model_artifact(artifacts["student"], "artifacts.student"),
        teachers=teachers,
        evaluation_reference=_model_artifact(artifacts["evaluation_reference"], "artifacts.evaluation_reference"),
        dataset=_dataset_artifact(artifacts["dataset"], "artifacts.dataset"),
        validation_dataset=_dataset_artifact(artifacts["validation_dataset"], "artifacts.validation_dataset"),
        hardware=Hardware(
            gpu=_string(hardware_value["gpu"], "hardware.gpu"),
            cpu=_integer(hardware_value["cpu"], "hardware.cpu"),
            memory=_string(hardware_value["memory"], "hardware.memory"),
            disk=_string(hardware_value["disk"], "hardware.disk"),
        ),
        environment=Environment(python=_string(environment_value["python"], "environment.python"), packages=packages),
        training=training,
        known_deviations=tuple(deviations),
    )


def gpu_count(gpu_slice: str) -> int:
    match = _GPU_SLICE_PATTERN.fullmatch(gpu_slice)
    if match is None:
        raise ValueError(f"Malformed Iris GPU slice: {gpu_slice!r}")
    count = int(match.group(1))
    if count != 8:
        raise ValueError(f"The released Open-MOPD stack requires one 8-GPU node, not {gpu_slice!r}")
    return count


def validate_output_uri(output_uri: str, *, option_name: str = "--output-uri") -> None:
    parsed = urlparse(output_uri)
    if parsed.scheme not in {"gs", "s3"} or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"{option_name} must be a non-root gs:// or s3:// durable prefix")


def _provenance(task_image: str, source: LauncherSource) -> tuple[str, str]:
    if not re.fullmatch(r"[^@]+@sha256:[0-9a-f]{64}", task_image):
        raise ValueError("--task-image must be a digest-addressed image reference")
    return source.commit, task_image


def _deviations(config: FidelityConfig, gpu_slice: str) -> tuple[str, ...]:
    deviations = list(config.known_deviations)
    if gpu_slice != config.hardware.gpu:
        deviations.append(f"Hardware override uses {gpu_slice}; the authors report {config.hardware.gpu}.")
    return tuple(deviations)


def _iris_command(
    config: FidelityConfig,
    *,
    config_path: Path,
    gate: str,
    cluster_config: Path,
    output_uri: str,
    task_image: str,
    launcher_commit: str,
    launcher_root: Path,
    gpu_slice: str,
) -> tuple[str, ...]:
    try:
        task_config_path = config_path.resolve().relative_to(launcher_root)
    except ValueError as error:
        raise ValueError("--config must be inside the checkout bundled by Iris") from error
    return (
        "uv",
        "run",
        "--frozen",
        "iris",
        "--config",
        str(cluster_config),
        "job",
        "run",
        "--enable-extra-resources",
        "--gpu",
        gpu_slice,
        "--cpu",
        str(config.hardware.cpu),
        "--memory",
        config.hardware.memory,
        "--disk",
        config.hardware.disk,
        "--priority",
        "interactive",
        "--no-preemptible",
        "--max-retries",
        "0",
        "--task-image",
        task_image,
        "--no-sync",
        "--no-wait",
        "--job-name",
        f"open-mopd-fidelity-{gate.replace('_', '-')}",
        "--",
        "python",
        "-m",
        TASK_MODULE,
        "--config",
        task_config_path.as_posix(),
        "--gate",
        gate,
        "--output-uri",
        output_uri,
        "--gpu-slice",
        gpu_slice,
        "--task-image",
        task_image,
        "--launcher-commit",
        launcher_commit,
    )


def build_plan(
    config: FidelityConfig,
    *,
    config_path: Path,
    gate: str,
    cluster_config: Path,
    output_uri: str,
    task_image: str,
    gpu_slice: str | None = None,
) -> FidelityLaunchPlan:
    if gate not in GATES:
        raise ValueError(f"Unknown gate {gate!r}; choose one of {sorted(GATES)}")
    validate_output_uri(output_uri)
    gpu_slice = gpu_slice or config.hardware.gpu
    gpu_count(gpu_slice)
    launcher_source = resolve_launcher_source()
    launcher_commit, pinned_image = _provenance(task_image, launcher_source)
    command = _iris_command(
        config,
        config_path=config_path,
        gate=gate,
        cluster_config=cluster_config,
        output_uri=output_uri,
        task_image=pinned_image,
        launcher_commit=launcher_commit,
        launcher_root=launcher_source.root,
        gpu_slice=gpu_slice,
    )
    return FidelityLaunchPlan(
        gate=gate,
        steps=GATES[gate],
        gpu_slice=gpu_slice,
        source_commit=config.source.commit,
        launcher_commit=launcher_commit,
        task_image=pinned_image,
        output_uri=output_uri,
        evaluation_reference=config.evaluation_reference,
        prompt_limit=config.training.prompt_limit,
        iris_command=command,
        known_deviations=_deviations(config, gpu_slice),
    )


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gate", choices=tuple(GATES), default="one_step")
    parser.add_argument("--cluster-config", type=Path, required=True)
    parser.add_argument("--output-uri", required=True, help="Unique durable gs:// or s3:// prefix for this run")
    parser.add_argument(
        "--task-image", required=True, help="Digest-addressed image containing the authors' dependencies"
    )
    parser.add_argument("--gpu-slice", help="Schedulable Iris GPU slice override, for example H100x8")
    parser.add_argument("--allow-known-deviations", action="store_true", help="Required with --submit")
    parser.add_argument("--submit", action="store_true", help="Submit after printing the resolved plan")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    config = load_config(args.config)
    plan = build_plan(
        config,
        config_path=args.config,
        gate=args.gate,
        cluster_config=args.cluster_config.resolve(),
        output_uri=args.output_uri,
        task_image=args.task_image,
        gpu_slice=args.gpu_slice,
    )
    print(plan.json())
    print(shlex.join(plan.iris_command))
    if not args.submit:
        print("Dry run only. Add --submit --allow-known-deviations after reviewing the plan.")
        return 0
    if not args.allow_known_deviations:
        raise SystemExit("--submit requires --allow-known-deviations")
    return subprocess.run(plan.iris_command, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
