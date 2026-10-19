"""Build an inspectable Iris evaluation command for an Open-MOPD model or training checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from cloud.iris.open_mopd_fidelity import (
    Hardware,
    actor_checkpoint_relative_path,
    gpu_count,
    load_config,
    validate_output_uri,
)
from cloud.iris.runtime_bundle import resolve_launcher_source

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "open_mopd_evaluation.json"
TASK_MODULE = "cloud.iris.open_mopd_evaluation_task"
BENCHMARK_NAMES = ("aime24", "aime25", "livecodebench_v5", "livecodebench_v6", "ifeval", "ifbench_test")
BENCHMARK_DOMAINS = ("math", "math", "code", "code", "if", "if")
RELEASED_SCORERS = ("aime24", "aime25", "ifeval")
GATES = ("smoke", "full")
SCORE_MODES = ("released", "unavailable")
SMOKE_MAX_TOKENS = 512
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class EvaluationData:
    repository: str
    revision: str


@dataclass(frozen=True)
class EvaluationProtocol:
    source_commit: str
    max_model_len: int
    dtype: str
    max_num_seqs: int
    top_p: float
    top_k: int
    stop_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class EvaluationBenchmark:
    name: str
    domain: str
    path: str
    size: int
    sha256: str
    rows: int
    samples: int
    temperature: float
    max_tokens: int
    gpu_memory_utilization: float
    enable_thinking: bool | None
    score_mode: str
    published_score: float
    unavailable_reason: str | None


@dataclass(frozen=True)
class EvaluationConfig:
    schema_version: int
    fidelity_config: str
    data: EvaluationData
    protocol: EvaluationProtocol
    hardware: Hardware
    benchmarks: tuple[EvaluationBenchmark, ...]
    known_omissions: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkCoverage:
    name: str
    rollout_rows: int
    score_status: str
    comparable_to_paper: bool
    note: str | None


@dataclass(frozen=True)
class EvaluationLaunchPlan:
    gate: str
    job_name: str
    gpu_slice: str
    launcher_commit: str
    source_commit: str
    protocol_source_commit: str
    model_repository: str | None
    model_revision: str | None
    checkpoint_uri: str | None
    checkpoint_step: int | None
    model_export_uri: str | None
    model_export_identity: str | None
    data_repository: str
    data_revision: str
    output_uri: str
    task_image: str
    planned_completions: int
    maximum_output_tokens: int
    coverage: tuple[BenchmarkCoverage, ...]
    iris_command: tuple[str, ...]
    known_omissions: tuple[str, ...]

    def json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _object(value: Any, name: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    if set(value) != keys:
        raise ValueError(f"{name} keys must be {sorted(keys)}; found {sorted(value)}")
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


def _commit(value: Any, name: str) -> str:
    value = _string(value, name)
    if not _COMMIT_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a full 40-character commit")
    return value


def _benchmark(value: Any, index: int) -> EvaluationBenchmark:
    name = f"benchmarks[{index}]"
    item = _object(value, name, set(EvaluationBenchmark.__dataclass_fields__))
    path = _string(item["path"], f"{name}.path")
    parsed_path = PurePosixPath(path)
    if (
        parsed_path.is_absolute()
        or not parsed_path.parts
        or path != parsed_path.as_posix()
        or ".." in parsed_path.parts
    ):
        raise ValueError(f"{name}.path must be a normalized safe relative path")
    size = _integer(item["size"], f"{name}.size")
    rows = _integer(item["rows"], f"{name}.rows")
    samples = _integer(item["samples"], f"{name}.samples")
    max_tokens = _integer(item["max_tokens"], f"{name}.max_tokens")
    sha256 = _string(item["sha256"], f"{name}.sha256")
    score_mode = _string(item["score_mode"], f"{name}.score_mode")
    unavailable_reason = item["unavailable_reason"]
    gpu_memory_utilization = _number(item["gpu_memory_utilization"], f"{name}.gpu_memory_utilization")
    if size <= 0 or rows <= 0 or samples <= 0 or max_tokens <= 0:
        raise ValueError(f"{name} size, rows, samples, and max_tokens must be positive")
    if not 0 < gpu_memory_utilization < 1:
        raise ValueError(f"{name}.gpu_memory_utilization must be in (0, 1)")
    if not _DIGEST_PATTERN.fullmatch(sha256):
        raise ValueError(f"{name}.sha256 must be a lowercase 64-character SHA-256")
    if score_mode not in SCORE_MODES:
        raise ValueError(f"{name}.score_mode must be one of {SCORE_MODES}")
    if (score_mode == "unavailable") != (isinstance(unavailable_reason, str) and bool(unavailable_reason)):
        raise ValueError(f"{name}.unavailable_reason must be set exactly when scoring is unavailable")
    enable_thinking = item["enable_thinking"]
    if enable_thinking is not None and not isinstance(enable_thinking, bool):
        raise TypeError(f"{name}.enable_thinking must be true, false, or null")
    return EvaluationBenchmark(
        name=_string(item["name"], f"{name}.name"),
        domain=_string(item["domain"], f"{name}.domain"),
        path=path,
        size=size,
        sha256=sha256,
        rows=rows,
        samples=samples,
        temperature=_number(item["temperature"], f"{name}.temperature"),
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_thinking=enable_thinking,
        score_mode=score_mode,
        published_score=_number(item["published_score"], f"{name}.published_score"),
        unavailable_reason=unavailable_reason,
    )


def load_evaluation_config(path: Path) -> EvaluationConfig:
    root = _object(
        json.loads(path.read_text()),
        "config",
        {"schema_version", "fidelity_config", "data", "protocol", "hardware", "benchmarks", "known_omissions"},
    )
    if _integer(root["schema_version"], "schema_version") != 1:
        raise ValueError("Open-MOPD evaluation config schema_version must be 1")
    data = _object(root["data"], "data", {"repository", "revision"})
    protocol = _object(
        root["protocol"],
        "protocol",
        {"source_commit", "max_model_len", "dtype", "max_num_seqs", "top_p", "top_k", "stop_token_ids"},
    )
    stop_token_ids = protocol["stop_token_ids"]
    if not isinstance(stop_token_ids, list) or not stop_token_ids:
        raise ValueError("protocol.stop_token_ids must be a non-empty list")
    max_model_len = _integer(protocol["max_model_len"], "protocol.max_model_len")
    max_num_seqs = _integer(protocol["max_num_seqs"], "protocol.max_num_seqs")
    top_p = _number(protocol["top_p"], "protocol.top_p")
    if max_model_len <= 0 or max_num_seqs <= 0 or not 0 < top_p <= 1:
        raise ValueError("protocol limits must be positive and top_p must be in (0, 1]")
    hardware = _object(root["hardware"], "hardware", {"gpu", "cpu", "memory", "disk"})
    raw_benchmarks = root["benchmarks"]
    if not isinstance(raw_benchmarks, list):
        raise TypeError("benchmarks must be a list")
    benchmarks = tuple(_benchmark(value, index) for index, value in enumerate(raw_benchmarks))
    if tuple(benchmark.name for benchmark in benchmarks) != BENCHMARK_NAMES:
        raise ValueError(f"benchmarks must be ordered as {BENCHMARK_NAMES}")
    if tuple(benchmark.domain for benchmark in benchmarks) != BENCHMARK_DOMAINS:
        raise ValueError(f"benchmark domains must be ordered as {BENCHMARK_DOMAINS}")
    released_scorers = tuple(benchmark.name for benchmark in benchmarks if benchmark.score_mode == "released")
    if released_scorers != RELEASED_SCORERS:
        raise ValueError(f"released scoring is only available for {RELEASED_SCORERS}")
    for domain in ("math", "code", "if"):
        domain_benchmarks = [benchmark for benchmark in benchmarks if benchmark.domain == domain]
        settings = {
            (item.samples, item.temperature, item.max_tokens, item.gpu_memory_utilization, item.enable_thinking)
            for item in domain_benchmarks
        }
        if len(settings) != 1:
            raise ValueError(f"{domain} benchmarks must share sampling settings")
    omissions = root["known_omissions"]
    if (
        not isinstance(omissions, list)
        or not omissions
        or not all(isinstance(item, str) and item for item in omissions)
    ):
        raise ValueError("known_omissions must be a non-empty list of strings")
    return EvaluationConfig(
        schema_version=1,
        fidelity_config=_string(root["fidelity_config"], "fidelity_config"),
        data=EvaluationData(
            repository=_string(data["repository"], "data.repository"),
            revision=_commit(data["revision"], "data.revision"),
        ),
        protocol=EvaluationProtocol(
            source_commit=_commit(protocol["source_commit"], "protocol.source_commit"),
            max_model_len=max_model_len,
            dtype=_string(protocol["dtype"], "protocol.dtype"),
            max_num_seqs=max_num_seqs,
            top_p=top_p,
            top_k=_integer(protocol["top_k"], "protocol.top_k"),
            stop_token_ids=tuple(_integer(value, "protocol.stop_token_ids[]") for value in stop_token_ids),
        ),
        hardware=Hardware(
            gpu=_string(hardware["gpu"], "hardware.gpu"),
            cpu=_integer(hardware["cpu"], "hardware.cpu"),
            memory=_string(hardware["memory"], "hardware.memory"),
            disk=_string(hardware["disk"], "hardware.disk"),
        ),
        benchmarks=benchmarks,
        known_omissions=tuple(omissions),
    )


def benchmark_coverage(config: EvaluationConfig, gate: str) -> tuple[BenchmarkCoverage, ...]:
    if gate not in GATES:
        raise ValueError(f"Unknown gate {gate!r}; choose one of {GATES}")
    coverage = []
    for benchmark in config.benchmarks:
        if gate == "smoke":
            coverage.append(BenchmarkCoverage(benchmark.name, 1, "smoke", False, "One prompt, one bounded completion"))
        elif benchmark.score_mode == "released":
            coverage.append(BenchmarkCoverage(benchmark.name, benchmark.rows * benchmark.samples, "scored", True, None))
        else:
            coverage.append(
                BenchmarkCoverage(
                    benchmark.name,
                    benchmark.rows * benchmark.samples,
                    "rollout_only",
                    False,
                    benchmark.unavailable_reason,
                )
            )
    return tuple(coverage)


def evaluation_scale(config: EvaluationConfig, gate: str) -> tuple[int, int]:
    if gate == "smoke":
        completions = len(config.benchmarks)
        return completions, completions * SMOKE_MAX_TOKENS
    if gate != "full":
        raise ValueError(f"Unknown gate {gate!r}; choose one of {GATES}")
    completions = sum(benchmark.rows * benchmark.samples for benchmark in config.benchmarks)
    maximum_tokens = sum(benchmark.rows * benchmark.samples * benchmark.max_tokens for benchmark in config.benchmarks)
    return completions, maximum_tokens


def evaluation_job_name(gate: str, output_uri: str) -> str:
    output_id = hashlib.sha256(output_uri.encode()).hexdigest()[:8]
    return f"open-mopd-final-eval-{gate}-{output_id}"


def validate_checkpoint_source(checkpoint_uri: str | None, checkpoint_step: int | None) -> None:
    """Validate an optional durable FSDP actor checkpoint selector."""
    if checkpoint_uri is None and checkpoint_step is None:
        return
    if checkpoint_uri is None or checkpoint_step is None:
        raise ValueError("--checkpoint-uri and --checkpoint-step must be specified together")
    validate_output_uri(checkpoint_uri)
    if checkpoint_step <= 0:
        raise ValueError("--checkpoint-step must be positive")
    expected_suffix = f"/{actor_checkpoint_relative_path(checkpoint_step)}"
    if not checkpoint_uri.rstrip("/").endswith(expected_suffix):
        raise ValueError(f"--checkpoint-uri must end with {expected_suffix}")


def validate_model_source(
    checkpoint_uri: str | None, model_export_uri: str | None, model_export_identity: str | None
) -> None:
    """Require one complete, durable model source selector."""
    if (model_export_uri is None) != (model_export_identity is None):
        raise ValueError("--model-export-uri and --model-export-identity must be specified together")
    if model_export_uri is not None:
        validate_output_uri(model_export_uri)
        if not model_export_identity:
            raise ValueError("--model-export-identity must be nonempty")
        if checkpoint_uri is not None:
            raise ValueError("A reference checkpoint and a native model export cannot be selected together")


def build_plan(
    config: EvaluationConfig,
    *,
    config_path: Path,
    gate: str,
    cluster_config: Path,
    output_uri: str,
    task_image: str,
    gpu_slice: str | None = None,
    checkpoint_uri: str | None = None,
    checkpoint_step: int | None = None,
    model_export_uri: str | None = None,
    model_export_identity: str | None = None,
) -> EvaluationLaunchPlan:
    validate_output_uri(output_uri)
    validate_checkpoint_source(checkpoint_uri, checkpoint_step)
    validate_model_source(checkpoint_uri, model_export_uri, model_export_identity)
    normalized_checkpoint_uri = checkpoint_uri.rstrip("/") if checkpoint_uri is not None else None
    normalized_model_export_uri = model_export_uri.rstrip("/") if model_export_uri is not None else None
    source = resolve_launcher_source()
    if not re.fullmatch(r"[^@]+@sha256:[0-9a-f]{64}", task_image):
        raise ValueError("--task-image must be a digest-addressed image reference")
    selected_slice = gpu_slice or config.hardware.gpu
    gpu_count(selected_slice)
    try:
        task_config = config_path.resolve().relative_to(source.root)
        fidelity_path = (source.root / config.fidelity_config).resolve().relative_to(source.root)
    except ValueError as error:
        raise ValueError("Evaluation and fidelity configs must be inside the checkout bundled by Iris") from error
    fidelity = load_config(source.root / fidelity_path)
    planned_completions, maximum_output_tokens = evaluation_scale(config, gate)
    job_name = evaluation_job_name(gate, output_uri)
    omissions = list(config.known_omissions)
    if selected_slice != config.hardware.gpu:
        omissions.append(f"Hardware override uses {selected_slice}; the authors report {config.hardware.gpu}.")
    task_args = (
        "--config",
        task_config.as_posix(),
        "--fidelity-config",
        fidelity_path.as_posix(),
        "--gate",
        gate,
        "--output-uri",
        output_uri,
        "--gpu-slice",
        selected_slice,
        "--task-image",
        task_image,
        "--launcher-commit",
        source.commit,
    )
    if normalized_checkpoint_uri is not None and checkpoint_step is not None:
        task_args += ("--checkpoint-uri", normalized_checkpoint_uri, "--checkpoint-step", str(checkpoint_step))
    if normalized_model_export_uri is not None and model_export_identity is not None:
        task_args += (
            "--model-export-uri",
            normalized_model_export_uri,
            "--model-export-identity",
            model_export_identity,
        )
    command = (
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
        selected_slice,
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
        job_name,
        "--",
        "python",
        "-m",
        TASK_MODULE,
        *task_args,
    )
    return EvaluationLaunchPlan(
        gate=gate,
        job_name=job_name,
        gpu_slice=selected_slice,
        launcher_commit=source.commit,
        source_commit=fidelity.source.commit,
        protocol_source_commit=config.protocol.source_commit,
        model_repository=(
            fidelity.evaluation_reference.repository
            if normalized_checkpoint_uri is None and normalized_model_export_uri is None
            else None
        ),
        model_revision=(
            fidelity.evaluation_reference.revision
            if normalized_checkpoint_uri is None and normalized_model_export_uri is None
            else None
        ),
        checkpoint_uri=normalized_checkpoint_uri,
        checkpoint_step=checkpoint_step,
        model_export_uri=normalized_model_export_uri,
        model_export_identity=model_export_identity,
        data_repository=config.data.repository,
        data_revision=config.data.revision,
        output_uri=output_uri,
        task_image=task_image,
        planned_completions=planned_completions,
        maximum_output_tokens=maximum_output_tokens,
        coverage=benchmark_coverage(config, gate),
        iris_command=command,
        known_omissions=tuple(omissions),
    )


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gate", choices=GATES, default="smoke")
    parser.add_argument("--cluster-config", type=Path, required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--task-image", required=True)
    parser.add_argument("--gpu-slice")
    parser.add_argument("--checkpoint-uri")
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--model-export-uri")
    parser.add_argument("--model-export-identity")
    parser.add_argument("--allow-known-omissions", action="store_true")
    parser.add_argument("--submit", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    config = load_evaluation_config(args.config)
    plan = build_plan(
        config,
        config_path=args.config,
        gate=args.gate,
        cluster_config=args.cluster_config.resolve(),
        output_uri=args.output_uri,
        task_image=args.task_image,
        gpu_slice=args.gpu_slice,
        checkpoint_uri=args.checkpoint_uri,
        checkpoint_step=args.checkpoint_step,
        model_export_uri=args.model_export_uri,
        model_export_identity=args.model_export_identity,
    )
    print(plan.json())
    print(shlex.join(plan.iris_command))
    if not args.submit:
        print("Dry run only. Add --submit --allow-known-omissions after reviewing the plan.")
        return 0
    if not args.allow_known_omissions:
        raise SystemExit("--submit requires --allow-known-omissions")
    return subprocess.run(plan.iris_command, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
