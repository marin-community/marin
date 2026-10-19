import hashlib
import io
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import cloud.iris.open_mopd_evaluation as evaluation
import cloud.iris.open_mopd_evaluation_task as evaluation_task
from cloud.iris.open_mopd_evaluation_task import (
    EvaluationCommandError,
    EvaluationInputs,
    StagedBenchmark,
    run_logged_command,
    rollout_commands,
    stage_checkpoint_model,
    stage_native_model,
    verify_benchmark_file,
)
from cloud.iris.open_mopd_fidelity import load_config
from cloud.iris.open_mopd_fidelity_task import ArtifactVerification
from cloud.iris.open_mopd_vllm_rollout import evaluation_port_seed, worker_port

TASK_IMAGE = "registry.example/open-mopd-eval@sha256:" + "1" * 64
OUTPUT_URI = "s3://bucket/open-mopd/final-eval"


def _inputs(config: evaluation.EvaluationConfig) -> EvaluationInputs:
    verification = ArtifactVerification(repository="fixture", revision="1" * 40, files=())
    return EvaluationInputs(
        source=Path("/work/Open-MOPD"),
        model=Path("/work/model"),
        benchmarks=tuple(
            StagedBenchmark(benchmark=benchmark, path=Path("/work/data") / benchmark.path, verification=verification)
            for benchmark in config.benchmarks
        ),
        model_verification=verification,
    )


def _options(command: tuple[str, ...]) -> dict[str, str | bool]:
    options: dict[str, str | bool] = {}
    index = command.index("--output-dir") + 2
    while index < len(command):
        option = command[index]
        if option == "--trust-remote-code":
            options[option] = True
            index += 1
        else:
            options[option] = command[index + 1]
            index += 2
    return options


def test_full_gate_reports_scored_and_rollout_only_coverage() -> None:
    config = evaluation.load_evaluation_config(evaluation.DEFAULT_CONFIG)

    coverage = {item.name: item for item in evaluation.benchmark_coverage(config, "full")}

    assert coverage["aime24"].rollout_rows == 30 * 64
    assert coverage["livecodebench_v6"].rollout_rows == 175 * 10
    assert coverage["ifeval"].score_status == "scored"
    assert coverage["ifbench_test"].score_status == "rollout_only"
    assert coverage["ifbench_test"].note
    assert evaluation.evaluation_scale(config, "full") == (8_101, 230_050_000)


def test_rollout_commands_preserve_released_domain_protocols() -> None:
    config = evaluation.load_evaluation_config(evaluation.DEFAULT_CONFIG)
    port_seed = evaluation_port_seed(OUTPUT_URI)

    commands = rollout_commands(
        config,
        _inputs(config),
        "full",
        Path("/work/output"),
        world_size=8,
        vllm_port_seed=port_seed,
    )
    math, code, instruction = (_options(command.argv) for command in commands)

    assert math["--temperature"] == "0.6"
    assert math["--n"] == "64"
    assert math["--max-tokens"] == "31000"
    assert math["--gpu-memory-utilization"] == "0.9"
    assert math["--enable-thinking"] == "true"
    assert code["--temperature"] == "1.0"
    assert code["--n"] == "10"
    assert code["--max-tokens"] == "30000"
    assert code["--gpu-memory-utilization"] == "0.85"
    assert code["--enable-thinking"] == "true"
    assert instruction["--n"] == "1"
    assert instruction["--max-tokens"] == "10000"
    assert instruction["--gpu-memory-utilization"] == "0.85"
    assert instruction["--enable-thinking"] == "true"
    for options in (math, code, instruction):
        assert options["--data-parallel-size"] == "8"
        assert options["--max-model-len"] == "32768"
        assert options["--dtype"] == "bfloat16"
        assert options["--top-p"] == "0.95"
        assert options["--top-k"] == "-1"
        assert options["--stop-token-ids"] == "128012"
        assert options["--vllm-port-seed"] == str(port_seed)
        assert options["--trust-remote-code"] is True
    assert all(Path(command.argv[1]).name == "open_mopd_vllm_rollout.py" for command in commands)
    assert all(Path(command.argv[1]).is_file() for command in commands)


def test_rollout_workers_get_distinct_vllm_port_ranges() -> None:
    port_seed = evaluation_port_seed(OUTPUT_URI)

    ports = [worker_port(port_seed, rank) for rank in range(8)]

    assert len(set(ports)) == 8
    assert all(20_000 <= port <= 59_999 for port in ports)
    assert min(abs(left - right) for index, left in enumerate(ports) for right in ports[index + 1 :]) > 1


def test_smoke_gate_bounds_every_domain_without_claiming_comparability() -> None:
    config = evaluation.load_evaluation_config(evaluation.DEFAULT_CONFIG)

    commands = rollout_commands(
        config,
        _inputs(config),
        "smoke",
        Path("/work/output"),
        world_size=8,
        vllm_port_seed=evaluation_port_seed(OUTPUT_URI),
    )
    coverage = evaluation.benchmark_coverage(config, "smoke")

    assert all(item.rollout_rows == 1 and not item.comparable_to_paper for item in coverage)
    for command in commands:
        options = _options(command.argv)
        assert options["--n"] == "1"
        assert options["--max-tokens"] == "512"
        assert options["--offset"] == "1"


def test_config_cannot_enable_unreleased_code_scorer(tmp_path: Path) -> None:
    raw = json.loads(evaluation.DEFAULT_CONFIG.read_text())
    code = next(item for item in raw["benchmarks"] if item["name"] == "livecodebench_v5")
    code["score_mode"] = "released"
    code["unavailable_reason"] = None
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="released scoring"):
        evaluation.load_evaluation_config(path)


def test_benchmark_verification_rejects_corrupt_download(tmp_path: Path) -> None:
    config = evaluation.load_evaluation_config(evaluation.DEFAULT_CONFIG)
    benchmark = config.benchmarks[0]
    corrupt = tmp_path / "aime24.parquet"
    corrupt.write_bytes(b"not the released parquet")

    with pytest.raises(ValueError, match="integrity mismatch"):
        verify_benchmark_file(benchmark, corrupt)


def test_benchmark_verification_returns_auditable_observation(tmp_path: Path) -> None:
    config = evaluation.load_evaluation_config(evaluation.DEFAULT_CONFIG)
    original = config.benchmarks[0]
    content = b"fixture parquet"
    benchmark = replace(
        original,
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
    path = tmp_path / "aime24.parquet"
    path.write_bytes(content)

    observed = verify_benchmark_file(benchmark, path)

    assert observed.expected_size == observed.observed_size == len(content)
    assert observed.expected_sha256 == observed.observed_sha256 == benchmark.sha256


def test_native_model_stage_records_downloaded_file_hashes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    files = {
        "config.json": b"{}",
        "tokenizer.json": b"{}",
        "model.safetensors": b"fixture weights",
    }
    for name, content in files.items():
        (source / name).write_bytes(content)

    model, verification = stage_native_model(source.as_uri(), "checkpoint-step-2", tmp_path / "model")

    assert {item.path: item.sha256 for item in verification.files} == {
        name: hashlib.sha256(content).hexdigest() for name, content in files.items()
    }
    assert {name: (model / name).read_bytes() for name in files} == files
    assert verification.source_uri == source.as_uri()
    assert verification.source_identity == "checkpoint-step-2"


def test_native_model_stage_normalizes_transformers_5_fast_tokenizer_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    tokenizer_config = {"tokenizer_class": "TokenizersBackend", "eos_token": "</s>"}
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "tokenizer.json").write_text("{}")
    (source / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))

    model, verification = stage_native_model(source.as_uri(), "checkpoint-step-2", tmp_path / "model")

    assert json.loads((model / "tokenizer_config.json").read_text()) == {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "eos_token": "</s>",
    }
    assert json.loads((source / "tokenizer_config.json").read_text()) == tokenizer_config
    original = next(item for item in verification.files if item.path == "tokenizer_config.json")
    assert original.sha256 == hashlib.sha256((source / "tokenizer_config.json").read_bytes()).hexdigest()
    assert verification.staged_tokenizer_config is not None
    assert (
        verification.staged_tokenizer_config.sha256
        == hashlib.sha256((model / "tokenizer_config.json").read_bytes()).hexdigest()
    )


def test_native_model_stage_rejects_incomplete_weight_index(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "tokenizer.json").write_text("{}")
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": "missing.safetensors"}})
    )
    (source / "other.safetensors").write_bytes(b"unreferenced shard")

    with pytest.raises(RuntimeError, match="missing 1 referenced safetensors shard"):
        stage_native_model(source.as_uri(), "checkpoint-step-2", tmp_path / "model")


def test_native_model_evaluation_plan_selects_export_without_reference_model(monkeypatch: pytest.MonkeyPatch) -> None:
    root = evaluation.DEFAULT_CONFIG.parents[3]
    monkeypatch.setattr(evaluation, "resolve_launcher_source", lambda: SimpleNamespace(root=root, commit="abc123"))
    export_uri = "s3://bucket/native/exports/step-2"

    plan = evaluation.build_plan(
        evaluation.load_evaluation_config(evaluation.DEFAULT_CONFIG),
        config_path=evaluation.DEFAULT_CONFIG,
        gate="smoke",
        cluster_config=Path("/tmp/iris.yaml"),
        output_uri=OUTPUT_URI,
        task_image=TASK_IMAGE,
        model_export_uri=export_uri,
        model_export_identity="run-abc-step-2",
    )

    assert plan.model_repository is None
    assert plan.model_export_uri == export_uri
    assert plan.model_export_identity == "run-abc-step-2"
    assert plan.iris_command[plan.iris_command.index("--model-export-uri") + 1] == export_uri
    assert plan.iris_command[plan.iris_command.index("--model-export-identity") + 1] == "run-abc-step-2"
    assert plan.iris_command[plan.iris_command.index("--priority") + 1] == "interactive"


def test_failed_evaluation_command_persists_combined_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    log_path = tmp_path / "logs" / "rollout-code.log"
    command = [
        sys.executable,
        "-c",
        "import sys; print('rank stdout'); print('rank stderr', file=sys.stderr); raise SystemExit(7)",
    ]

    with pytest.raises(EvaluationCommandError) as caught:
        run_logged_command(command, cwd=tmp_path, log_path=log_path)

    assert caught.value.returncode == 7
    assert caught.value.log_path == log_path
    assert set(log_path.read_text().splitlines()) == {"rank stdout", "rank stderr"}
    assert set(capsys.readouterr().out.splitlines()) == {"rank stdout", "rank stderr"}


def test_dry_run_exposes_immutable_inputs_without_submitting(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = evaluation.DEFAULT_CONFIG.parents[3]
    monkeypatch.setattr(evaluation, "resolve_launcher_source", lambda: SimpleNamespace(root=root, commit="abc123"))
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("dry run submitted a process"))

    assert (
        evaluation.main(
            [
                "--cluster-config",
                "/tmp/iris.yaml",
                "--output-uri",
                OUTPUT_URI,
                "--task-image",
                TASK_IMAGE,
                "--gpu-slice",
                "H100x8",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    plan = json.loads(output.split("\nuv run", 1)[0])
    fidelity = load_config(root / "cloud/iris/configs/open_mopd_fidelity.json")
    assert plan["source_commit"] == fidelity.source.commit
    assert plan["model_revision"] == fidelity.evaluation_reference.revision
    assert plan["job_name"] == plan["iris_command"][plan["iris_command"].index("--job-name") + 1]


def test_submission_requires_omission_acknowledgement(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = evaluation.DEFAULT_CONFIG.parents[3]
    monkeypatch.setattr(evaluation, "resolve_launcher_source", lambda: SimpleNamespace(root=root, commit="abc123"))
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("guard did not stop submission"))

    with pytest.raises(SystemExit, match="allow-known-omissions"):
        evaluation.main(
            [
                "--cluster-config",
                "/tmp/iris.yaml",
                "--output-uri",
                OUTPUT_URI,
                "--task-image",
                TASK_IMAGE,
                "--submit",
            ]
        )
    capsys.readouterr()


def test_distinct_output_prefixes_produce_distinct_job_names(monkeypatch: pytest.MonkeyPatch) -> None:
    root = evaluation.DEFAULT_CONFIG.parents[3]
    monkeypatch.setattr(evaluation, "resolve_launcher_source", lambda: SimpleNamespace(root=root, commit="abc123"))
    config = evaluation.load_evaluation_config(evaluation.DEFAULT_CONFIG)
    common = {
        "config_path": evaluation.DEFAULT_CONFIG,
        "gate": "smoke",
        "cluster_config": Path("/tmp/iris.yaml"),
        "task_image": TASK_IMAGE,
    }

    first = evaluation.build_plan(config, output_uri=f"{OUTPUT_URI}-first", **common)
    second = evaluation.build_plan(config, output_uri=f"{OUTPUT_URI}-second", **common)

    assert first.job_name != second.job_name


def test_checkpoint_evaluation_plan_selects_durable_actor_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    root = evaluation.DEFAULT_CONFIG.parents[3]
    monkeypatch.setattr(evaluation, "resolve_launcher_source", lambda: SimpleNamespace(root=root, commit="abc123"))
    checkpoint_uri = "s3://bucket/run/checkpoints/global_step_2/actor"

    plan = evaluation.build_plan(
        evaluation.load_evaluation_config(evaluation.DEFAULT_CONFIG),
        config_path=evaluation.DEFAULT_CONFIG,
        gate="full",
        cluster_config=Path("/tmp/iris.yaml"),
        output_uri=f"{OUTPUT_URI}-step-2",
        task_image=TASK_IMAGE,
        checkpoint_uri=checkpoint_uri,
        checkpoint_step=2,
    )

    assert plan.model_repository is None
    assert plan.model_revision is None
    assert plan.checkpoint_uri == checkpoint_uri
    assert plan.checkpoint_step == 2
    assert plan.iris_command[plan.iris_command.index("--checkpoint-uri") + 1] == checkpoint_uri
    assert plan.iris_command[plan.iris_command.index("--checkpoint-step") + 1] == "2"


def test_stage_checkpoint_model_downloads_only_model_state_and_merges(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    actor_root = "bucket/run/checkpoints/global_step_2/actor"
    files = {
        "bucket/run/checkpoints/latest_checkpointed_iteration.txt": b"4",
        f"{actor_root}/fsdp_config.json": b'{"world_size": 2}',
        f"{actor_root}/huggingface/config.json": b"{}",
        f"{actor_root}/huggingface/tokenizer.json": b"{}",
        f"{actor_root}/model_world_size_2_rank_0.pt": b"rank zero",
        f"{actor_root}/model_world_size_2_rank_1.pt": b"rank one",
        f"{actor_root}/optim_world_size_2_rank_0.pt": b"optimizer",
        f"{actor_root}/extra_state_world_size_2_rank_0.pt": b"extra",
    }

    class MemoryFilesystem:
        downloaded: list[str] = []

        def exists(self, path: str) -> bool:
            return path in files

        def open(self, path: str, encoding: str):
            del encoding
            return io.StringIO(files[path].decode())

        def find(self, target: str, *, detail: bool = False) -> list[str] | dict[str, dict[str, int | str]]:
            paths = [path for path in files if path.startswith(f"{target}/")]
            if detail:
                return {path: {"size": len(files[path]), "type": "file"} for path in paths}
            return paths

        def get_file(self, remote: str, local: str) -> None:
            self.downloaded.append(remote)
            Path(local).write_bytes(files[remote])

    filesystem = MemoryFilesystem()
    monkeypatch.setattr(evaluation_task, "fs_and_path", lambda _: (filesystem, actor_root))

    def merge(command: list[str], *, cwd: Path | None = None) -> None:
        assert command[command.index("--backend") + 1] == "fsdp"
        assert cwd == tmp_path / "source/training/verl"
        model = Path(command[command.index("--target_dir") + 1])
        model.mkdir()
        (model / "config.json").write_text("{}")
        (model / "model.safetensors").write_bytes(b"merged")

    monkeypatch.setattr(evaluation_task, "_run", merge)

    model, verification = stage_checkpoint_model(
        tmp_path / "source",
        "s3://bucket/run/checkpoints/global_step_2/actor",
        2,
        tmp_path / "checkpoint",
        tmp_path / "model",
    )

    assert model == tmp_path / "model"
    assert verification.step == 2
    assert verification.committed_through_step == 4
    assert {item.path for item in verification.files} == {
        "fsdp_config.json",
        "huggingface/config.json",
        "huggingface/tokenizer.json",
        "model_world_size_2_rank_0.pt",
        "model_world_size_2_rank_1.pt",
    }
    assert all("optim" not in path and "extra_state" not in path for path in filesystem.downloaded)
