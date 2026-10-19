import hashlib
import io
import json
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

import cloud.iris.open_mopd_fidelity as fidelity
import cloud.iris.open_mopd_fidelity_task as fidelity_task
from cloud.iris.open_mopd_fidelity_task import (
    FileVerification,
    StagedInputs,
    patch_source_for_reference,
    restore_latest_checkpoint,
    sync_tree,
    training_command,
    validate_runtime,
    validate_resume_manifest,
    verify_lfs_files,
    verify_shared_tokenizer,
)

TASK_IMAGE = "registry.example/open-mopd@sha256:" + "1" * 64
OUTPUT_URI = "s3://bucket/open-mopd/one-step"
RELEASED_SCORER_PACKAGES = {
    "absl-py",
    "appdirs",
    "emoji",
    "immutabledict",
    "jsonlines",
    "langdetect",
    "nltk",
    "syllapy",
    "tempdir",
    "wget",
}


def test_runtime_accepts_cuda_local_version_for_public_release(monkeypatch: pytest.MonkeyPatch) -> None:
    config = fidelity.load_config(fidelity.DEFAULT_CONFIG)
    versions = config.environment.packages | {"torch": "2.8.0+cu128"}
    monkeypatch.setattr(fidelity_task.importlib.metadata, "version", versions.__getitem__)

    validate_runtime(config)


def test_runtime_manifest_includes_released_scorer_dependencies() -> None:
    config = fidelity.load_config(fidelity.DEFAULT_CONFIG)

    assert RELEASED_SCORER_PACKAGES <= config.environment.packages.keys()


def test_inline_aime_uses_released_validation_artifact() -> None:
    config = fidelity.load_config(fidelity.DEFAULT_CONFIG)
    evaluation = json.loads(fidelity.DEFAULT_CONFIG.with_name("open_mopd_evaluation.json").read_text())
    aime = next(benchmark for benchmark in evaluation["benchmarks"] if benchmark["name"] == "aime24")

    assert (config.validation_dataset.repository, config.validation_dataset.revision) == (
        evaluation["data"]["repository"],
        evaluation["data"]["revision"],
    )
    assert (config.validation_dataset.path, config.validation_dataset.size, config.validation_dataset.sha256) == (
        aime["path"],
        aime["size"],
        aime["sha256"],
    )


def test_reference_source_patches_route_domain_response_limits(tmp_path: Path) -> None:
    launcher = tmp_path / "scripts" / "local" / "mt_opd.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text('cmd+=("actor_rollout_ref.rollout.reward_mode=mt_opd")\n')
    trainer = tmp_path / "training" / "verl" / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text(
        'reward_model_keys = ({"data_source", "reward_model", "extra_info", "uid", "domain"} '
        "& batch.non_tensor_batch.keys())\n"
        "        if self.async_rollout_mode:\n"
        "            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)\n"
    )
    rollout = tmp_path / "training" / "verl" / "verl" / "workers" / "rollout" / "vllm_rollout" / "vllm_rollout_spmd.py"
    rollout.parent.mkdir(parents=True)
    rollout.write_text(
        "class PatchedRollout:\n"
        "    def generate(self, kwargs, is_validate, non_tensor_batch, batch_size, vllm_inputs):\n"
        "        with self.update_sampling_params(**kwargs):\n"
        "            outputs = self.inference_engine.generate(\n"
        "                prompts=vllm_inputs,\n"
        "                sampling_params=self.sampling_params,\n"
        "            )\n"
        "        return outputs\n"
    )
    rollout_config = tmp_path / "training" / "verl" / "verl" / "workers" / "config" / "rollout.py"
    rollout_config.parent.mkdir(parents=True)
    rollout_config.write_text(
        "from dataclasses import dataclass, field\n@dataclass\nclass RolloutConfig:\n    response_length: int = 512\n"
    )

    patch_source_for_reference(tmp_path)

    assert launcher.read_text() == 'cmd+=("+actor_rollout_ref.rollout.reward_mode=mt_opd")\n'
    assert '"domain", "raw_prompt"' in trainer.read_text()
    assert 'gen_batch.non_tensor_batch["domain"]' in trainer.read_text()
    config_namespace: dict[str, object] = {}
    exec(compile(rollout_config.read_text(), str(rollout_config), "exec"), config_namespace)
    config_namespace["RolloutConfig"](domain_response_limits={"math": 16384, "code": 16384, "if": 2048})

    class FakeSamplingParams:
        def __init__(self, max_tokens: int):
            self.max_tokens = max_tokens

        def clone(self) -> "FakeSamplingParams":
            return FakeSamplingParams(self.max_tokens)

    namespace: dict[str, object] = {}
    exec(compile(rollout.read_text(), str(rollout), "exec"), namespace)
    patched = namespace["PatchedRollout"]()
    patched.config = SimpleNamespace(domain_response_limits={"math": 16384, "code": 16384, "if": 2048})
    patched.sampling_params = FakeSamplingParams(16384)
    patched.update_sampling_params = lambda **kwargs: nullcontext()
    patched.inference_engine = SimpleNamespace(generate=lambda **kwargs: kwargs["sampling_params"])

    routed = patched.generate({}, False, {"domain": ["math", "code", "if"]}, 3, [1, 2, 3])
    assert [params.max_tokens for params in routed] == [16384, 16384, 2048]
    assert patched.sampling_params.max_tokens == 16384
    assert patched.generate({}, True, {}, 1, [1]) is patched.sampling_params
    with pytest.raises(ValueError, match="one domain per training request"):
        patched.generate({}, False, {"domain": ["math"]}, 2, [1, 2])


@pytest.mark.parametrize(
    "lfs_files",
    [
        [],
        [{"path": ".", "size": 7, "sha256": "1" * 64}],
        [{"path": "../model.safetensors", "size": 7, "sha256": "1" * 64}],
        [
            {"path": "model.safetensors", "size": 7, "sha256": "1" * 64},
            {"path": "model.safetensors", "size": 7, "sha256": "2" * 64},
        ],
    ],
)
def test_config_parser_rejects_unsafe_or_ambiguous_lfs_manifests(
    tmp_path: Path, lfs_files: list[dict[str, object]]
) -> None:
    raw = json.loads(fidelity.DEFAULT_CONFIG.read_text())
    raw["artifacts"]["student"]["lfs_files"] = lfs_files
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw))

    with pytest.raises(ValueError):
        fidelity.load_config(config_path)


def test_model_verification_reports_observed_integrity(tmp_path: Path) -> None:
    content = b"weights"
    model = tmp_path / "model.safetensors"
    model.write_bytes(content)
    expected = fidelity.LfsFile(
        path=model.name,
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )

    verified = verify_lfs_files(tmp_path, (expected,))

    assert verified == (
        FileVerification(
            path=model.name,
            expected_size=len(content),
            observed_size=len(content),
            expected_sha256=expected.sha256,
            observed_sha256=expected.sha256,
        ),
    )


def test_reference_requires_identical_student_and_teacher_tokenizers(tmp_path: Path) -> None:
    models = [tmp_path / name for name in ("student", "math", "code", "if")]
    for model in models:
        model.mkdir()
        (model / "tokenizer.json").write_bytes(b"vocabulary")
        (model / "tokenizer_config.json").write_bytes(b"template")

    verify_shared_tokenizer(models[0], tuple(models[1:]))

    (models[1] / "tokenizer_config.json").write_bytes(b"different template")
    with pytest.raises(ValueError, match="tokenizer mismatch"):
        verify_shared_tokenizer(models[0], tuple(models[1:]))

    (models[1] / "tokenizer_config.json").unlink()
    with pytest.raises(ValueError, match="missing tokenizer"):
        verify_shared_tokenizer(models[0], tuple(models[1:]))


@pytest.mark.parametrize("failure", ["missing", "size", "digest", "internal_symlink", "external_symlink"])
def test_model_verification_rejects_untrusted_files(tmp_path: Path, failure: str) -> None:
    content = b"weights"
    expected_path = "model.safetensors"
    expected_size = len(content)
    expected_sha256 = hashlib.sha256(content).hexdigest()
    if failure == "size":
        (tmp_path / expected_path).write_bytes(content + b"!")
    elif failure == "digest":
        (tmp_path / expected_path).write_bytes(b"WEIGHTS")
    elif failure == "internal_symlink":
        actual = tmp_path / "actual.safetensors"
        actual.write_bytes(content)
        (tmp_path / expected_path).symlink_to(actual)
    elif failure == "external_symlink":
        outside = tmp_path.parent / f"{tmp_path.name}-outside"
        outside.write_bytes(content)
        (tmp_path / expected_path).symlink_to(outside)
    expected = fidelity.LfsFile(path=expected_path, size=expected_size, sha256=expected_sha256)

    with pytest.raises(ValueError, match="integrity verification failed"):
        verify_lfs_files(tmp_path, (expected,))


def test_dataset_verification_is_recorded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    content = b"parquet"
    relative_path = "rl_prompt_mix/train.parquet"
    dataset = tmp_path / relative_path
    dataset.parent.mkdir()
    dataset.write_bytes(content)
    artifact = SimpleNamespace(
        repository="organization/dataset",
        revision="1" * 40,
        path=relative_path,
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
    monkeypatch.setattr(fidelity_task, "_run", lambda *args, **kwargs: None)

    path, verification = fidelity_task._dataset(artifact, tmp_path)

    assert path == dataset
    assert verification.repository == artifact.repository
    assert verification.revision == artifact.revision
    assert verification.files == (
        FileVerification(
            path=relative_path,
            expected_size=len(content),
            observed_size=len(content),
            expected_sha256=artifact.sha256,
            observed_sha256=artifact.sha256,
        ),
    )


def test_training_command_routes_config_budgets_domains_and_gate_schedule() -> None:
    config = fidelity.load_config(fidelity.DEFAULT_CONFIG)
    training = config.training
    inputs = StagedInputs(
        source=Path("/work/Open-MOPD"),
        student=Path("/work/student"),
        teachers=(Path("/work/math"), Path("/work/code"), Path("/work/if")),
        dataset=Path("/work/train.parquet"),
        validation_dataset=Path("/work/aime24.parquet"),
    )
    command = training_command(config, inputs, "one_step", Path("/work/output"), world_size=8)
    bash_index = command.index("bash")
    environment = dict(value.split("=", 1) for value in command[1:bash_index])
    overrides = dict(command[index + 1].split("=", 1) for index, value in enumerate(command) if value == "--extra")

    assert environment["TRAIN_BATCH_SIZE"] == str(training.train_batch_size)
    assert environment["MAX_PROMPT_LENGTH"] == str(training.prompt_limit)
    assert environment["MAX_RESPONSE_LENGTH"] == str(training.response_limit)
    assert overrides["actor_rollout_ref.rollout.max_num_batched_tokens"] == str(
        training.prompt_limit + training.response_limit
    )
    # The default config's per-domain response limits and sampling weights, as Hydra reads them.
    assert overrides["+actor_rollout_ref.rollout.domain_response_limits"] == "{math:16384,code:16384,if:2048}"
    assert [overrides[f"+data.domain_weights.{d}"] for d in ("math", "code", "if")] == ["2", "2", "1"]
    # A one-step gate clamps the save and eval cadence so its only step is checkpointed and validated.
    assert overrides["trainer.total_training_steps"] == "1"
    assert overrides["trainer.save_freq"] == "1"
    assert overrides["trainer.test_freq"] == "1"
    assert command[command.index("--train") + 1] == str(inputs.dataset)
    assert command[command.index("--val") + 1] == str(inputs.validation_dataset)
    assert command[command.index("--gpus") + 1] == "8"


def test_all_acceptance_gates_resolve_steps_and_checkpoints() -> None:
    config = fidelity.load_config(fidelity.DEFAULT_CONFIG)
    inputs = StagedInputs(
        Path("/source"),
        Path("/student"),
        tuple(Path(f"/{d}") for d in fidelity.DOMAINS),
        Path("/data"),
        Path("/aime24"),
    )
    for gate, steps, expected_eval_frequency in (
        ("one_step", 1, 1),
        ("paper_checkpoint", 200, 2),
        ("paper_schedule", 600, 2),
    ):
        command = training_command(config, inputs, gate, Path("/output"), world_size=8)
        assert f"trainer.total_training_steps={steps}" in command
        assert f"trainer.save_freq={min(steps, config.training.save_every)}" in command
        assert f"trainer.test_freq={expected_eval_frequency}" in command


def test_restore_latest_checkpoint_downloads_only_committed_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class MemoryFilesystem:
        files = {
            "bucket/run/checkpoints/global_step_2/actor/model.pt": b"old",
            "bucket/run/checkpoints/global_step_4/actor/model.pt": b"weights",
            "bucket/run/checkpoints/global_step_4/data.pt": b"dataloader",
            "bucket/run/checkpoints/latest_checkpointed_iteration.txt": b"4",
        }

        def find(self, target: str) -> list[str]:
            assert target == "bucket/run"
            return list(self.files)

        def open(self, path: str, encoding: str):
            return io.StringIO(self.files[path].decode(encoding))

        def get_file(self, remote: str, local: str) -> None:
            Path(local).write_bytes(self.files[remote])

    filesystem = MemoryFilesystem()
    monkeypatch.setattr(fidelity_task, "fs_and_path", lambda _: (filesystem, "bucket/run"))

    step = restore_latest_checkpoint(OUTPUT_URI, tmp_path)

    assert step == 4
    assert (tmp_path / "checkpoints/global_step_4/actor/model.pt").read_bytes() == b"weights"
    assert (tmp_path / "checkpoints/global_step_4/data.pt").read_bytes() == b"dataloader"
    assert (tmp_path / "checkpoints/latest_checkpointed_iteration.txt").read_text() == "4"
    assert not (tmp_path / "checkpoints/global_step_2").exists()


def test_sync_tree_skips_published_checkpoint_files_and_commits_pointer_last(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class MemoryFilesystem:
        files = {
            "bucket/run/checkpoints/global_step_2/actor/model.pt": b"old-local",
            "bucket/run/checkpoints/latest_checkpointed_iteration.txt": b"2",
            "bucket/run/control-manifest.json": b"old-manifest",
        }
        uploads: list[str] = []

        def find(self, target: str, *, detail: bool, withdirs: bool) -> dict[str, dict[str, int]]:
            assert target == "bucket/run"
            assert detail
            assert not withdirs
            return {path: {"size": len(payload)} for path, payload in self.files.items()}

        def makedirs(self, _path: str, *, exist_ok: bool) -> None:
            assert exist_ok

        def put_file(self, local: str, remote: str) -> None:
            self.files[remote] = Path(local).read_bytes()
            self.uploads.append(remote)
            if remote == "bucket/run/control-manifest.json":
                pointer.write_text("6")

        def pipe_file(self, remote: str, payload: bytes) -> None:
            self.files[remote] = payload
            self.uploads.append(remote)

    checkpoint_root = tmp_path / "checkpoints"
    step_2 = checkpoint_root / "global_step_2" / "actor" / "model.pt"
    step_2.parent.mkdir(parents=True)
    step_2.write_bytes(b"old-local")
    step_4 = checkpoint_root / "global_step_4" / "actor" / "model.pt"
    step_4.parent.mkdir(parents=True)
    step_4.write_bytes(b"new-weights")
    step_6 = checkpoint_root / "global_step_6" / "actor" / "model.pt"
    step_6.parent.mkdir(parents=True)
    step_6.write_bytes(b"still-writing")
    pointer = checkpoint_root / fidelity_task.LATEST_CHECKPOINT_NAME
    pointer.write_text("4")
    (tmp_path / fidelity_task.CONTROL_MANIFEST_NAME).write_bytes(b"new-manifest")
    filesystem = MemoryFilesystem()
    monkeypatch.setattr(fidelity_task, "fs_and_path", lambda _: (filesystem, "bucket/run"))

    sync_tree(tmp_path, OUTPUT_URI)

    assert "bucket/run/checkpoints/global_step_2/actor/model.pt" not in filesystem.uploads
    assert filesystem.files["bucket/run/checkpoints/global_step_4/actor/model.pt"] == b"new-weights"
    assert "bucket/run/checkpoints/global_step_6/actor/model.pt" not in filesystem.files
    assert filesystem.files["bucket/run/control-manifest.json"] == b"new-manifest"
    assert pointer.read_text() == "6"
    assert filesystem.files["bucket/run/checkpoints/latest_checkpointed_iteration.txt"] == b"4"
    assert filesystem.uploads[-1] == "bucket/run/checkpoints/latest_checkpointed_iteration.txt"
    pointer.unlink()
    original_rglob = Path.rglob

    def rglob_after_checkpoint(tmp_path_: Path, pattern: str):
        pointer.write_text("2")
        return original_rglob(tmp_path_, pattern)

    filesystem.uploads.clear()
    monkeypatch.setattr(Path, "rglob", rglob_after_checkpoint)

    sync_tree(tmp_path, OUTPUT_URI)

    assert "bucket/run/checkpoints/latest_checkpointed_iteration.txt" not in filesystem.uploads
    assert filesystem.files["bucket/run/checkpoints/latest_checkpointed_iteration.txt"] == b"4"


def test_resume_manifest_rejects_mismatched_run_identity() -> None:
    identity = {"gate": "paper_checkpoint", "steps": 200}

    validate_resume_manifest(identity, identity, OUTPUT_URI)
    with pytest.raises(ValueError, match="steps"):
        validate_resume_manifest(identity, identity | {"steps": 201}, OUTPUT_URI)
    with pytest.raises(ValueError, match="completed control"):
        validate_resume_manifest(identity | {"returncode": 0}, identity, OUTPUT_URI)


def test_periodic_sync_retries_after_upload_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class StopAfterTwoAttempts:
        waits = 0

        def wait(self, _interval: int) -> bool:
            self.waits += 1
            return self.waits > 2

    attempts = 0

    def sync_tree(_local: Path, _output_uri: str) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("transient failure")

    monkeypatch.setattr(fidelity_task, "sync_tree", sync_tree)

    fidelity_task.periodic_sync(tmp_path, OUTPUT_URI, StopAfterTwoAttempts(), interval=1)

    assert attempts == 2


def test_gpu_override_records_deviation_and_enforces_authors_world_size() -> None:
    config = fidelity.load_config(fidelity.DEFAULT_CONFIG)
    plan = fidelity.build_plan(
        config,
        config_path=fidelity.DEFAULT_CONFIG,
        gate="one_step",
        cluster_config=Path("/tmp/iris.yaml"),
        output_uri=OUTPUT_URI,
        task_image=TASK_IMAGE,
        gpu_slice="H100x8",
    )

    assert plan.gpu_slice == "H100x8"
    assert any("H100x8" in deviation for deviation in plan.known_deviations)
    assert "--no-sync" in plan.iris_command
    assert plan.iris_command[plan.iris_command.index("--priority") + 1] == "interactive"
    assert plan.iris_command[plan.iris_command.index("--gpu-slice") + 1] == "H100x8"
    with pytest.raises(ValueError, match="8-GPU"):
        fidelity.gpu_count("H100x4")
    with pytest.raises(ValueError, match="Malformed"):
        fidelity.gpu_count("H100")


@pytest.mark.parametrize("output_uri", ["/tmp/output", "file:///tmp/output", "s3://bucket"])
def test_plan_rejects_non_durable_or_root_output_prefix(output_uri: str) -> None:
    with pytest.raises(ValueError, match="durable prefix"):
        fidelity.build_plan(
            fidelity.load_config(fidelity.DEFAULT_CONFIG),
            config_path=fidelity.DEFAULT_CONFIG,
            gate="one_step",
            cluster_config=Path("/tmp/iris.yaml"),
            output_uri=output_uri,
            task_image=TASK_IMAGE,
        )


def test_cli_dry_run_does_not_submit(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    root = fidelity.DEFAULT_CONFIG.parents[3]
    monkeypatch.setattr(fidelity, "resolve_launcher_source", lambda: SimpleNamespace(root=root, commit="abc123"))
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("dry run submitted a process"))

    assert (
        fidelity.main(["--cluster-config", "/tmp/iris.yaml", "--output-uri", OUTPUT_URI, "--task-image", TASK_IMAGE])
        == 0
    )
    assert "Dry run only" in capsys.readouterr().out


def test_submission_requires_deviation_acknowledgement(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = fidelity.DEFAULT_CONFIG.parents[3]
    monkeypatch.setattr(fidelity, "resolve_launcher_source", lambda: SimpleNamespace(root=root, commit="abc123"))
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("guard did not stop submission"))

    with pytest.raises(SystemExit, match="allow-known-deviations"):
        fidelity.main(
            ["--cluster-config", "/tmp/iris.yaml", "--output-uri", OUTPUT_URI, "--task-image", TASK_IMAGE, "--submit"]
        )
    capsys.readouterr()
