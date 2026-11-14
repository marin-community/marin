# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""remote_inference mirrors uncached Hugging Face weights on CPU before reserving accelerators."""

import os
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import marin.inference.iris as iris_module
import marin.inference.iris_cli as cli_module
import marin.inference.model_cache as cache_module
import pytest
from click.testing import CliRunner
from fray.types import ANY_REGION, CpuConfig, JobStatus, ResourceConfig, create_environment
from iris.resources.state import JobState
from levanter.model_cache import DEFAULT_COMPLETE_MARKER
from marin.evaluation.runner import _inference_session
from marin.inference.config import (
    IrisConfig,
    RemoteInferenceConfig,
    ResolvedModelLocator,
    ServedModelConfig,
    SpeculativeMethod,
    SpeculativeServingConfig,
    VllmEngineConfig,
)
from marin.inference.iris import InferenceStartupPhase, RemoteInferenceStartupError, remote_inference
from marin.inference.model_preparation import resolve_model_path
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import secret_file_contents, secret_file_environment
from rigging.timing import Duration

REVISION = "a" * 40
DRAFT_REVISION = "b" * 40


class FakePrepareJob:
    """Resolves the cache entry when it runs, under the prefix of the cluster it runs on."""

    def __init__(self, request, state):
        self.job_id = f"/prepare/{request.name}"
        self.entrypoint = request.entrypoint.callable_entrypoint
        self.state = state
        self.events = state.events
        self.result = None

    def status(self) -> JobStatus:
        if self.state.timeout_prepare:
            raise TimeoutError("prepare deadline exceeded")
        if self.result is None:
            with pytest.MonkeyPatch.context() as env:
                env.setenv("MARIN_PREFIX", self.state.worker_prefix)
                try:
                    self.entrypoint.callable(*self.entrypoint.args)
                    self.result = JobStatus.SUCCEEDED
                except OSError:
                    self.result = JobStatus.FAILED
        return self.result

    def terminate(self) -> None:
        self.events.append("prepare stopped")


def _config(revision: str | None, speculative: SpeculativeServingConfig | None = None) -> RemoteInferenceConfig:
    return RemoteInferenceConfig(
        model=ServedModelConfig(weights="org/candidate", revision=revision),
        engine=VllmEngineConfig(speculative=speculative),
        iris=IrisConfig(
            worker_resources=ResourceConfig.with_gpu("H100", count=1, regions=["us-east1"], target_cluster="cw-rno2a"),
            worker_environment=create_environment(env_vars={"HF_TOKEN": "test-token"}),
        ),
    )


@pytest.fixture
def serving(monkeypatch, tmp_path):
    """Record prepare submissions and the config the accelerator job would receive."""
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "cache"))
    state = SimpleNamespace(
        events=[],
        requests=[],
        served=[],
        downloads=[],
        fail_prepare=False,
        fail_submit=False,
        timeout_prepare=False,
        worker_prefix=str(tmp_path / "worker-cache"),
    )

    def stream(fs, dest, model_id, revision, complete_marker):
        if state.fail_prepare:
            raise OSError("snapshot download failed")
        state.downloads.append((model_id, revision, dest))
        fs.makedirs(dest, exist_ok=True)
        with fs.open(f"{dest}/config.json", "w") as config:
            config.write("{}")

    monkeypatch.setattr("levanter.model_cache._stream_hf_snapshot", stream)

    def submit(request):
        state.events.append("prepare")
        state.requests.append(request)
        if state.fail_submit:
            raise RuntimeError("Iris submission failed")
        return FakePrepareJob(request, state)

    @contextmanager
    def start_direct(config):
        state.events.append("gpu")
        state.served.append(config)
        yield SimpleNamespace(jobs=())

    monkeypatch.setattr(
        iris_module,
        "get_job_info",
        lambda: SimpleNamespace(worker_region="us-east1", constraints=[], task_id="/test/coordinator/0"),
    )
    monkeypatch.setattr(cache_module, "get_job_info", iris_module.get_job_info)
    monkeypatch.setattr(iris_module, "current_client", lambda: SimpleNamespace(submit=submit))
    monkeypatch.setattr(iris_module, "_start_direct_inference", start_direct)
    return state


def _seed(model: str, revision: str, ttl: int = 14) -> StoragePath:
    return StoragePath(resolve_model_path(model, ttl, revision)) / DEFAULT_COMPLETE_MARKER


def test_cold_model_is_mirrored_on_cpu_before_the_accelerator_job(serving, monkeypatch):
    with remote_inference(_config(REVISION)):
        pass

    assert serving.events == ["prepare", "gpu"]
    (request,) = serving.requests
    assert isinstance(request.resources.device, CpuConfig)
    assert request.resources.regions == ["us-east1"]
    assert request.resources.target_cluster == "cw-rno2a"
    assert request.environment.env_vars["HF_TOKEN"] == "test-token"
    (served,) = serving.served
    assert served.model.revision == REVISION
    assert served.model.model_id == "org/candidate"
    assert len(serving.downloads) == 1
    monkeypatch.setenv("MARIN_PREFIX", serving.worker_prefix)
    # The worker resolves the real job's output, without a second download.
    resolved = resolve_model_path(served.model.weights, served.iris.cache_ttl_days, served.model.revision)
    assert (StoragePath(resolved) / DEFAULT_COMPLETE_MARKER).exists()
    assert len(serving.downloads) == 1


def test_warm_colocated_coordinator_skips_cpu_preparation(serving):
    _seed("org/candidate", REVISION)
    serving.downloads.clear()
    serving.worker_prefix = os.environ["MARIN_PREFIX"]
    config = _config(REVISION)
    config = replace(
        config, iris=replace(config.iris, worker_resources=replace(config.iris.worker_resources, target_cluster=None))
    )
    with remote_inference(config):
        pass
    assert serving.events == ["gpu"]
    assert serving.downloads == []


def test_long_prepare_outlives_default_deadline_with_explicit_worker_budget(serving, monkeypatch):
    clock = SimpleNamespace(now=0.0)
    original_status = FakePrepareJob.status

    def status(job):
        return JobStatus.RUNNING if clock.now == 0 else original_status(job)

    def advance(_seconds):
        clock.now += 5 * 3600

    monkeypatch.setattr(FakePrepareJob, "status", status)
    monkeypatch.setattr("fray.client.time.monotonic", lambda: clock.now)
    monkeypatch.setattr("fray.client.time.sleep", advance)
    config = _config(REVISION)
    config = replace(config, iris=replace(config.iris, worker_timeout=Duration.from_hours(72)))
    with remote_inference(config):
        pass
    assert serving.events == ["prepare", "gpu"]
    assert len(serving.downloads) == 1


def test_warm_coordinator_does_not_skip_named_target_preparation(serving, monkeypatch):
    _seed("org/candidate", REVISION)
    serving.downloads.clear()
    with remote_inference(_config(REVISION)):
        pass
    assert serving.events == ["prepare", "gpu"]
    assert len(serving.downloads) == 1
    monkeypatch.setenv("MARIN_PREFIX", serving.worker_prefix)
    assert _seed("org/candidate", REVISION).exists()
    assert len(serving.downloads) == 1


def test_speculative_draft_model_is_mirrored_and_pinned(serving, monkeypatch):
    draft = SpeculativeServingConfig(
        method=SpeculativeMethod.EAGLE3,
        model=ResolvedModelLocator(uri=f"org/draft@{DRAFT_REVISION}", identity="draft"),
        num_speculative_tokens=3,
    )

    with remote_inference(_config(REVISION, speculative=draft)):
        pass

    assert serving.events == ["prepare", "prepare", "gpu"]
    monkeypatch.setenv("MARIN_PREFIX", serving.worker_prefix)
    assert _seed("org/draft", DRAFT_REVISION).exists()
    assert len(serving.downloads) == 2
    (served,) = serving.served
    assert served.engine.speculative.model.uri == f"org/draft@{DRAFT_REVISION}"


@pytest.mark.parametrize("failure", ["job", "submit", "resolve"])
def test_prepare_failure_never_reserves_accelerators(serving, monkeypatch, failure):
    revision = REVISION
    if failure == "job":
        serving.fail_prepare = True
    elif failure == "submit":
        serving.fail_submit = True
    else:
        revision = None

        def fail_revision(*_args, **_kwargs):
            raise RuntimeError("Hugging Face revision lookup failed")

        monkeypatch.setattr("levanter.model_cache.model_info", fail_revision)

    with pytest.raises(RemoteInferenceStartupError, match="Model cache preparation failed") as exc_info:
        with remote_inference(_config(revision)):
            pass

    assert "gpu" not in serving.events
    if failure == "job":
        assert [job.job_id for job in exc_info.value.jobs] == [f"/prepare/{serving.requests[0].name}"]
        assert serving.events == ["prepare", "prepare stopped"]
    else:
        assert exc_info.value.jobs == ()


def test_prepare_timeout_is_not_retried_as_a_serving_readiness_timeout(serving):
    serving.timeout_prepare = True
    config = _config(REVISION)
    config = replace(config, iris=replace(config.iris, max_retries_failure=1))
    with ExitStack() as stack, pytest.raises(RemoteInferenceStartupError) as caught:
        _inference_session(stack, config)
    assert caught.value.phase is InferenceStartupPhase.PREPARATION
    assert isinstance(caught.value.__cause__.__cause__, TimeoutError)
    assert serving.events == ["prepare", "prepare stopped"]
    assert len(serving.requests) == 1


@pytest.mark.parametrize(
    "device,regions,target_cluster,prepares",
    [
        ("gpu", ["us-east1"], None, True),
        ("tpu", [ANY_REGION], None, False),
        ("tpu", ["us-east1", "us-west1"], None, False),
        ("tpu", ["us-east1"], None, True),
        ("tpu", None, None, True),
        ("gpu", [ANY_REGION], "cw-rno2a", True),
        ("gpu", [ANY_REGION], None, True),
    ],
)
def test_preparation_requires_a_matchable_serving_placement(serving, device, regions, target_cluster, prepares):
    config = _config(REVISION)
    resources = (
        ResourceConfig.with_gpu("H100", count=1, regions=regions, target_cluster=target_cluster)
        if device == "gpu"
        else ResourceConfig.with_tpu("v4-8", regions=regions, target_cluster=target_cluster)
    )
    config = replace(config, iris=replace(config.iris, worker_resources=resources))
    with remote_inference(config):
        pass
    assert serving.events == (["prepare", "gpu"] if prepares else ["gpu"])
    assert len(serving.downloads) == int(prepares)


@pytest.mark.parametrize("device", ["gpu", "tpu"])
def test_root_unpinned_accelerator_does_not_submit_a_prepare_job(serving, monkeypatch, device):
    monkeypatch.setattr(iris_module, "get_job_info", lambda: None)
    monkeypatch.setattr(cache_module, "get_job_info", lambda: None)
    config = _config(REVISION)
    resources = (
        ResourceConfig.with_gpu("H100", count=1, regions=["us-east1"])
        if device == "gpu"
        else ResourceConfig.with_tpu("v4-8")
    )
    config = replace(config, iris=replace(config.iris, worker_resources=resources))
    iris_module._prepared_inference_config(config)
    assert serving.requests == []
    assert serving.downloads == []


@pytest.mark.parametrize(
    "placement,prepares,fail",
    [
        ([], False, False),
        (["--region", "us-central2"], True, False),
        (["--region", "us-central2,us-east1"], False, False),
        (["--region", "*"], False, False),
        (["--target-cluster", "cw-rno2a", "--gpu", "H100x1"], True, False),
        (["--gpu", "H100x1", "--region", "us-east5"], False, False),
        (["--region", "us-central2"], True, True),
    ],
)
def test_cli_prepares_only_a_pinned_cache_before_accelerator_submission(serving, monkeypatch, placement, prepares, fail):
    serving.fail_prepare = fail
    monkeypatch.setattr(cache_module, "get_job_info", lambda: None)
    monkeypatch.setattr("levanter.model_cache.model_info", lambda *_a, **_k: SimpleNamespace(sha=REVISION))

    class Job:
        job_id = "/test/prepare"

        def __init__(self, entrypoint):
            self.entrypoint = entrypoint

        def state_only(self):
            with pytest.MonkeyPatch.context() as env:
                env.setenv("MARIN_PREFIX", serving.worker_prefix)
                function, args, kwargs = self.entrypoint.resolve()
                try:
                    function(*args, **kwargs)
                except OSError:
                    return JobState.FAILED
            return JobState.SUCCEEDED

        def cancel(self):
            serving.events.append("prepare stopped")

    def submit(**kwargs):
        prepare = kwargs["name"].startswith("prepare-model-")
        serving.events.append("prepare" if prepare else "gpu")
        return Job(kwargs["entrypoint"])

    client = SimpleNamespace(submit=submit, resolve_endpoint=lambda _endpoint: "https://controller/proxy/serve.test")

    @contextmanager
    def connect(**_kwargs):
        yield SimpleNamespace(url="https://controller", config=None, credentials=None)

    @contextmanager
    def remote(*_args, **_kwargs):
        yield client

    monkeypatch.setattr(cli_module, "connect_controller", connect)
    monkeypatch.setattr(cli_module.IrisClient, "remote", remote)
    result = CliRunner().invoke(cli_module.main, ["org/candidate", "--no-wait", *placement])
    if prepares and fail:
        assert result.exit_code != 0
        assert serving.events == ["prepare", "prepare stopped"]
    else:
        assert result.exit_code == 0, result.output
        assert serving.events == (["prepare", "gpu"] if prepares else ["gpu"])
        assert len(serving.downloads) == int(prepares)
    if not prepares:
        assert "skipping CPU model preparation" in result.output
        assert "Preparing Hugging Face weights on CPU" not in result.output


@pytest.mark.parametrize("failure", [None, "revision", "download"])
def test_cli_private_model_uses_explicit_secret_through_preparation(monkeypatch, tmp_path, caplog, failure):
    token = "hf-explicit-test-token"
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("MODEL_TOKEN", token)
    monkeypatch.setattr(huggingface_hub.constants, "HF_TOKEN_PATH", str(tmp_path / "absent-token"))
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "cache"))
    monkeypatch.setattr(cache_module, "get_job_info", lambda: None)
    events = []
    submissions = []

    def model_info(repo, *, revision=None, token=None):
        assert huggingface_hub.get_token() is None
        assert repo == "org/private"
        if token != "hf-explicit-test-token":
            raise OSError("private revision requires explicit authentication")
        events.append("revision")
        if failure == "revision":
            raise OSError("private revision access denied")
        return SimpleNamespace(sha=REVISION)

    def list_repo_tree(repo, *, revision, recursive):
        assert repo == "org/private" and revision == REVISION
        assert huggingface_hub.get_token() == token
        assert "HF_TOKEN" not in os.environ
        events.append("list")
        return [huggingface_hub.RepoFile(path="config.json", size=2, oid="0" * 40)]

    def download(repo, filename, *, revision, local_dir):
        assert huggingface_hub.get_token() == token
        assert repo == "org/private" and revision == REVISION
        events.append("download")
        if failure == "download":
            raise OSError("private download access denied")
        path = Path(local_dir) / filename
        path.write_text("{}")
        return str(path)

    monkeypatch.setattr("levanter.model_cache.model_info", model_info)
    monkeypatch.setattr("levanter.model_cache.list_repo_tree", list_repo_tree)
    monkeypatch.setattr("levanter.model_cache.hf_hub_download", download)

    class Job:
        job_id = "/test/private-prepare"

        def __init__(self, submission):
            self.submission = submission

        def state_only(self):
            # Emulate Iris's file delivery in a fresh worker, including HF's import-time path.
            environment = self.submission["environment"]
            directory = tmp_path / "secrets"
            directory.mkdir(exist_ok=True)
            for name, value in secret_file_contents(environment.secrets).items():
                (directory / name).write_text(value)
            with pytest.MonkeyPatch.context() as env:
                env.delenv("MODEL_TOKEN", raising=False)
                for name, value in secret_file_environment(environment.secrets, directory=directory).items():
                    env.setenv(name, value)
                env.setattr(huggingface_hub.constants, "HF_TOKEN_PATH", str(directory / "HF_TOKEN"))
                function, args, kwargs = self.submission["entrypoint"].resolve()
                try:
                    function(*args, **kwargs)
                except OSError:
                    return JobState.FAILED
            return JobState.SUCCEEDED

        def cancel(self):
            events.append("cancel")

    def submit(**kwargs):
        submissions.append(kwargs)
        events.append("prepare" if kwargs["name"].startswith("prepare-model-") else "serve")
        return Job(kwargs)

    @contextmanager
    def connect(**_kwargs):
        yield SimpleNamespace(url="https://controller", config=None, credentials=None)

    @contextmanager
    def remote(*_args, **_kwargs):
        yield SimpleNamespace(submit=submit, resolve_endpoint=lambda _endpoint: "https://controller/proxy/test")

    monkeypatch.setattr(cli_module, "connect_controller", connect)
    monkeypatch.setattr(cli_module.IrisClient, "remote", remote)
    result = CliRunner().invoke(
        cli_module.main,
        [
            "org/private",
            "--no-wait",
            "--region",
            "us-central2",
            "--secret",
            "HF_TOKEN",
            "env:MODEL_TOKEN",
            "--secret",
            "SERVING_ONLY",
            "env:MODEL_TOKEN",
        ],
    )
    assert token not in result.output + caplog.text
    assert huggingface_hub.get_token() is None
    assert "HF_TOKEN" not in os.environ
    for submission in submissions:
        assert token not in repr(submission["environment"])
        assert token not in repr(submission["entrypoint"].resolve())
        assert token not in repr(submission["environment"].env_vars)
    if failure is None:
        assert result.exit_code == 0, result.output
        assert events == ["revision", "prepare", "list", "download", "serve"]
        assert submissions[0]["environment"].secrets == {"HF_TOKEN": token}
        assert submissions[1]["environment"].secrets == {"HF_TOKEN": token, "SERVING_ONLY": token}
        _, (service,), _ = submissions[1]["entrypoint"].resolve()
        assert service.model.revision == REVISION
        # A completed snapshot is usable without credentials or another Hub download.
        resolved = resolve_model_path(service.model.weights, service.iris.cache_ttl_days, service.model.revision)
        assert (Path(resolved) / "config.json").read_text() == "{}"
    else:
        assert result.exit_code != 0
        assert events == (
            ["revision"] if failure == "revision" else ["revision", "prepare", "list", "download", "cancel"]
        )
