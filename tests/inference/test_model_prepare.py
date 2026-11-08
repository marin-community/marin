# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""remote_inference mirrors uncached Hugging Face weights on CPU before reserving accelerators."""

from contextlib import contextmanager
from types import SimpleNamespace

import marin.inference.iris as iris_module
import pytest
from fray.types import CpuConfig, JobStatus, ResourceConfig, create_environment
from levanter.model_cache import DEFAULT_COMPLETE_MARKER
from marin.inference.config import (
    IrisConfig,
    RemoteInferenceConfig,
    ResolvedModelLocator,
    ServedModelConfig,
    SpeculativeMethod,
    SpeculativeServingConfig,
    VllmEngineConfig,
)
from marin.inference.iris import RemoteInferenceStartupError, remote_inference
from marin.inference.model_cache import model_cache_entry
from marin.inference.model_preparation import resolve_model_path
from rigging.filesystem.storage_path import StoragePath

REVISION = "a" * 40
DRAFT_REVISION = "b" * 40


class FakePrepareJob:
    def __init__(self, request, events: list[str], fail: bool):
        self.job_id = f"/prepare/{request.name}"
        self.marker = StoragePath(request.entrypoint.callable_entrypoint.args[0]) / DEFAULT_COMPLETE_MARKER
        self.events = events
        self.fail = fail

    def status(self) -> JobStatus:
        if self.fail:
            return JobStatus.FAILED
        self.marker.parent.mkdirs()
        self.marker.write_text("{}")
        return JobStatus.SUCCEEDED

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
    state = SimpleNamespace(events=[], requests=[], served=[], fail_prepare=False, fail_submit=False)

    def submit(request):
        state.events.append("prepare")
        state.requests.append(request)
        if state.fail_submit:
            raise RuntimeError("Iris submission failed")
        return FakePrepareJob(request, state.events, state.fail_prepare)

    @contextmanager
    def start_direct(config):
        state.events.append("gpu")
        state.served.append(config)
        yield SimpleNamespace(jobs=())

    monkeypatch.setattr(iris_module, "get_job_info", lambda: SimpleNamespace())
    monkeypatch.setattr(iris_module, "current_client", lambda: SimpleNamespace(submit=submit))
    monkeypatch.setattr(iris_module, "_start_direct_inference", start_direct)
    return state


def _marker(model: str, revision: str | None = None) -> StoragePath:
    entry = model_cache_entry(model, 14, revision)
    assert entry is not None
    return StoragePath(entry.path) / DEFAULT_COMPLETE_MARKER


def test_cold_model_is_mirrored_on_cpu_before_the_accelerator_job(serving):
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
    # The accelerator worker resolves the same entry the prepare job populated: a cache hit, no download.
    resolved = resolve_model_path(served.model.weights, served.iris.cache_ttl_days, served.model.revision)
    assert (StoragePath(resolved) / DEFAULT_COMPLETE_MARKER).exists()


def test_warm_model_skips_the_prepare_job(serving):
    marker = _marker("org/candidate", REVISION)
    marker.parent.mkdirs()
    marker.write_text("{}")

    with remote_inference(_config(REVISION)):
        pass

    assert serving.events == ["gpu"]


def test_speculative_draft_model_is_mirrored_and_pinned(serving):
    draft = SpeculativeServingConfig(
        method=SpeculativeMethod.EAGLE3,
        model=ResolvedModelLocator(uri=f"org/draft@{DRAFT_REVISION}", identity="draft"),
        num_speculative_tokens=3,
    )

    with remote_inference(_config(REVISION, speculative=draft)):
        pass

    assert serving.events == ["prepare", "prepare", "gpu"]
    assert _marker("org/draft", DRAFT_REVISION).exists()
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
