# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Model admission follows the running server's context limit."""

from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from evalrecords.records import (
    CuratedAgentRef,
    CuratedSettings,
    TaskImageBuild,
    TaskMachineBackend,
    TaskMachineSettings,
)
from marin.evaluation.dataset_evaluator import DatasetEvaluator, runtime_config
from marin.evaluation.model_config import AgentConfig, GenerationConfig, ModelConfig, ServeConfig
from marin.evaluation.task_worker import DAYTONA_API_KEY, DAYTONA_EGRESS, DaytonaConnection, machine_factories
from marin.inference.types import InferenceBackendState
from openai import OpenAI
from rolloutengine.agents.mini_swe.agent import MiniSweModelFormat

SERVED_CONTEXT = 4096
MODEL = ModelConfig(
    name="solver",
    location="example/solver",
    revision="pinned",
    tokenizer_revision="tokenizer-pin",
    serve=ServeConfig(max_model_len=8192),
    generation=GenerationConfig(max_gen_toks=8192),
)
SESSION = SimpleNamespace(
    model=SimpleNamespace(endpoint=SimpleNamespace(base_url="http://server/v1", api_key=None, model="served-alias"))
)
AGENT = CuratedAgentRef(harness="terminus-2", revision="1", max_input_tokens=4096, max_output_tokens=2048)


@pytest.fixture
def created(monkeypatch) -> list[int]:
    """Serve a model list whose creation timestamp the test can change."""
    created = [100]

    def endpoint(request):
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": "served-alias",
                        "object": "model",
                        "created": created[0],
                        "owned_by": "vllm",
                        "max_model_len": SERVED_CONTEXT,
                    }
                ],
            },
        )

    def client(**kwargs):
        return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(endpoint)))

    monkeypatch.setattr("marin.evaluation.dataset_evaluator.OpenAI", client)
    return created


@pytest.fixture
def submitted(monkeypatch) -> dict:
    """Capture the evaluation worker submission and finish it immediately."""
    request: dict = {}

    class Client:
        def submit(self, **kwargs):
            request.update(kwargs)
            return SimpleNamespace(job_id="/test/eval", wait=lambda **_kwargs: None)

    monkeypatch.setattr("marin.evaluation.dataset_evaluator.iris_ctx", lambda: SimpleNamespace(client=Client()))
    return request


def test_context_admission_uses_server_limit_and_resume_identity_ignores_server_timestamp(created):
    settings = CuratedSettings(max_tokens=8192, source_limit=2, seed=42, judge=None)
    first = runtime_config(MODEL, settings, SESSION)
    assert first.model.context_window == SERVED_CONTEXT
    assert first.max_tokens == SERVED_CONTEXT - 1
    created[0] += 100
    assert runtime_config(MODEL, settings, SESSION) == first


@pytest.mark.parametrize(
    ("agent_kwargs", "model_format"),
    [({}, MiniSweModelFormat.OPENAI), ({"model_format": "k2_horizon"}, MiniSweModelFormat.K2_HORIZON)],
)
def test_runs_take_mini_swe_agents_history_format_from_the_models_agent_arguments(created, agent_kwargs, model_format):
    model = replace(MODEL, agent=AgentConfig(agent_kwargs=agent_kwargs))

    config = runtime_config(model, CuratedSettings(), SESSION)

    assert config.mini_swe_model_format is model_format
    # The default format leaves the identity of runs that predate the setting unchanged.
    assert ("mini_swe_model_format" in config.model_dump(mode="json")) is (model_format is MiniSweModelFormat.K2_HORIZON)


@pytest.mark.parametrize("backend", list(TaskMachineBackend))
def test_agent_worker_runs_planned_task_machines_with_daytona_credentials_only_for_daytona(created, submitted, backend):
    model = replace(MODEL, serve=ServeConfig(max_model_len=SERVED_CONTEXT))
    settings = CuratedSettings(task_machine=TaskMachineSettings(backend=backend, cpus=2, memory_mb=4096))
    evaluator = DatasetEvaluator(("s3://corpora/agent",), model, settings, agent=AGENT)
    daytona = backend is TaskMachineBackend.DAYTONA
    # The launcher grants the eval-org key only to Daytona evaluations.
    env_vars = {DAYTONA_API_KEY: "eval-org-key"} if daytona else {}

    evaluator(SESSION, "s3://results/agent", env_vars, backend_state=lambda: InferenceBackendState.READY)

    _, args, _ = submitted["entrypoint"].resolve()
    task_machine = args[2].task_machine
    assert (task_machine.backend, task_machine.cpus, task_machine.memory_mb) == (backend, 2, 4096)
    environment = submitted["environment"]
    assert (environment.secrets.get(DAYTONA_API_KEY) == "eval-org-key") is daytona
    assert (tuple(submitted["egress_allow"]) == DAYTONA_EGRESS) is daytona
    assert ("daytona" in environment.setup_scripts[0]) is daytona


@pytest.mark.parametrize("image_build", list(TaskImageBuild))
def test_daytona_task_machines_get_images_as_the_settings_select(created, image_build):
    # The Daytona factory needs the optional Daytona SDK (marin-core's daytona extra).
    daytona_machine = pytest.importorskip("shellbox.backends.daytona.machine")
    settings = CuratedSettings(task_machine=TaskMachineSettings(image_build=image_build))

    config = runtime_config(MODEL, settings, SESSION)
    factories = machine_factories(config, DaytonaConnection(api_key="eval-org-key"))

    assert factories["daytona"].image_build is daytona_machine.DaytonaImageBuild(image_build.value)


def test_agent_worker_receives_task_secrets_as_secrets_its_run_may_resolve(created, submitted):
    model = replace(MODEL, serve=ServeConfig(max_model_len=SERVED_CONTEXT))
    settings = CuratedSettings(task_machine=TaskMachineSettings(backend=TaskMachineBackend.LOCAL))
    evaluator = DatasetEvaluator(
        ("s3://corpora/agent",), model, settings, agent=AGENT, task_secrets=("SEC_EDGAR_API_KEY",)
    )

    evaluator(
        SESSION,
        "s3://results/agent",
        {"SEC_EDGAR_API_KEY": "edgar-key"},
        backend_state=lambda: InferenceBackendState.READY,
    )

    _, args, _ = submitted["entrypoint"].resolve()
    assert args[2].task_secrets == ("SEC_EDGAR_API_KEY",)
    environment = submitted["environment"]
    assert environment.secrets["SEC_EDGAR_API_KEY"] == "edgar-key"


def test_agent_worker_rejects_a_server_window_other_than_the_planned_one(created, submitted):
    settings = CuratedSettings(task_machine=TaskMachineSettings())
    evaluator = DatasetEvaluator(("s3://corpora/agent",), MODEL, settings, agent=AGENT)

    with pytest.raises(ValueError, match="planned agent window"):
        evaluator(SESSION, "s3://results/agent", {}, backend_state=lambda: InferenceBackendState.READY)
    assert submitted == {}
