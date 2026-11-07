# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace

import requests
from iris.client.client import IrisContext
from iris.cluster.client.job_info import JobInfo, set_job_info
from iris.cluster.types import JobName
from marin.inference.config import ServedModelConfig, VllmEngineConfig
from marin.inference.types import OpenAIEndpoint, RunningModel

from experiments.post_training.standalone_draft import main


def test_serve_registers_an_http_endpoint_without_named_job_ports(monkeypatch):
    """Generic Iris job submissions have no allocated named HTTP port."""
    job = JobName.from_wire("/alice/standalone-draft/0")
    context = IrisContext(job_id=job.parent, ports={})
    responses = []

    @contextmanager
    def stored_export(_output):
        yield (
            ServedModelConfig(
                weights="target",
                api_model="target",
                max_model_len=65536,
                tensor_parallel_size=1,
                chat_template_content="{{ messages }}",
            ),
            VllmEngineConfig(),
        )

    @contextmanager
    def gpu_backend(*_arguments):
        yield SimpleNamespace(
            model=RunningModel(OpenAIEndpoint("http://127.0.0.1:1/v1", "target")),
            backend_name="vllm",
            tensor_parallel_size=1,
            chat_template_content="{{ messages }}",
            check_alive=lambda: None,
        )

    @contextmanager
    def registered(name, address, metadata, **_arguments):
        response = requests.get(address + "/info", timeout=5)
        response.raise_for_status()
        responses.append((name, response.json()))
        yield

    monkeypatch.setattr("experiments.post_training.standalone_draft.draft_serving", stored_export)
    monkeypatch.setattr("marin.inference.iris._prepared_local_inference", gpu_backend)
    monkeypatch.setattr(
        "marin.inference.iris.iris_ctx",
        lambda: SimpleNamespace(get_port=context.get_port, registry=SimpleNamespace(registered=registered)),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "standalone_draft",
            "serve",
            "--output",
            "stored-run",
            "--endpoint",
            "standalone-draft",
            "--timeout-hours",
            "0",
        ],
    )
    set_job_info(JobInfo(task_id=job))
    try:
        main()
    finally:
        set_job_info(None)
    assert len(responses) == 1
    name, info = responses[0]
    assert name == info["endpoint"] == "standalone-draft"
    assert info["model"] == "target"
    assert info["max_model_len"] == 65536
