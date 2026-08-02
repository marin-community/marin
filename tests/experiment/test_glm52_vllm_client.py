# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager

import experiments.rollout_data.glm52_vllm as glm52_vllm
from experiments.rollout_data.glm52_vllm import Glm52LaunchConfig, ServerConfig


def test_run_vllm_calls_registered_client_and_stops_server(monkeypatch) -> None:
    class Process:
        return_code: int | None = None
        terminated = False

        def poll(self) -> int | None:
            return self.return_code

        def terminate(self) -> None:
            self.terminated = True
            self.return_code = 0

        def wait(self, timeout: float | None = None) -> int:
            assert self.return_code is not None
            return self.return_code

        def kill(self) -> None:
            raise AssertionError("The process must stop after terminate")

    process = Process()
    monkeypatch.setattr(glm52_vllm.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(glm52_vllm, "_poll_until_ready", lambda *args, **kwargs: None)

    class Registry:
        @contextmanager
        def registered(self, name: str, url: str):
            yield

    class Context:
        registry = Registry()

    urls = []
    launch = Glm52LaunchConfig(
        "endpoint",
        "ray",
        ServerConfig(max_model_len=1_024, max_num_seqs=1),
        client=urls.append,
    )

    glm52_vllm._run_vllm(
        Context(),
        "127.0.0.1",
        8_000,
        "127.0.0.1:9_000",
        ["vllm"],
        {},
        "weights",
        launch,
    )

    assert urls == ["http://127.0.0.1:8000"]
    assert process.terminated
