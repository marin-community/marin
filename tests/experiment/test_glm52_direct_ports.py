# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import experiments.rollout_data.glm52_vllm as glm52_vllm
from experiments.rollout_data.glm52_vllm import Glm52LaunchConfig, ServerConfig


def test_serve_glm52_passes_explicit_ports_to_head(monkeypatch) -> None:
    class Info:
        advertise_host = "127.0.0.1"
        task_index = 0

    class Context:
        pass

    calls = []
    monkeypatch.setattr(glm52_vllm, "get_job_info", Info)
    monkeypatch.setattr(glm52_vllm, "iris_ctx", Context)
    monkeypatch.setattr(glm52_vllm, "_vllm_launch_context", lambda: (["vllm"], {}))
    monkeypatch.setattr(glm52_vllm, "_cuda_home", lambda *args: "/cuda")
    monkeypatch.setattr(glm52_vllm, "_network_interface", lambda *args: "eth0")
    monkeypatch.setattr(glm52_vllm, "_reserve_port", lambda host, port: port)
    monkeypatch.setattr(glm52_vllm, "_serve_ray_head", lambda *args: calls.append(args[-2:]))
    launch = Glm52LaunchConfig("http", "ray", ServerConfig(1_024, 1), tensor_parallel_size=8)

    glm52_vllm.serve_glm52(launch, 6_379, 8_000)

    assert calls == [(6_379, 8_000)]
