# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from marin.vllm.docker_server import VllmDockerServerConfig, build_docker_run_command


def test_build_docker_run_command() -> None:
    config = VllmDockerServerConfig(
        image="vllm/vllm-tpu:nightly",
        model_name_or_path="/opt/gcsfuse_mount/models/foo",
        host="127.0.0.1",
        port=None,
        env={"TOKENIZERS_PARALLELISM": "false", "HF_TOKEN": "secret"},
        volumes=[("/opt/gcsfuse_mount", "/opt/gcsfuse_mount"), ("/tmp", "/tmp")],
        extra_vllm_args=["--max-model-len", "1024"],
        docker_run_args=["--privileged", "--shm-size=8g"],
    )

    cmd = build_docker_run_command(config, port=8123, container_name="marin-vllm-test-8123")

    assert cmd[:6] == ["docker", "run", "-d", "--net=host", "--name", "marin-vllm-test-8123"]
    assert "--privileged" in cmd
    assert "--shm-size=8g" in cmd
    assert ["-v", "/opt/gcsfuse_mount:/opt/gcsfuse_mount"] in [cmd[i : i + 2] for i in range(len(cmd) - 1)]
    assert ["-v", "/tmp:/tmp"] in [cmd[i : i + 2] for i in range(len(cmd) - 1)]

    assert cmd.count("-e") == 2
    assert "TOKENIZERS_PARALLELISM=false" in cmd
    assert "HF_TOKEN=secret" in cmd

    assert "vllm/vllm-tpu:nightly" in cmd
    assert ["vllm", "serve", "/opt/gcsfuse_mount/models/foo"] == cmd[cmd.index("vllm") : cmd.index("vllm") + 3]
    assert "--host" in cmd
    assert "--port" in cmd
    assert "--trust-remote-code" in cmd
    assert ["--max-model-len", "1024"] == cmd[-2:]
