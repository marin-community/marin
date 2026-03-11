# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Docker CLI environment handling in the runtime."""

from iris.cluster.runtime.docker import _docker_env, _docker_run


def test_docker_env_sets_compatible_api_version_by_default():
    env = _docker_env()

    assert env["DOCKER_API_VERSION"] == "1.43"


def test_docker_run_uses_compatible_api_version(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr("iris.cluster.runtime.docker.subprocess.run", fake_run)

    _docker_run(["docker", "version"], capture_output=True, text=True, check=False)

    assert captured["cmd"] == ["docker", "version"]
    assert captured["env"]["DOCKER_API_VERSION"] == "1.43"
