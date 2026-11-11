# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the Pi and verifier file lifecycle with real guest UID 10001, without inference."""

import json

import pytest

pytest.importorskip("harbor", reason="Requires the SkyRL harbor-test runtime environment")
from harbor.environments import daytona
from harbor.host_capabilities import DaytonaConnection, HostCapabilities
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.trial.trial import Trial
from shellbox.backends.docker.machine import DockerMachineFactory

from experiments.post_training.long_loop.eval_runtime.pi_image import NODE_IMAGE
from experiments.post_training.long_loop.pi_rl_data import AGENT_HOME, AGENT_PATH, NONROOT_LAYER


@pytest.mark.docker
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_pi_setup_cleanup_and_verifier_as_nonroot(tmp_path, monkeypatch):
    # Replace only the provider transport. Harbor still drives the real Daytona
    # adapter, Pi setup/run/cleanup and verifier against an ordinary local container.
    monkeypatch.setattr(daytona, "DaytonaMachineFactory", lambda *args, **kwargs: DockerMachineFactory())
    task = tmp_path / "task"
    environment = task / "environment"
    environment.mkdir(parents=True)
    (task / "instruction.md").write_text("Write the answer in /testbed.")
    (task / "task.toml").write_text(
        '[agent]\nuser="10001"\ntimeout_sec=30\n'
        '[verifier]\nuser="10001"\ntimeout_sec=30\n'
        '[environment]\nworkdir="/testbed"\n'
        f'[environment.env]\nHOME="{AGENT_HOME}"\nPATH="{AGENT_PATH}"\n'
    )
    message = {"role": "assistant", "content": [{"type": "text", "text": "Done."}], "stopReason": "stop"}
    events = "\n".join(
        json.dumps(e)
        for e in ({"type": "message_end", "message": message}, {"type": "agent_end", "messages": [message]})
    )
    (environment / "pi").write_text(
        "#!/bin/bash\nset -eu\n"
        'if [ "${1:-}" = --version ]; then echo 0.87.0; exit 0; fi\n'
        'test "$(id -u)" = 10001\n'
        "inode=$(stat -c %i /testbed)\n"
        "if rm -rf /testbed 2>/dev/null; then exit 91; fi\n"
        'test "$(stat -c %i /testbed)" = "$inode"\n'
        "id -u > /testbed/agent-uid\n"
        "printf answer > /testbed/answer\n"
        f"cat <<'PI_EVENTS'\n{events}\nPI_EVENTS\n"
    )
    (environment / "Dockerfile").write_text(
        f"FROM {NODE_IMAGE} AS pi_node\n"
        "FROM python:3.12-slim-bookworm\n"
        "COPY --from=pi_node /usr/local/bin/node /opt/long-loop-node/node\n"
        "COPY pi /opt/long-loop-pi/node_modules/.bin/pi\n"
        "RUN chmod 0755 /opt/long-loop-pi/node_modules/.bin/pi\n" + NONROOT_LAYER + "\nWORKDIR /testbed\n"
    )
    setup = task / "setup_files"
    setup.mkdir()
    (setup / "readable.txt").write_text("admin fixture")
    tests = task / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text(
        "#!/bin/bash\nset -eu\n"
        'test "$(id -u)" = 10001\n'
        'test "$(cat /testbed/agent-uid)" = 10001\n'
        'test "$(cat /testbed/answer)" = answer\n'
        'test "$(cat /setup_files/readable.txt)" = "admin fixture"\n'
        "test ! -e /logs/agent/harbor-pi-endpoint.env\n"
        "test ! -e /logs/agent/harbor-pi-context-budget.js\n"
        f"test ! -e {AGENT_HOME}/.pi/agent/models.json\n"
        "id -u > /logs/verifier/verifier-uid\n"
        "printf '{\"reward\":1.0}' > /logs/verifier/reward.json\n"
    )
    config = TrialConfig(
        task=TaskConfig(path=task),
        trial_name="pi-nonroot",
        trials_dir=tmp_path / "trials",
        environment=EnvironmentConfig(type="daytona"),
        agent=AgentConfig(
            name="pi",
            model_name="hosted_vllm/sft-handoff",
            model_alias="hosted_vllm/sft-handoff",
            kwargs={
                "version": "0.87.0",
                "api_base": "https://model.example.test/v1",
                "api_key": "EMPTY",
                "thinking_format": "chat-template",
            },
        ),
    )
    trial = await Trial.create(
        config,
        host_capabilities=HostCapabilities(
            daytona=DaytonaConnection(api_key="unused-transport-fixture", api_url="https://daytona.invalid", target="us")
        ),
    )
    result = await trial.run()
    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": 1.0}
    assert (trial.paths.verifier_dir / "verifier-uid").read_text().strip() == "10001"
    assert trial.paths.result_path.exists()
