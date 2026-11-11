# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the Pi and verifier file lifecycle with real guest UID 10001, without inference."""

import json
import shlex
import zipfile

import pytest

pytest.importorskip("harbor", reason="Requires the SkyRL harbor-test runtime environment")
from harbor.environments import daytona
from harbor.host_capabilities import DaytonaConnection, HostCapabilities
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.trial.trial import Trial
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.image import DockerfileSource
from shellbox.machine import Command, MachineSpec
from skyrl_train.dataset.harbor_import import harbor_trial_task, restore_harbor_trial_task
from taskcompendium.models import Source

from experiments.post_training.long_loop.eval_runtime.pi_image import NODE_IMAGE
from experiments.post_training.long_loop.pi_rl_data import AGENT_HOME, NONROOT_LAYER, pi_nonroot_task


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
        f'[agent.env]\nHOME="{AGENT_HOME}"\n'
        '[verifier]\nuser="10001"\ntimeout_sec=30\n'
        '[environment]\nworkdir="/testbed"\n'
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


@pytest.mark.docker
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_pi_overlay_preserves_original_verifier_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(daytona, "DaytonaMachineFactory", lambda *args, **kwargs: DockerMachineFactory())
    original = tmp_path / "original"
    environment = original / "environment"
    environment.mkdir(parents=True)
    (original / "instruction.md").write_text("Write the answer in /testbed.")
    (original / "task.toml").write_text(
        '[agent]\nuser="root"\ntimeout_sec=30\n'
        '[verifier]\nuser="root"\ntimeout_sec=30\n'
        '[environment]\nworkdir="/testbed"\n'
        '[environment.env]\nHOME="/root"\nVIRTUAL_ENV="/task-specified-venv"\n'
        'PATH="/usr/local/bin:/usr/bin:/bin"\n'
    )
    probe = (
        "import json,os,sys; print(json.dumps({"
        "'PATH':os.environ['PATH'],'HOME':os.environ['HOME'],"
        "'VIRTUAL_ENV':os.environ.get('VIRTUAL_ENV'),"
        "'python':sys.executable,'prefix':sys.prefix}))"
    )
    wheel_name = "longloop_dependency_fixture-0.0.0-py3-none-any.whl"
    wheel_files = {
        "longloop_dependency_fixture.py": "VALUE = 'installed-by-task'\ndef main():\n    print(VALUE)\n",
        "longloop_dependency_fixture-0.0.0.dist-info/METADATA": (
            "Metadata-Version: 2.1\nName: longloop-dependency-fixture\nVersion: 0.0.0\n"
        ),
        "longloop_dependency_fixture-0.0.0.dist-info/WHEEL": (
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
        "longloop_dependency_fixture-0.0.0.dist-info/entry_points.txt": (
            "[console_scripts]\nlongloop-dependency-fixture = longloop_dependency_fixture:main\n"
        ),
    }
    with zipfile.ZipFile(environment / wheel_name, "w") as wheel:
        for name, content in wheel_files.items():
            wheel.writestr(name, content)
        wheel.writestr(
            "longloop_dependency_fixture-0.0.0.dist-info/RECORD",
            "".join(name + ",,\n" for name in wheel_files),
        )
    (environment / "Dockerfile").write_text(
        "FROM python:3.12-slim-bookworm\nWORKDIR /testbed\n"
        f"COPY {wheel_name} /testbed/{wheel_name}\n"
        "RUN touch /usr/local/bin/longloop-protected-control\n"
    )
    baseline_machine = await DockerMachineFactory().create(
        MachineSpec(
            source=DockerfileSource(environment, environment / "Dockerfile"),
            env={"HOME": "/root", "VIRTUAL_ENV": "/task-specified-venv", "PATH": "/usr/local/bin:/usr/bin:/bin"},
            workdir="/testbed",
        )
    )
    try:
        baseline = await baseline_machine.run(Command(("python", "-c", probe)))
        assert baseline.exit_code == 0, baseline.stderr
        expected_verifier = json.loads(baseline.stdout)
    finally:
        await baseline_machine.close()
    message = {"role": "assistant", "content": [{"type": "text", "text": "Done."}], "stopReason": "stop"}
    events = "\n".join(
        json.dumps(e)
        for e in ({"type": "message_end", "message": message}, {"type": "agent_end", "messages": [message]})
    )
    # Stub only the model-facing CLI after the production runtime layer is built.
    install = f"python -m pip install --no-index --no-deps /testbed/{wheel_name}" " && longloop-dependency-fixture"
    bash_tool = (
        "const {createBashTool} = await import("
        "'/opt/long-loop-pi/node_modules/@earendil-works/pi-coding-agent/dist/index.js');"
        "const result = await createBashTool('/testbed').execute('offline-install',"
        f"{{command:{json.dumps(install)},timeout:30}},new AbortController().signal);"
        "if(result.isError)throw new Error(JSON.stringify(result));"
    )
    (environment / "pi").write_text(
        "#!/bin/bash\nset -eu\n"
        'if [ "${1:-}" = --version ]; then echo 0.87.0; exit 0; fi\n'
        'test "$(id -u)" = 10001\n'
        "if rm /usr/local/bin/longloop-protected-control 2>/dev/null; then exit 92; fi\n"
        f"python -c {shlex.quote(probe)} > /testbed/agent-env.json\n"
        f"node --input-type=module -e {shlex.quote(bash_tool)}\n"
        f"cat <<'PI_EVENTS'\n{events}\nPI_EVENTS\n"
    )
    tests = original / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text(
        "#!/bin/bash\nset -eu\n"
        'test "$(id -u)" = 0\n'
        "python -c \"import longloop_dependency_fixture as f; assert f.VALUE == 'installed-by-task'\"\n"
        'test "$(longloop-dependency-fixture)" = installed-by-task\n'
        f"python -c {shlex.quote(probe)} > /logs/verifier/env.json\n"
        "printf '{\"reward\":1.0}' > /logs/verifier/reward.json\n"
        "cp /testbed/agent-env.json /logs/verifier/agent-env.json\n"
    )
    task = harbor_trial_task(
        original,
        source=Source(
            dataset="long-loop-smartie-swesmith",
            revision="fixture",
            row="1",
            importer_revision="fixture",
        ),
    )
    prepared = pi_nonroot_task(task)
    restored = tmp_path / "derived"
    restore_harbor_trial_task(prepared, restored)
    with (restored / "environment/Dockerfile").open("a") as dockerfile:
        dockerfile.write(
            "\nCOPY pi /opt/long-loop-pi/node_modules/.bin/pi\nRUN chmod 0755 /opt/long-loop-pi/node_modules/.bin/pi\n"
        )
    trial = await Trial.create(
        TrialConfig(
            task=TaskConfig(path=restored),
            trial_name="pi-verifier-env",
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
        ),
        host_capabilities=HostCapabilities(
            daytona=DaytonaConnection(api_key="unused-transport-fixture", api_url="https://daytona.invalid", target="us")
        ),
    )
    result = await trial.run()
    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": 1.0}
    actual = json.loads((trial.paths.verifier_dir / "env.json").read_text())
    assert actual == expected_verifier
    agent_env = json.loads((trial.paths.verifier_dir / "agent-env.json").read_text())
    assert agent_env["HOME"] == AGENT_HOME
    assert agent_env["PATH"] == expected_verifier["PATH"]
    assert agent_env["python"] == expected_verifier["python"]
    assert agent_env["VIRTUAL_ENV"] == expected_verifier["VIRTUAL_ENV"]
