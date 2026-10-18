# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Harbor agents that tag every model request with a per-trial id.

Loaded by Harbor through ``import_path`` inside its isolated environment, so this module imports only
Harbor. The GLM proxy groups its request log by the ``x-ot-trial-id`` header (needed to rebuild
conversations and to keep relay state per trial). Each agent writes the id to
``trial_correlation_id.txt`` in its logs directory, which maps the Harbor trial to the proxy log.

With ``agent_domain_allow_list`` set, the Daytona sandbox's outbound network is limited to those
domains for the agent's run only. Setup and verification keep the task's own policy, because both
download uv from github.com; the agent run cannot reach github.com (e.g. the public
``github.com/swesmith`` mirrors that hold every task's clean code). The policy applied and the
sandbox's reported state are written to ``agent_network.json`` in the logs directory.

mini-swe-agent is installed from ``mini-swe-agent-2.1.0-constraints.txt`` (every dependency pinned) on
Python 3.12, so its tool env, which the agent can read and which holds clean copies of some
libraries, is the same in every trial. The pool excludes repos found in it.
"""

import hashlib
import json
import shlex
import time
import uuid
from pathlib import Path
from typing import Any

from harbor.agents.installed import mini_swe_agent
from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.agents.terminus_2.terminus_2 import Terminus2

TRIAL_HEADER = "x-ot-trial-id"
ID_FILE = "trial_correlation_id.txt"
NETWORK_FILE = "agent_network.json"
READY_TIMEOUT_SEC = 60
MINI_SWE_CONSTRAINTS = Path(__file__).with_name("mini-swe-agent-2.1.0-constraints.txt")
MINI_SWE_CONSTRAINTS_SHA256 = hashlib.sha256(MINI_SWE_CONSTRAINTS.read_bytes()).hexdigest()
MINI_SWE_PYTHON = "3.12"
ENV_API_BASE_MODEL = Path(__file__).with_name("env_api_base_model.py")


def _record(logs_dir: Any, trial_id: str) -> None:
    path = Path(str(logs_dir))
    path.mkdir(parents=True, exist_ok=True)
    (path / ID_FILE).write_text(trial_id + "\n")


def _sandbox_network(sandbox: Any) -> dict[str, Any]:
    return {
        "network_block_all": sandbox.network_block_all,
        "network_allow_list": sandbox.network_allow_list,
        "domain_allow_list": sandbox.domain_allow_list,
    }


class AgentPhaseNetwork:
    """Limits the sandbox to ``agent_domain_allow_list`` while ``run`` executes, then lifts the limit."""

    def __init__(self, *args: Any, agent_domain_allow_list: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.agent_domain_allow_list = agent_domain_allow_list

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        if not self.agent_domain_allow_list:
            return await super().run(instruction=instruction, environment=environment, context=context)
        sandbox = environment._sandbox
        record: dict[str, Any] = {"agent_domain_allow_list": self.agent_domain_allow_list}
        started = time.time()
        await sandbox.update_network_settings(domain_allow_list=self.agent_domain_allow_list)
        record["agent_phase"] = {"at": started, **_sandbox_network(sandbox)}
        try:
            # Wait (up to READY_TIMEOUT_SEC) until each allowed domain resolves, and record whether it did.
            for domain in self.agent_domain_allow_list.split(","):
                wait = (
                    f"for i in $(seq {READY_TIMEOUT_SEC}); do getent hosts {domain} >/dev/null && exit 0; sleep 1; done;"
                    " exit 1"
                )
                result = await environment.exec(command=wait, timeout_sec=READY_TIMEOUT_SEC + 30)
                record["agent_phase"][f"resolves:{domain}"] = result.return_code == 0
            record["agent_phase"]["ready_after_s"] = round(time.time() - started, 1)
            return await self.run_limited(instruction, environment, context)
        finally:
            await sandbox.update_network_settings(network_block_all=False)
            record["restored"] = {"at": time.time(), **_sandbox_network(sandbox)}
            (Path(str(self.logs_dir)) / NETWORK_FILE).write_text(json.dumps(record, indent=2) + "\n")

    async def run_limited(self, instruction: str, environment: Any, context: Any) -> None:
        await super().run(instruction=instruction, environment=environment, context=context)


class PinnedMiniSweInstall(MiniSweAgent):
    """Harbor's mini-swe-agent install, with the pinned constraints file and Python version."""

    async def install(self, environment: Any) -> None:
        await self.exec_as_root(
            environment,
            command="apt-get update && apt-get install -y curl build-essential git",
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        constraints = f"{mini_swe_agent._MODEL_DIRECTORY}/{MINI_SWE_CONSTRAINTS.name}"
        await self.exec_as_root(environment, command=f"mkdir -p {mini_swe_agent._MODEL_DIRECTORY}")
        await environment.upload_file(MINI_SWE_CONSTRAINTS, constraints)
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh && "
                'if ! grep -q \'export PATH="$HOME/.local/bin:$PATH"\' "$HOME/.bashrc" 2>/dev/null; then'
                '  echo \'export PATH="$HOME/.local/bin:$PATH"\' >> "$HOME/.bashrc";'
                " fi && "
                'source "$HOME/.local/bin/env" && '
                f"uv tool install --python {MINI_SWE_PYTHON} -c {constraints} mini-swe-agent=={self._version} && "
                "mini-swe-agent --help"
            ),
        )
        if self._model_info is not None:
            for source in (
                Path(mini_swe_agent.__file__).with_name("mini_swe_model.py"),
                Path(mini_swe_agent.__file__).parents[2] / "llms" / "vllm_tokenization.py",
                ENV_API_BASE_MODEL,
            ):
                await environment.upload_file(source, f"{mini_swe_agent._MODEL_DIRECTORY}/{source.name}")


class TrialTerminus2(Terminus2):
    """Terminus-2 whose chat requests carry the trial id header."""

    def __init__(self, *args: Any, llm_call_kwargs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        trial_id = uuid.uuid4().hex
        call_kwargs = dict(llm_call_kwargs or {})
        call_kwargs["extra_headers"] = {**(call_kwargs.get("extra_headers") or {}), TRIAL_HEADER: trial_id}
        super().__init__(*args, llm_call_kwargs=call_kwargs, **kwargs)
        self.trial_correlation_id = trial_id
        _record(self.logs_dir, trial_id)


class TrialMiniSweAgent(AgentPhaseNetwork, PinnedMiniSweInstall):
    """Mini-SWE whose requests carry the trial id header (the adapter's rollout correlation id)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.rollout_correlation_id = uuid.uuid4().hex
        _record(self.logs_dir, self.rollout_correlation_id)

    async def exec_as_agent(self, environment: Any, command: str, *args: Any, **kwargs: Any) -> Any:
        # The run command: drop the API base flag (the capability URL; see env_api_base_model.py) and use the
        # model class that reads it from the environment. The sandbox env still holds it, so outputs are scrubbed.
        if "mini-swe-agent --yolo" in command and self._api_base is not None:
            replacements = {
                f"-c {shlex.quote(f'model.api_base={self._api_base}')} ": "",
                "-c model.model_class=mini_swe_model.ContextLimitedModel ": (
                    f"-c model.model_class={ENV_API_BASE_MODEL.stem}.EnvApiBaseModel "
                ),
            }
            for old, new in replacements.items():
                if old not in command:
                    raise RuntimeError(f"mini-swe-agent run command changed; cannot find {old.split('=')[0]!r}")
                command = command.replace(old, new)
        return await super().exec_as_agent(environment, command, *args, **kwargs)
