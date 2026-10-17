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
"""

import json
import time
import uuid
from pathlib import Path
from typing import Any

from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.agents.terminus_2.terminus_2 import Terminus2

TRIAL_HEADER = "x-ot-trial-id"
ID_FILE = "trial_correlation_id.txt"
NETWORK_FILE = "agent_network.json"
READY_TIMEOUT_SEC = 60


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
                wait = f"for i in $(seq {READY_TIMEOUT_SEC}); do getent hosts {domain} >/dev/null && exit 0; sleep 1; done; exit 1"
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


class TrialTerminus2(Terminus2):
    """Terminus-2 whose chat requests carry the trial id header."""

    def __init__(self, *args: Any, llm_call_kwargs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        trial_id = uuid.uuid4().hex
        call_kwargs = dict(llm_call_kwargs or {})
        call_kwargs["extra_headers"] = {**(call_kwargs.get("extra_headers") or {}), TRIAL_HEADER: trial_id}
        super().__init__(*args, llm_call_kwargs=call_kwargs, **kwargs)
        self.trial_correlation_id = trial_id
        _record(self.logs_dir, trial_id)


class TrialMiniSweAgent(AgentPhaseNetwork, MiniSweAgent):
    """Mini-SWE whose requests carry the trial id header (the adapter's rollout correlation id)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.rollout_correlation_id = uuid.uuid4().hex
        _record(self.logs_dir, self.rollout_correlation_id)
