# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ShellSim verifier for the coding-expert Harbor task packages."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import shellsim
from coding_expert_verifier import SOLUTION_PATH, VerificationSummary, parse_cases
from harbor.agents.factory import AgentFactory
from harbor.environments.base import ExecResult
from harbor.models.agent.name import AgentName
from shellbox.agent import BashAgent
from shellbox.backends.shellsim.environment import ShellSimEnvironment

VERIFY_ENV = "MARIN_CODING_EXPERT_VERIFY"
CASES_PATH = "/tests/cases.json"
REWARD_PATH = "/logs/verifier/reward.txt"
VERIFIER_TIMEOUT = 30

# MarinSkyRL does not yet expose AgentConfig.import_path. This module is loaded
# through EnvironmentConfig.import_path before Harbor creates the trial agent.
AgentFactory._AGENT_MAP[AgentName.ORACLE] = BashAgent


def _verify(solution_path: Path, cases_path: Path) -> VerificationSummary:
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("coding_expert_verifier.py")),
                str(solution_path),
                str(cases_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=VERIFIER_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return VerificationSummary(0, f"verifier exceeded {VERIFIER_TIMEOUT} seconds")
    payload = json.loads(result.stdout)
    return VerificationSummary(reward=int(payload["reward"]), message=str(payload["message"]))


class CodingExpertShellSimEnvironment(ShellSimEnvironment):
    """Run private code cases outside the stateful agent simulation."""

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if not env or env.get(VERIFY_ENV) != "1":
            return await super().exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)
        if user not in (None, "root", 0):
            raise ValueError("The coding verifier supports only the root user")
        if self.machine is None:
            raise RuntimeError("ShellSim environment is not running")

        with tempfile.TemporaryDirectory(prefix="coding-expert-verifier-") as temporary:
            root = Path(temporary)
            solution_path = root / "solution.py"
            cases_path = root / "cases.json"
            try:
                await self.machine.download(SOLUTION_PATH, solution_path)
            except shellsim.SimulationError as error:
                if "No such file or directory" not in str(error):
                    raise
                summary = VerificationSummary(0, "solution.py is missing")
            else:
                await self.machine.download(CASES_PATH, cases_path)
                parse_cases(json.loads(cases_path.read_text()))
                summary = await asyncio.to_thread(_verify, solution_path, cases_path)

            reward_path = root / "reward.txt"
            reward_path.write_text(f"{summary.reward}\n")
            await self.machine.upload(reward_path, REWARD_PATH)

        return ExecResult(
            stdout=summary.message + "\n",
            stderr="",
            return_code=0,
            stdout_truncated=False,
            stderr_truncated=False,
        )
