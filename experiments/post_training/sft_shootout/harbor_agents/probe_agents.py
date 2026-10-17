# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A null agent (changes nothing) that records which hosts the sandbox reaches before, during and after
the agent phase's network limit (``AgentPhaseNetwork``). Used to check that the agent cannot reach
github.com while the verifier, which downloads uv from github.com, still completes; on an unmodified
task the verifier should score 0. Results go to ``network_probe.json`` in the logs directory.
"""

import json
import shlex
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.agents.installed.mini_swe_agent import MiniSweAgent

from trial_agents import AgentPhaseNetwork

TARGETS = [
    "https://github.com/swesmith",
    "https://github.com/astral-sh/uv/releases/download/0.7.13/uv-x86_64-unknown-linux-gnu.tar.gz",
    "https://api.github.com/orgs/swesmith/repos",
    "https://raw.githubusercontent.com/octocat/Hello-World/master/README",
    "https://codeload.github.com/octocat/Hello-World/zip/refs/heads/master",
    "https://pypi.org/simple/pip/",
    "https://files.pythonhosted.org/",
    "https://astral.sh/uv/0.7.13/install.sh",
    "http://deb.debian.org/debian/",
    "http://archive.ubuntu.com/ubuntu/",
    "https://iris.oa.dev/",
    "https://www.google.com/",
    # By IP address, without DNS: GitHub (140.82.112.0/20) and Cloudflare's resolver.
    "https://140.82.112.3/",
    "https://140.82.114.4/",
    "https://1.1.1.1/",
]
# Prints one JSON object: target -> HTTP status, or the error class when the request fails.
PROBE = r"""
import json, ssl, sys, urllib.request, urllib.error
insecure = ssl._create_unverified_context()
out = {}
for url in sys.argv[1:]:
    try:
        out[url] = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=10, context=insecure).status
    except urllib.error.HTTPError as e:
        out[url] = e.code
    except Exception as e:
        out[url] = type(e).__name__ + ": " + str(e)[:120]
print(json.dumps(out))
"""


class NullAgent(BaseAgent):
    """Does nothing in the agent phase."""

    @staticmethod
    def name() -> str:
        return "null"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment: Any) -> None:
        return

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        return


class NetworkProbe(AgentPhaseNetwork, NullAgent):
    async def probe(self, environment: Any, label: str) -> None:
        python = "$(command -v python3 || echo /opt/miniconda3/bin/python3)"
        command = f"{python} -c {shlex.quote(PROBE)} {' '.join(shlex.quote(t) for t in TARGETS)}"
        result = await environment.exec(command=command, timeout_sec=300)
        path = Path(str(self.logs_dir)) / "network_probe.json"
        record = json.loads(path.read_text()) if path.exists() else {}
        try:
            record[label] = json.loads((result.stdout or "").strip().splitlines()[-1])
        except (ValueError, IndexError):
            record[label] = {"return_code": result.return_code, "stdout": result.stdout, "stderr": result.stderr}
        path.write_text(json.dumps(record, indent=2) + "\n")

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        await self.probe(environment, "before")
        await super().run(instruction, environment, context)
        await self.probe(environment, "after")

    async def run_limited(self, instruction: str, environment: Any, context: Any) -> None:
        await self.probe(environment, "agent_phase")
        await super().run_limited(instruction, environment, context)


# Lists what mini-swe-agent's own install puts in the sandbox: its tool env's distributions and their
# top-level import names, plus every other site-packages directory on the sandbox's filesystem.
TOOL_ENV_LISTING = r"""
. "$HOME/.local/bin/env"
tool=$(uv tool dir)/mini-swe-agent
echo "== python"; "$tool/bin/python" --version
echo "== distributions"; uv pip list --python "$tool/bin/python" --format json
echo "== top_level"
for d in "$tool"/lib/python3*/site-packages/*.dist-info; do
  name=$(basename "$d" .dist-info)
  if [ -f "$d/top_level.txt" ]; then echo "$name: $(tr '\n' ' ' < "$d/top_level.txt")";
  else echo "$name: (no top_level.txt) $(sed -n 's|^\([^/,]*\)/.*|\1|p' "$d/RECORD" | sort -u | grep -v dist-info | tr '\n' ' ')"; fi
done
echo "== site_packages entries"; ls "$tool"/lib/python3*/site-packages
echo "== other site-packages dirs"; find / -xdev -type d \( -name site-packages -o -name dist-packages \) 2>/dev/null | grep -v "^$tool"
"""


class ToolEnvProbe(MiniSweAgent):
    """Runs mini-swe-agent's real setup, then lists its tool env instead of calling a model."""

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        result = await environment.exec(command=TOOL_ENV_LISTING, timeout_sec=300)
        (Path(str(self.logs_dir)) / "tool_env.txt").write_text((result.stdout or "") + (result.stderr or ""))
