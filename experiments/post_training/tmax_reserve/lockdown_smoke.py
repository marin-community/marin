# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check on one Daytona sandbox that the post-setup network lockdown takes effect while it runs.

The sandbox starts unrestricted, as task setup does. A background bash process inside it holds one keep-alive
HTTP connection to ``HELD_HOST`` and logs each round trip. The script then applies the pilot's agent policy through
``DaytonaMachine.update_network`` and checks:

- the held connection worked before the lockdown (at least ``MIN_ROUND_TRIPS``), and every round after it failed;
- new connections fail: the trial's own probe command to ``PROBE_HOST``, and one to ``HELD_HOST``;
- ``ALLOWED_URL`` (the allow-listed model host) still answers.

The sandbox is closed at the end. Its snapshot is deleted only if it did not exist before the run. Runs in the
vendored Harbor runtime with ``DAYTONA_API_KEY`` and ``DAYTONA_TARGET`` set; writes ``--output`` as JSON.
"""

import argparse
import asyncio
import json
import os
import shlex
from pathlib import Path

from daytona import AsyncDaytona, DaytonaConfig
from harbor.environments.daytona import LOCKDOWN_PROBE_TIMEOUT
from shellbox.backends.daytona.machine import (
    DaytonaMachineFactory,
    DaytonaNetworkMode,
    DaytonaNetworkPolicy,
)
from shellbox.backends.daytona.snapshot import snapshot_request_for_spec
from shellbox.image import DockerfileSource
from shellbox.machine import Command, MachineSpec, NetworkPolicy

from experiments.post_training.tmax_reserve import snapshots

# An HTTP host that keeps connections alive (pypi.org answers port 80 with "Connection: close").
HELD_HOST = "archive.ubuntu.com"
# The trial's lockdown probe target (harbor/pilot-200.yaml).
PROBE_HOST, PROBE_PORT = "pypi.org", 443
ALLOWED_URL = "https://iris.oa.dev/"
AGENT_POLICY = DaytonaNetworkPolicy(DaytonaNetworkMode.DOMAIN_ALLOW_LIST, "iris.oa.dev")
SETUP_POLICY = DaytonaNetworkPolicy(DaytonaNetworkMode.UNRESTRICTED)
HOLD_LOG = "/tmp/held-connection.log"
MIN_ROUND_TRIPS = 3
# Seconds to watch the held connection on each side of the lockdown.
WATCH_SECONDS = 20
# Keep-alive HEAD requests over one connection; each line is "<epoch> ok" or "<epoch> fail <status>". A read
# timeout (status above 128) means packets stopped; status 1 means the server closed the connection.
HOLDER = f"""trap '' PIPE
exec 3<>/dev/tcp/{HELD_HOST}/80 || {{ echo "$(date +%s) connect-fail" >> {HOLD_LOG}; exit 1; }}
while :; do
  printf 'HEAD / HTTP/1.1\\r\\nHost: {HELD_HOST}\\r\\nConnection: keep-alive\\r\\n\\r\\n' >&3 2>/dev/null
  if IFS= read -r -t 5 status <&3 && [ "${{status#HTTP/}}" != "$status" ]; then
    while IFS= read -r -t 2 line <&3 && [ "$line" != $'\\r' ]; do :; done
    echo "$(date +%s) ok" >> {HOLD_LOG}
  else
    rc=$?
    echo "$(date +%s) fail $rc" >> {HOLD_LOG}
  fi
  sleep 1
done"""


async def run(machine, script: str, timeout: float = 60) -> tuple[int | None, str]:
    result = await machine.run(Command(("bash", "-c", script), cwd="/", timeout=timeout, user="root"))
    return result.exit_code, result.stdout.decode(errors="replace") + result.stderr.decode(errors="replace")


def held_rounds(log: str, locked_at: float) -> dict:
    rows = [line.split() for line in log.splitlines() if line.strip()]
    before = [row[1:] for row in rows if float(row[0]) < locked_at]
    after = [row[1:] for row in rows if float(row[0]) >= locked_at + 2]
    return {
        "before_ok": before.count(["ok"]),
        "before_fail": len(before) - before.count(["ok"]),
        "after_ok": after.count(["ok"]),
        "after_fail": len(after) - after.count(["ok"]),
        "after_fail_statuses": sorted({int(row[1]) for row in after if row[0] == "fail"}),
    }


async def smoke(source: DockerfileSource) -> dict:
    config = DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"], target=os.environ["DAYTONA_TARGET"])
    factory = DaytonaMachineFactory(lambda: AsyncDaytona(config), network_policy=SETUP_POLICY)
    machine = await factory.create(smoke_spec(source))
    try:
        await run(machine, f"rm -f {HOLD_LOG}; setsid nohup bash -c {shlex.quote(HOLDER)} >/dev/null 2>&1 &")
        await asyncio.sleep(WATCH_SECONDS)
        # The holder stamps lines with the sandbox clock, so the lockdown time comes from it too.
        _, locked_at = await run(machine, "date +%s")
        await machine.update_network(AGENT_POLICY)
        await asyncio.sleep(WATCH_SECONDS)
        _, log = await run(machine, f"cat {HOLD_LOG}")
        probe, _ = await run(
            machine, f"timeout {LOCKDOWN_PROBE_TIMEOUT} bash -c 'exec 3<>/dev/tcp/{PROBE_HOST}/{PROBE_PORT}'"
        )
        held_host_probe, _ = await run(
            machine, f"timeout {LOCKDOWN_PROBE_TIMEOUT} bash -c 'exec 3<>/dev/tcp/{HELD_HOST}/80'"
        )
        allowed, allowed_output = await run(
            machine, "timeout 20 bash -c 'exec 3<>/dev/tcp/iris.oa.dev/443' && echo reachable"
        )
    finally:
        await machine.close()
    rounds = held_rounds(log, float(locked_at.strip()))
    return {
        "environment": str(source.context),
        "held_connection": rounds,
        "held_log": log.splitlines(),
        "new_connection_probe_exit": probe,
        "held_host_new_connection_exit": held_host_probe,
        "allowed_host_reachable": allowed == 0 and "reachable" in allowed_output,
        "pass": (
            rounds["before_ok"] >= MIN_ROUND_TRIPS
            and rounds["before_fail"] == 0
            and rounds["after_ok"] == 0
            and rounds["after_fail"] >= MIN_ROUND_TRIPS
            # Read timeouts only: the connection was cut, not closed by the server.
            and all(status > 128 for status in rounds["after_fail_statuses"])
            and probe != 0
            and held_host_probe != 0
            and allowed == 0
        ),
    }


def smoke_spec(source: DockerfileSource) -> MachineSpec:
    return MachineSpec(source=source, workdir="/", env={}, network=NetworkPolicy.ALLOW, cpus=1, memory_mb=2048)


def snapshot_name(source: DockerfileSource) -> str:
    """The name the factory gives this smoke's snapshot."""
    return snapshot_request_for_spec(smoke_spec(source)).name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--environment", type=Path, required=True, help="A template-A task's environment/ (the image A context)."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = DockerfileSource(args.environment, args.environment / "Dockerfile")
    name = snapshot_name(source)
    preexisting = bool(snapshots.existing([name]))
    try:
        result = asyncio.run(smoke(source))
    finally:
        deletion = [] if preexisting else snapshots.delete([name])
    result["snapshot"] = {"name": name, "preexisting": preexisting, "deletion": deletion}
    args.output.write_text(json.dumps(result, indent=1))
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "held_connection",
                    "new_connection_probe_exit",
                    "held_host_new_connection_exit",
                    "allowed_host_reachable",
                    "pass",
                    "snapshot",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
