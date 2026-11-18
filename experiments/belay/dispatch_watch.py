# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Belay dispatch: a Loom script watch that starts a Belay session as soon as there is work.

Loom runs this program every minute with no agent. Each round it updates its own checkout of the recipe branch, runs
``experiments.belay sync --light`` there (Iris only: each run the Belay page shows live that has ended is reported, which
queues its ``run_finished`` event), and, when events wait, no session is running and dispatch is on, launches a session
with the ``belay`` agent watch's own profile, repository and prompt through ``runs.create``. The idempotency key is the
queued event ids, so a repeat round launches nothing new. The ``belay`` agent watch stays scheduled as the full sync
(W&B progress, storage, registry, #9514 intake) and as a backstop.

It runs inside the Loom server container under the ``launch`` capability, with ``weaver_loom`` on its path and the
server's own Iris and kube credentials in its home directory.
"""

import json
import subprocess
from pathlib import Path

from weaver_loom import TERMINAL_STATUSES, Round

CHECKOUT = Path("/home/app/marin_belay_dispatch")
BRANCH = "marin_belay_v1_recipe"
AGENT_WATCH = "belay"


def sh(*args: str) -> str:
    return subprocess.run(args, cwd=CHECKOUT, check=True, capture_output=True, text=True).stdout


def queue_status(reap: bool) -> dict:
    """Update the checkout and run the light sync (``reap``: also clear sessions that died); its last stdout line is
    the queue as JSON."""
    sh("git", "fetch", "--quiet", "origin", BRANCH)
    sh("git", "reset", "--quiet", "--hard", f"origin/{BRANCH}")
    command = ["uv", "run", "--quiet", "python", "-m", "experiments.belay", "sync", "--light"]
    return json.loads(sh(*command, *(["--reap"] if reap else [])).splitlines()[-1])


def belay_session_alive(rnd: Round, profile: str) -> bool:
    """Whether Loom has a live session on the Belay profile. When it has none, a session the Belay page still shows
    running has died (its agent was lost or it ended without reporting) and holds its events until reaped."""
    sessions = rnd.client._request("POST", "/sessions/list", {}) or []
    return any(s.get("profile") == profile and s.get("status") not in TERMINAL_STATUSES for s in sessions)


def session_spec(rnd: Round) -> tuple[str, dict]:
    """The ``belay`` agent watch's profile and the session it launches (its repository and prompt)."""
    agent = rnd.client._request("POST", "/watches/get", {"key": AGENT_WATCH})["agent"]
    return agent["profile"], {
        "title": "Marin Belay pass (dispatched)",
        "goal": agent["prompt"],
        "repo": agent["repo"],
        "profile": agent["profile"],
    }


def main(rnd: Round) -> None:
    profile, session = session_spec(rnd)
    status = queue_status(reap=not rnd.dry_run and not belay_session_alive(rnd, profile))
    ended = f"{len(status['ended'])} ended run(s) reported; " if status["ended"] else ""
    if status["reaped"]:
        ended += f"reaped dead session(s) {', '.join(status['reaped'])}; "
    events = status["queued_event_ids"]
    if not events or status["session_active"] or not status["dispatch_enabled"]:
        why = (
            "no queued events" if not events else "a session is running" if status["session_active"] else "dispatch off"
        )
        rnd.finish(f"{ended}{why}", outcome="noop")
        return
    key = f"belay-events-{events[0]}-{events[-1]}-{len(events)}"
    if rnd.dry_run:
        rnd.would("launch", idempotency_key=key, events=len(events))
        rnd.finish(f"{ended}would launch a Belay session for {len(events)} event(s)")
        return
    run = rnd.client.run(profile, key, session, source="ops")
    rnd.did("launch", idempotency_key=key, run=(run or {}).get("id"), events=len(events))
    rnd.finish(f"{ended}launched a Belay session for {len(events)} event(s)")


Round.main(main)
