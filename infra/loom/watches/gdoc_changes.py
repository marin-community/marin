# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""gdoc-changes: send a waiting gdoc session "The doc changed" when a person changes its doc.

A Loom script watch: Loom runs it with `weaver_loom` on PYTHONPATH and the server's gcloud login when a session goes
idle, every `SLOW_CHECK` seconds, and sooner when a round asks to look again. Each session with a gdoc profile names its
doc in its goal; see `infra.loom.gdoc.doc_changes` for how often a doc is read and when a change is sent. Register it on
a local Loom, from the root of the marin checkout, with:

    loom watch add gdoc-changes --program "$PWD/infra/loom/watches/gdoc_changes.py" --capabilities observe,nudge
    loom watch run gdoc-changes --dry-run
    loom watch enable gdoc-changes

Params: `profiles`, the profile names to watch (default `["gdoc"]`).
"""

import math
import sys
import time
from pathlib import Path

# The engine puts only weaver_loom on PYTHONPATH; the gdoc tool is in the marin checkout that holds this file.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from weaver_loom import Round, WeaverError

from infra.loom.gdoc.doc_changes import SLOW_CHECK, Watched, doc_version, goal_doc_id, next_check, next_round
from infra.loom.gdoc.google_api import GoogleApi, access_token, gcloud_account

TRIGGERS = {"every": f"{SLOW_CHECK}s", "on": ["session.idle"]}
DEFAULT_PROFILES = ("gdoc",)
# Loom tags a session `idle` when its turn ends and clears the tag when the next turn starts.
IDLE_TAG = "idle"
# Only a running session has an agent to take a message; an orphaned one keeps its `idle` tag after its agent is gone.
RUNNING = "running"


def is_waiting(session: dict) -> bool:
    return any(tag.get("key") == IDLE_TAG for tag in (session.get("branch") or {}).get("tags") or [])


def main(rnd: Round) -> None:
    profiles = set(rnd.params.get("profiles") or DEFAULT_PROFILES)
    state = rnd.state or {}
    api: GoogleApi | None = None
    next_state: dict[str, dict] = {}
    checks: list[float] = []
    for session in rnd.sessions():
        if session.get("profile") not in profiles or session.get("status") != RUNNING:
            continue
        sid = session["id"]
        if sid in state:
            watched = Watched.from_json(state[sid])
        else:
            doc_id = goal_doc_id(rnd.client.session(sid)["branch"]["goal"])
            if doc_id is None:
                rnd.would("skip", session=sid, note="the goal links no Google Doc")
                continue
            watched = Watched(doc_id)
        version = None
        now = time.time()
        if is_waiting(session):
            api = api or GoogleApi(access_token(gcloud_account()))
            try:
                version = doc_version(api, watched.doc_id)
            except RuntimeError as error:
                # One unreadable doc, such as one not shared with this account, must not stop the others.
                rnd.would("skip", session=sid, doc=watched.doc_id, note=str(error)[:500])
                next_state[sid] = watched.to_json()
                continue
        watched, message = next_round(watched, version, now)
        next_state[sid] = watched.to_json()
        check = next_check(watched, now)
        if check is not None:
            checks.append(check)
        if message is None:
            continue
        if not rnd.can("nudge"):
            rnd.would("nudge", session=sid, text=message, note="nudge capability not granted")
        elif rnd.dry_run:
            rnd.would("nudge", session=sid, text=message)
        else:
            try:
                rnd.client.nudge(sid, message, by=rnd.name)
            except WeaverError as error:
                # The session stopped after the listing. Keep the change pending, and keep watching the others.
                next_state[sid] = state[sid]
                rnd.would("skip", session=sid, note=str(error)[:500])
                continue
            rnd.did("nudge", session=sid, text=message)
    rnd.set_state(next_state)
    # The schedule looks every SLOW_CHECK seconds; a doc that needs a sooner look asks for an extra round.
    if checks and min(checks) < SLOW_CHECK:
        rnd.wake_in(math.ceil(min(checks)))
    rnd.finish(f"watched {len(next_state)} gdoc sessions, {len(rnd.actions)} actions")


if __name__ == "__main__":
    Round.main(main, TRIGGERS)
