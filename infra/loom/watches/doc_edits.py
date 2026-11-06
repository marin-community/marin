# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""doc-edits: send a waiting `doc` session "A person edited the doc" when a person saves an edit to its doc.

A Loom script watch: Loom runs it with `weaver_loom` on PYTHONPATH when a session goes idle and every 15 seconds.
See `infra.loom.doc_edits` for when a message is sent. An edit that a person saves while the agent works is sent when
the session next waits. Register it on a local Loom, from the root of the marin checkout, with:

    loom watch add doc-edits --program "$PWD/infra/loom/watches/doc_edits.py" --capabilities observe,nudge
    loom watch run doc-edits --dry-run
    loom watch enable doc-edits

Params: `profiles`, the profile names to watch (default `["doc"]`).
"""

import sys
from pathlib import Path

# The engine puts only weaver_loom on PYTHONPATH; the rules are in the marin checkout that holds this file.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from weaver_loom import Round, WeaverError

from infra.loom.doc_edits import DOC_ARTIFACT, edit_message

TRIGGERS = {"every": "15s", "on": ["session.idle"]}
DEFAULT_PROFILES = ("doc",)
# Loom tags a session `idle` when its turn ends and clears the tag when the next turn starts.
IDLE_TAG = "idle"
# Only a running session has an agent to take a message.
RUNNING = "running"
# The most characters of an error that a skip note keeps.
NOTE_CHARS = 500


def is_waiting(session: dict) -> bool:
    return any(tag.get("key") == IDLE_TAG for tag in (session.get("branch") or {}).get("tags") or [])


def doc_rev(rnd: Round, sid: str) -> int:
    """The latest revision of the session's doc, or 0 before the agent writes it."""
    # weaver_loom has no artifact helpers; `_request` calls the operation's REST route, /api/artifacts/list.
    listing = rnd.client._request("POST", "/artifacts/list", {"branch": sid}) or []
    return next((artifact["rev"] for artifact in listing if artifact["name"] == DOC_ARTIFACT), 0)


def main(rnd: Round) -> None:
    profiles = set(rnd.params.get("profiles") or DEFAULT_PROFILES)
    state = rnd.state or {}
    next_state: dict[str, int] = {}
    for session in rnd.sessions():
        if session.get("profile") not in profiles or session.get("status") != RUNNING:
            continue
        sid = session["id"]
        rev = doc_rev(rnd, sid)
        if sid not in state:
            # The first look: edits from before the watch saw the session are not sent.
            next_state[sid] = rev
            continue
        seen = state[sid]
        next_state[sid] = seen
        if rev <= seen or not is_waiting(session):
            continue
        versions = rnd.client._request("POST", "/artifacts/get", {"branch": sid, "name": DOC_ARTIFACT})["versions"]
        latest, message = edit_message(versions, seen)
        if message is None:
            next_state[sid] = latest
            continue
        if not rnd.can("nudge"):
            rnd.would("nudge", session=sid, text=message, note="nudge capability not granted")
            continue
        if rnd.dry_run:
            rnd.would("nudge", session=sid, text=message)
            continue
        try:
            rnd.client.nudge(sid, message, by=rnd.name)
        except WeaverError as error:
            # The session stopped after the listing. Keep the edit pending, and keep watching the others.
            rnd.would("skip", session=sid, note=str(error)[:NOTE_CHARS])
            continue
        next_state[sid] = latest
        rnd.did("nudge", session=sid, text=message)
    rnd.set_state(next_state)
    rnd.finish(f"watched {len(next_state)} doc sessions, {len(rnd.actions)} actions")


if __name__ == "__main__":
    Round.main(main, TRIGGERS)
