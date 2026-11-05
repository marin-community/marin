# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""gdoc-changes: send a waiting gdoc session "The doc changed" when a person changes its doc.

A Loom script watch: Loom runs it every minute with `weaver_loom` on PYTHONPATH and the server's gcloud login. Each
session with a gdoc profile names its doc in its goal; see `infra.loom.gdoc.doc_changes` for when a change is sent.
Register it on a local Loom with:

    loom watch add gdoc-changes --program "$PWD/infra/loom/watches/gdoc_changes.py" --every 1m \
        --capabilities observe,nudge
    loom watch run gdoc-changes --dry-run
    loom watch enable gdoc-changes

Params: `profiles`, the profile names to watch (default `["gdoc"]`).
"""

import sys
from pathlib import Path

# The engine puts only weaver_loom on PYTHONPATH; the gdoc tool is in the marin checkout that holds this file.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from weaver_loom import Round

from infra.loom.gdoc.doc_changes import Watched, doc_version, goal_doc_id, next_round
from infra.loom.gdoc.google_api import GoogleApi, access_token, gcloud_account

TRIGGERS = {"every": "1m"}
DEFAULT_PROFILES = ("gdoc",)
# Loom tags a session `idle` when its turn ends and clears the tag when the next turn starts.
IDLE_TAG = "idle"


def is_waiting(session: dict) -> bool:
    return any(tag.get("key") == IDLE_TAG for tag in (session.get("branch") or {}).get("tags") or [])


def main(rnd: Round) -> None:
    profiles = set(rnd.params.get("profiles") or DEFAULT_PROFILES)
    state = rnd.state or {}
    api: GoogleApi | None = None
    next_state: dict[str, dict] = {}
    for session in rnd.sessions():
        if session.get("profile") not in profiles:
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
        if is_waiting(session):
            api = api or GoogleApi(access_token(gcloud_account()))
            try:
                version = doc_version(api, watched.doc_id)
            except RuntimeError as error:
                # One unreadable doc, such as one not shared with this account, must not stop the others.
                rnd.would("skip", session=sid, doc=watched.doc_id, note=str(error)[:500])
                next_state[sid] = watched.to_json()
                continue
        watched, message = next_round(watched, version)
        next_state[sid] = watched.to_json()
        if message is None:
            continue
        if not rnd.can("nudge"):
            rnd.would("nudge", session=sid, text=message, note="nudge capability not granted")
        elif rnd.dry_run:
            rnd.would("nudge", session=sid, text=message)
        else:
            rnd.client.nudge(sid, message, by=rnd.name)
            rnd.did("nudge", session=sid, text=message)
    rnd.set_state(next_state)
    rnd.finish(f"watched {len(next_state)} gdoc sessions, {len(rnd.actions)} actions")


if __name__ == "__main__":
    Round.main(main, TRIGGERS)
