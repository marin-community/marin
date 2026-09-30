"""Capture a complete ShellSim VFS immediately after pinned Harbor startup."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from taskcompendium.harbor.environments import ShellSimEnvironment

SNAPSHOT_LIMITS = {
    "max_entries": 100_000,
    "max_file_bytes": 64 * 1024 * 1024,
    "max_response_bytes": 8 * 1024 * 1024,
}
MUTATION_PATH = "/__capability_reset_marker"


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


class ShellSimResetEnvironment(ShellSimEnvironment):
    """Use upstream startup and record the raw bridge VFS before agent actions."""

    def __init__(self, *args, snapshot_path: str, calibrate: bool = False, **kwargs):
        self.snapshot_path = Path(snapshot_path)
        self.calibrate = calibrate
        self.session_nonce = uuid.uuid4().hex
        super().__init__(*args, **kwargs)

    async def start(self, force_build: bool) -> None:
        await super().start(force_build)
        session = self._session()
        _write(self.snapshot_path, {
            "session_nonce": self.session_nonce,
            "bridge_pid": session._process.pid,
            "snapshot": session._request("snapshot", path="/", limits=SNAPSHOT_LIMITS),
        })
        if self.calibrate:
            session.write_file(MUTATION_PATH, b"reset-mutation\n")
            _write(self.snapshot_path.with_name("mutated-snapshot.json"), {
                "session_nonce": self.session_nonce,
                "bridge_pid": session._process.pid,
                "snapshot": session._request("snapshot", path="/", limits=SNAPSHOT_LIMITS),
            })

    async def stop(self, delete: bool) -> None:
        process = self.session._process if self.session is not None else None
        await super().stop(delete)
        _write(self.snapshot_path.with_name("session-stop.json"), {
            "session_nonce": self.session_nonce,
            "process_exited": process is not None and process.poll() is not None,
        })
