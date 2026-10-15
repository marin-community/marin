# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public identifiers: content hashes for mixtures, exposures and schedules; salted opaque ids for runs."""

import csv
import hashlib
import os
import re
import secrets
import tempfile
from collections.abc import Mapping
from pathlib import Path

ROUND = 9
SALT_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


def _canon(values: Mapping[str, float]) -> str:
    # Adding 0.0 turns the -0.0 that rounding leaves from tiny negative noise into 0.0.
    return ";".join(f"{k}={round(float(v), ROUND) + 0.0:.{ROUND}f}" for k, v in sorted(values.items()))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def mixture_id(weights: Mapping[str, float]) -> str:
    return "mix_" + _sha(_canon(weights))[:16]


def exposure_id(weights: Mapping[str, float], pool_fractions: Mapping[str, float], partition_revision: str) -> str:
    return "exp_" + _sha(f"{partition_revision}|{_canon(weights)}|{_canon(pool_fractions)}")[:16]


def schedule_id(w0: Mapping[str, float], w1: Mapping[str, float], boundary_step: int, total_steps: int) -> str:
    return "sch_" + _sha(f"{_canon(w0)}|{_canon(w1)}|{boundary_step}/{total_steps}")[:16]


class ObservationRegistry:
    """Opaque per-run ids. The run-name map is private (PRIVATE dir), never written into the build tree."""

    def __init__(self, salt: str, map_path: Path):
        self.salt = salt
        self.map_path = map_path
        self._map: dict[str, str] = {}
        if map_path.exists():
            with map_path.open() as handle:
                self._map = {row["run_name"]: row["observation_id"] for row in csv.DictReader(handle)}

    def id_for(self, run_name: str) -> str:
        if run_name not in self._map:
            self._map[run_name] = "obs_" + _sha(f"{self.salt}|{run_name}")[:12]
        return self._map[run_name]

    def save(self) -> None:
        """Write the map owner-only, since it is the one file that undoes the pseudonymization.

        `tempfile.mkstemp` creates the new map exclusively with mode 0o600, and `os.replace` puts it in place of the old
        map, whose mode does not carry over.
        """
        self.map_path.parent.mkdir(parents=True, exist_ok=True)
        fd, staging = tempfile.mkstemp(dir=self.map_path.parent, prefix=f".{self.map_path.name}.")
        with os.fdopen(fd, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["run_name", "observation_id"])
            writer.writerows(sorted(self._map.items()))
        os.replace(staging, self.map_path)


def load_salt(path: Path) -> str:
    """The salt lives outside the repo; create it once, owner-only, with 32 random bytes.

    Raises ValueError unless the file holds exactly 64 hex characters, so a truncated salt never reaches `id_for`.
    """
    if not path.exists():
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(secrets.token_hex(32))
    salt = path.read_text().strip()
    if not SALT_PATTERN.fullmatch(salt):
        raise ValueError(f"malformed salt at {path}")
    return salt
