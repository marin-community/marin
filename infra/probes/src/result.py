# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The ProbeResult schema. Its own module so both the runner (infra_probes) and
the sinks can import it without a cycle, and without importing the `python -m`
entrypoint module (which would re-import it as a second `__main__` copy)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class ProbeResult:
    # name and started_at are supplied by the runner, which owns this metadata;
    # the probe fn only reports is_success. wall_time is filled in once the run
    # completes.
    is_success: bool
    name: str
    started_at: datetime  # UTC wall-clock time at probe start
    wall_time: float | None = None
