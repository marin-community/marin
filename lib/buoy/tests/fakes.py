# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-memory wandb fakes so the mirror layer runs without network or auth."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FakeSummary:
    _json_dict: dict


@dataclass
class FakeArtifact:
    type: str
    name: str
    local_dir: str
    size: int = 4096
    download_calls: int = 0

    def download(self, root: str | None = None) -> str:
        self.download_calls += 1
        return self.local_dir


@dataclass
class FakeRun:
    name: str = "fake-run"
    state: str = "finished"
    url: str = "https://wandb.ai/e/p/r"
    summary_dict: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)
    artifacts: list[FakeArtifact] = field(default_factory=list)

    @property
    def summary(self) -> FakeSummary:
        return FakeSummary(self.summary_dict)

    def scan_history(self, min_step=None):
        return iter(r for r in self.rows if min_step is None or r.get("_step", 0) >= min_step)

    def logged_artifacts(self):
        return self.artifacts


class FakeApi:
    """Records how many times a run was fetched (idempotency assertions)."""

    def __init__(self, run: FakeRun) -> None:
        self._run = run
        self.run_calls = 0

    def run(self, path: str) -> FakeRun:
        self.run_calls += 1
        return self._run

    def runs(self, path: str, per_page: int = 50):
        return []
