# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Harbor agents that tag every model request with a per-trial id for ``trace_proxy``.

Harbor loads this module through ``import_path`` (e.g. ``trial_agents:TrialMiniSweAgent``) inside its
own environment, so it imports only Harbor. Each agent sends ``TRIAL_HEADER`` on every chat request
and writes the id to ``ID_FILE`` in its logs directory (``<trial>/attempts/NNN/agent/``).
``rebuild_traces.py`` joins the proxy log to Harbor's ``result.json`` through that file.
"""

import uuid
from pathlib import Path
from typing import Any

from harbor.agents.installed.mini_swe_agent import MiniSweAgent  # pyrefly: ignore[missing-import]
from harbor.agents.terminus_2.terminus_2 import Terminus2  # pyrefly: ignore[missing-import]

TRIAL_HEADER = "x-ot-trial-id"
ID_FILE = "trial_correlation_id.txt"


def _record(logs_dir: Any, trial_id: str) -> None:
    path = Path(str(logs_dir))
    path.mkdir(parents=True, exist_ok=True)
    (path / ID_FILE).write_text(trial_id + "\n")


class TrialTerminus2(Terminus2):
    def __init__(self, *args: Any, llm_call_kwargs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        trial_id = uuid.uuid4().hex
        call_kwargs = dict(llm_call_kwargs or {})
        call_kwargs["extra_headers"] = {**(call_kwargs.get("extra_headers") or {}), TRIAL_HEADER: trial_id}
        super().__init__(*args, llm_call_kwargs=call_kwargs, **kwargs)
        _record(self.logs_dir, trial_id)


class TrialMiniSweAgent(MiniSweAgent):
    """Harbor's mini-swe-agent sends ``rollout_correlation_id`` as ``TRIAL_HEADER`` when it is set."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.rollout_correlation_id = uuid.uuid4().hex
        _record(self.logs_dir, self.rollout_correlation_id)
