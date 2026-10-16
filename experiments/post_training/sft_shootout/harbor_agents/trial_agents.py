# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Harbor agents that tag every model request with a per-trial id.

Loaded by Harbor through ``import_path`` inside its isolated environment, so this module imports only
Harbor. The GLM proxy groups its request log by the ``x-ot-trial-id`` header (needed to rebuild
conversations and to keep relay state per trial). Each agent writes the id to
``trial_correlation_id.txt`` in its logs directory, which maps the Harbor trial to the proxy log.
"""

import uuid
from pathlib import Path
from typing import Any

from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.agents.terminus_2.terminus_2 import Terminus2

TRIAL_HEADER = "x-ot-trial-id"
ID_FILE = "trial_correlation_id.txt"


def _record(logs_dir: Any, trial_id: str) -> None:
    path = Path(str(logs_dir))
    path.mkdir(parents=True, exist_ok=True)
    (path / ID_FILE).write_text(trial_id + "\n")


class TrialTerminus2(Terminus2):
    """Terminus-2 whose chat requests carry the trial id header."""

    def __init__(self, *args: Any, llm_call_kwargs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        trial_id = uuid.uuid4().hex
        call_kwargs = dict(llm_call_kwargs or {})
        call_kwargs["extra_headers"] = {**(call_kwargs.get("extra_headers") or {}), TRIAL_HEADER: trial_id}
        super().__init__(*args, llm_call_kwargs=call_kwargs, **kwargs)
        self.trial_correlation_id = trial_id
        _record(self.logs_dir, trial_id)


class TrialMiniSweAgent(MiniSweAgent):
    """Mini-SWE whose requests carry the trial id header (the adapter's rollout correlation id)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.rollout_correlation_id = uuid.uuid4().hex
        _record(self.logs_dir, self.rollout_correlation_id)
