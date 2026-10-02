# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay fixed provider responses through the production Harbor launch path."""

import json
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

from harbor.models.trial.result import TrialResult

from taskcompendium.harbor.runner import ChatLaunch, run_trial
from taskcompendium.lowering import ENVIRONMENT_CONFIG_FILE, read_environment_config


async def run_replay_trial(
    task_dir: Path,
    response: dict[str, Any],
    trials_dir: Path,
    trial_name: str,
) -> TrialResult:
    """Run the actual ChatAgent with a fixed response at the HTTP boundary."""
    environment_config = read_environment_config(task_dir / ENVIRONMENT_CONFIG_FILE)
    launch = ChatLaunch(model="fixture-model", api_base="https://example.invalid")
    body = BytesIO(json.dumps({"choices": [{"message": response}]}).encode())
    with patch("taskcompendium.harbor.adapter.urllib.request.urlopen", return_value=body):
        return await run_trial(task_dir, environment_config, launch, trials_dir, trial_name)
