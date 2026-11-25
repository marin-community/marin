"""Files shared by the Iris task runtime and its local SkyRL driver."""

import os
from pathlib import Path

DRIVER_PROGRESS_PATH_ENV = "MARIN_SKYRL_DRIVER_PROGRESS_PATH"
DRIVER_TIMING_FILENAME = "trainer_timing.json"
ELAPSED_EXCLUDING_EVAL = "elapsed_excluding_eval_seconds"


def driver_progress_path() -> Path | None:
    """Resolve the launcher's status path once at driver startup."""
    path = os.environ.get(DRIVER_PROGRESS_PATH_ENV)
    return Path(path) if path else None
