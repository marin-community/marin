# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import Any

_TRAINING_EXPORTS = {
    "TrainDpoOnPodConfig": "training",
    "TrainLmOnPodConfig": "training",
    "run_levanter_train_dpo": "levanter",
    "run_levanter_train_lm": "levanter",
}


def __getattr__(name: str) -> Any:
    if name in _TRAINING_EXPORTS:
        training_module = importlib.import_module(f".{_TRAINING_EXPORTS[name]}", __name__)
        return getattr(training_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
