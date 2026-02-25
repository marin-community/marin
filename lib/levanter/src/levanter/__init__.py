# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import importlib

__all__ = [
    "analysis",
    "callbacks",
    "checkpoint",
    "config",
    "data",
    "distributed",
    "eval",
    "eval_harness",
    "models",
    "optim",
    "tracker",
    "trainer",
    "visualization",
    "grug",
    "current_tracker",
    "initialize",
]

import levanter.analysis as analysis
import levanter.callbacks as callbacks
import levanter.checkpoint as checkpoint
import levanter.config as config
import levanter.data as data
import levanter.distributed as distributed
import levanter.models as models
import levanter.optim as optim
import levanter.tracker as tracker
import levanter.trainer as trainer
import levanter.visualization as visualization
import levanter.grug as grug
from levanter.tracker import current_tracker
from levanter.trainer import initialize

# eval and eval_harness are loaded lazily because they transitively import
# transformers (via levanter.data.text), which unconditionally imports torch.
# This fails on CPU-only workers that lack CUDA libs (see #2941).
_LAZY_SUBMODULES = {"eval", "eval_harness"}


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"levanter.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "1.2"
