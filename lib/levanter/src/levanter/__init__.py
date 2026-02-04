# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

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

__version__ = "1.2"


def __getattr__(name: str):
    """Lazy import submodules to avoid loading heavy dependencies (torch, jax) on package import."""
    import importlib

    if name == "analysis":
        return importlib.import_module("levanter.analysis")
    elif name == "callbacks":
        return importlib.import_module("levanter.callbacks")
    elif name == "checkpoint":
        return importlib.import_module("levanter.checkpoint")
    elif name == "config":
        return importlib.import_module("levanter.config")
    elif name == "data":
        return importlib.import_module("levanter.data")
    elif name == "distributed":
        return importlib.import_module("levanter.distributed")
    elif name == "eval":
        return importlib.import_module("levanter.eval")
    elif name == "eval_harness":
        return importlib.import_module("levanter.eval_harness")
    elif name == "models":
        return importlib.import_module("levanter.models")
    elif name == "optim":
        return importlib.import_module("levanter.optim")
    elif name == "tracker":
        return importlib.import_module("levanter.tracker")
    elif name == "trainer":
        return importlib.import_module("levanter.trainer")
    elif name == "visualization":
        return importlib.import_module("levanter.visualization")
    elif name == "grug":
        return importlib.import_module("levanter.grug")
    elif name == "current_tracker":
        from levanter.tracker import current_tracker
        return current_tracker
    elif name == "initialize":
        from levanter.trainer import initialize
        return initialize
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
