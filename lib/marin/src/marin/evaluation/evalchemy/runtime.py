# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evalchemy's isolated runtime and launch identity."""

from rigging.provenance import launch_provenance

EVALCHEMY_RUNTIME_PROJECT = "vendor/evalchemy"
EVALCHEMY_REQUIRED_EXTRAS = ("serve-eval",)
EVALCHEMY_PYTHON_VERSION = "3.12"


def evalchemy_runtime_descriptor(extras: tuple[str, ...]) -> str:
    selected = ",".join(sorted(set(extras)))
    tree = launch_provenance().dedup_key
    if not tree:
        raise RuntimeError(
            "Evalchemy evaluation requires Git provenance or MARIN_PROVENANCE for cache and resume identity"
        )
    return f"evalchemy[{selected}] @ {EVALCHEMY_RUNTIME_PROJECT}#tree={tree}"
