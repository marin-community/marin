# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The grug launchers defer their checkpoint version to the ambient build context.

Guards the adoption of :mod:`marin.execution.build_context` in the grug ``launch`` scripts: a
builder called with no explicit version resolves from the active :class:`VersionCodex` (by the
pre-namespacing name), and refuses to guess when no context is active.
"""

import pytest
from marin.execution.build_context import BuildContext, VersionCodex, build_context

from experiments.grug.base.launch import grug_base_trial
from experiments.grug.moe.launch import grug_moe_baseline

# (builder, the name it resolves/namespaces under — the override key a codex would use).
_LAUNCHERS = [
    (grug_base_trial, "grug/base-trial"),
    (grug_moe_baseline, "grug/4_10_baseline_moe"),
]


@pytest.mark.parametrize("builder, name", _LAUNCHERS)
def test_launcher_requires_a_version_source(builder, name):
    # No explicit version and no active context: deferral never silently defaults.
    with pytest.raises(ValueError, match="no active BuildContext"):
        builder()


@pytest.mark.parametrize("builder, name", _LAUNCHERS)
def test_launcher_takes_the_context_default(builder, name):
    # A calendar default resolves onto the checkpoint and keeps the shared (un-namespaced) name.
    with build_context(BuildContext(versions=VersionCodex(default="2026.07.16"))):
        step = builder()
    assert step.version == "2026.07.16"
    assert step.name == name


@pytest.mark.parametrize("builder, name", _LAUNCHERS)
def test_override_steers_the_launcher_by_name(builder, name):
    # An override keyed by the builder's name wins over the run-wide default.
    codex = VersionCodex(default="2026.07.16", overrides={name: "2026.06.28"})
    with build_context(BuildContext(versions=codex)):
        step = builder()
    assert step.version == "2026.06.28"
    assert not codex.unused_overrides()
