# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behaviour tests for deferred artifact versions (``marin.execution.build_context``).

Covers the resolution contract (explicit wins, codex fallback, no-context error), context nesting
and restoration on exception, and the unused-override signal that guards against typos.
"""

import pytest
from marin.execution.build_context import (
    BuildContext,
    VersionCodex,
    build_context,
    current_build_context,
    resolve_version,
)


def _codex(default="2026.07.16", **overrides):
    return VersionCodex(default=default, overrides=overrides)


def test_version_for_prefers_override_then_default():
    codex = VersionCodex(default="2026.07.16", overrides={"tokenizer": "2026.06.28"})
    assert codex.version_for("tokenizer") == "2026.06.28"
    assert codex.version_for("anything-else") == "2026.07.16"


def test_resolve_version_explicit_wins_over_context():
    # An explicit version is not overridable by the codex: deferral is opt-in per call.
    with build_context(BuildContext(versions=_codex(**{"data": "2026.06.28"}))):
        assert resolve_version("data", "2026.01.01") == "2026.01.01"


def test_resolve_version_defers_to_codex_inside_context():
    with build_context(BuildContext(versions=VersionCodex(default="dev", overrides={"pin-me": "2026.06.28"}))):
        assert resolve_version("pin-me", None) == "2026.06.28"
        assert resolve_version("free", None) == "dev"


def test_resolve_version_without_context_raises():
    # The safety contract: a deferred version never silently defaults outside a BuildContext.
    assert current_build_context() is None
    with pytest.raises(ValueError, match="no active BuildContext"):
        resolve_version("orphan", None)


def test_build_context_nests_and_restores_on_exception():
    outer = BuildContext(versions=_codex(default="2026.07.16"))
    inner = BuildContext(versions=_codex(default="2026.06.28"))
    with build_context(outer):
        assert resolve_version("x", None) == "2026.07.16"
        with pytest.raises(RuntimeError):
            with build_context(inner):
                assert resolve_version("x", None) == "2026.06.28"
                raise RuntimeError("boom")
        # The inner context is torn down even though the block raised.
        assert resolve_version("x", None) == "2026.07.16"
    assert current_build_context() is None


def test_unused_overrides_flags_names_never_consulted():
    codex = VersionCodex(default="dev", overrides={"used": "2026.06.28", "typo": "2026.06.28"})
    codex.version_for("used")
    codex.version_for("some-default-artifact")
    assert codex.unused_overrides() == frozenset({"typo"})


def test_codex_reports_which_override_is_malformed():
    # The version grammar itself is covered by validate_version's tests; here we only pin that the
    # codex validates overrides and names the offending one, so a bad --override is legible.
    with pytest.raises(ValueError, match="override 'a'"):
        VersionCodex(default="dev", overrides={"a": "not-a-version"})
