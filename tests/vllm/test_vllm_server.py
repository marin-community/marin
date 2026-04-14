# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for marin.inference.vllm_server helpers (do not require vllm)."""

import pytest

from marin.inference.vllm_server import resolve_vllm_mode


def test_default_mode_is_native(monkeypatch):
    """Default mode must be native so Iris workers without /var/run/docker.sock work out of the box."""
    monkeypatch.delenv("MARIN_VLLM_MODE", raising=False)
    assert resolve_vllm_mode(None) == "native"


def test_env_var_can_select_docker(monkeypatch):
    monkeypatch.setenv("MARIN_VLLM_MODE", "docker")
    assert resolve_vllm_mode(None) == "docker"


def test_explicit_argument_overrides_env(monkeypatch):
    monkeypatch.setenv("MARIN_VLLM_MODE", "docker")
    assert resolve_vllm_mode("native") == "native"


def test_unknown_mode_raises(monkeypatch):
    monkeypatch.setenv("MARIN_VLLM_MODE", "kubernetes")
    with pytest.raises(ValueError, match="MARIN_VLLM_MODE"):
        resolve_vllm_mode(None)
