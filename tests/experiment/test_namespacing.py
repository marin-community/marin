# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``user_namespaced_name`` isolates mutable checkpoints per user but leaves fixed ones shared."""

import pytest
from marin.experiment.namespacing import user_namespaced_name


@pytest.fixture
def fixed_user(monkeypatch):
    monkeypatch.setattr("rigging.provenance._getuser", lambda: "alice")


@pytest.mark.parametrize("version", ["dev", "grug-dev", "iter04-dev"])
def test_mutable_version_namespaces_per_user(fixed_user, version):
    assert user_namespaced_name("grug/baseline", version) == "users/alice/grug/baseline"


@pytest.mark.parametrize("version", ["2026.06.28", "2026.06.28.2"])
def test_fixed_version_keeps_shared_name(fixed_user, version):
    assert user_namespaced_name("grug/baseline", version) == "grug/baseline"


def test_namespacing_is_idempotent(fixed_user):
    # The policy is applied at several entry points; re-applying it must not stack users/.
    once = user_namespaced_name("grug/baseline", "dev")
    assert user_namespaced_name(once, "dev") == once


def test_dev_version_requires_a_resolvable_user(monkeypatch):
    # Per-user isolation must fail loudly, never land everyone in a shared users/unknown/ bucket.
    monkeypatch.setattr("rigging.provenance._getuser", lambda: None)
    with pytest.raises(RuntimeError):
        user_namespaced_name("grug/baseline", "dev")
