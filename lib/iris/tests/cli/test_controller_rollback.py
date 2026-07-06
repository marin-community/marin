# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for _resolve_rollback_target — the flag-to-(image, checkpoint) decision in cluster.py."""

import click
import pytest
from iris.cli.cluster import _resolve_rollback_target
from iris.cluster.controller.rollout import RolloutPhase, RolloutRecord

STATE_DIR = "gs://b/marin/state"
COMMITTED = RolloutRecord(
    phase=RolloutPhase.COMMITTED,
    image="img:new",
    previous_image="img:old",
    rollback_checkpoint="gs://b/marin/state/controller-state/123",
)


def _resolve(**kwargs):
    base = dict(
        remote_state_dir=STATE_DIR,
        prior_record=None,
        rollback=False,
        image_override=None,
        restore_checkpoint=None,
    )
    base.update(kwargs)
    prior = base.pop("prior_record")
    remote = base.pop("remote_state_dir")
    return _resolve_rollback_target(remote, prior, **base)


def test_forward_deploy_has_no_rollback_target():
    assert _resolve() == (None, None)


def test_pinned_forward_redeploy_is_not_a_rollback():
    # --image alone is a forward redeploy; the image override is applied later.
    assert _resolve(image_override="img:pinned") == (None, None)


def test_rollback_reads_previous_image_and_checkpoint():
    assert _resolve(rollback=True, prior_record=COMMITTED) == (
        "img:old",
        "gs://b/marin/state/controller-state/123",
    )


def test_rollback_without_record_errors():
    with pytest.raises(click.ClickException, match="No deploy to roll back to"):
        _resolve(rollback=True, prior_record=None)


def test_rollback_without_previous_image_errors():
    first_deploy = RolloutRecord(phase=RolloutPhase.COMMITTED, image="img:first")
    with pytest.raises(click.ClickException, match="No deploy to roll back to"):
        _resolve(rollback=True, prior_record=first_deploy)


def test_rollback_needs_remote_state_dir():
    with pytest.raises(click.ClickException, match="remote_state_dir"):
        _resolve(rollback=True, remote_state_dir=None, prior_record=COMMITTED)


def test_rollback_rejects_manual_flags():
    with pytest.raises(click.ClickException, match="don't combine"):
        _resolve(rollback=True, prior_record=COMMITTED, image_override="img:x")


def test_manual_rollback_returns_image_and_checkpoint():
    assert _resolve(image_override="img:old", restore_checkpoint="gs://b/cs/9") == ("img:old", "gs://b/cs/9")


def test_restore_checkpoint_requires_image():
    with pytest.raises(click.ClickException, match="requires --image"):
        _resolve(restore_checkpoint="gs://b/cs/9")
