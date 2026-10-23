"""Regression: TrialPaths.mkdir()/chmod_dir() must be a no-op on object storage.

A per-trial `mkdir` (+ `chmod`) on an `s3://`/`gs://` `trial_dir` is a wasted BLOCKING
cloud round-trip (+ TLS handshake). In the RolloutCoordinator it runs on the asyncio
dispatch loop and serializes trial creation — measured pegging coordinators at 100%
single-core and capping the `generate()` issue rate (engines under-fed, Waiting=0).
Object storage has no real directories (a prefix exists once the first object is
written) and no POSIX mode bits, so the local-only ops are pure waste there. Local
filesystems still need the dirs pre-created.
"""

import os
import tempfile
from unittest import mock

import pytest
from upath import UPath

from harbor.models.trial.paths import TrialPaths


@pytest.mark.unit
def test_mkdir_creates_local_dirs():
    d = tempfile.mkdtemp()
    tp = TrialPaths(trial_dir=UPath(d))
    assert tp._is_local() is True
    tp.mkdir()
    for sub in ("agent", "verifier", "artifacts"):
        assert os.path.isdir(os.path.join(d, sub)), f"{sub} not created locally"


@pytest.mark.unit
def test_mkdir_is_noop_on_object_storage():
    tp = TrialPaths(trial_dir=UPath("s3://fake-bucket-xyz/trial-123"))
    assert tp.trial_dir.protocol == "s3"
    assert tp._is_local() is False
    # Must touch NO cloud path object — patch UPath.mkdir/chmod to assert they're
    # never called (a real call would block on a network round-trip).
    with (
        mock.patch.object(UPath, "mkdir") as mk,
        mock.patch.object(UPath, "chmod") as ch,
    ):
        tp.mkdir()
        tp.chmod_dir()
    mk.assert_not_called()
    ch.assert_not_called()
