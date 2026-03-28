# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from marin.dispatch.schema import (
    RunPointer,
    RunTrack,
)


def test_run_pointer_validation_ray_missing():
    with pytest.raises(ValueError, match="ray config"):
        RunPointer(track=RunTrack.RAY)


def test_run_pointer_validation_iris_missing():
    with pytest.raises(ValueError, match="iris config"):
        RunPointer(track=RunTrack.IRIS)
