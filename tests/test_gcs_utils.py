# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from marin.utilities.gcs_utils import get_vm_region


def test_get_vm_region_returns_region():
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"projects/12345/zones/us-central2-b"
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = lambda s, *a: None

    with patch("iris.temp_buckets.urllib.request.urlopen", return_value=mock_resp):
        assert get_vm_region() == "us-central2"


def test_get_vm_region_raises_when_unavailable():
    with (
        patch("iris.temp_buckets.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        pytest.raises(ValueError),
    ):
        get_vm_region()
