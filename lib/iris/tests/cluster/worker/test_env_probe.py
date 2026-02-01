# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for worker environment probing — preemptible detection."""

from unittest.mock import patch

from iris.cluster.worker.env_probe import _detect_preemptible


def test_detect_preemptible_falls_back_to_extra_attributes():
    """When GCP metadata is unavailable, falls back to extra_attributes."""
    with patch("iris.cluster.worker.env_probe.urllib.request.urlopen", side_effect=OSError):
        assert _detect_preemptible({"preemptible": "true"}) is True
        assert _detect_preemptible({"preemptible": "false"}) is False
        assert _detect_preemptible({}) is False
