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

"""Tests for cluster manager lifecycle functions."""

from unittest.mock import MagicMock, patch

from iris.cluster.manager import stop_all
from iris.rpc import config_pb2


def _minimal_config() -> config_pb2.IrisClusterConfig:
    """Create a minimal config for testing stop_all."""
    config = config_pb2.IrisClusterConfig()
    config.controller.local.SetInParent()
    config.platform.local.SetInParent()
    return config


def test_stop_all_calls_platform_shutdown():
    """stop_all() must call platform.shutdown() to release resources."""
    mock_platform = MagicMock()
    mock_platform.list_slices.return_value = []

    config = _minimal_config()

    with patch("iris.cluster.manager.IrisConfig") as mock_iris_config_cls:
        mock_iris_config_cls.return_value.platform.return_value = mock_platform
        stop_all(config)

    mock_platform.shutdown.assert_called_once()


def test_stop_all_terminates_listed_slices():
    """stop_all() terminates every slice returned by list_slices for each scale group."""
    mock_slice_1 = MagicMock()
    mock_slice_2 = MagicMock()
    mock_platform = MagicMock()
    mock_platform.list_slices.return_value = [mock_slice_1, mock_slice_2]

    config = _minimal_config()
    config.scale_groups["group-a"].name = "group-a"
    config.scale_groups["group-a"].slice_template.SetInParent()

    with patch("iris.cluster.manager.IrisConfig") as mock_iris_config_cls:
        mock_iris_config_cls.return_value.platform.return_value = mock_platform
        stop_all(config)

    mock_slice_1.terminate.assert_called_once()
    mock_slice_2.terminate.assert_called_once()
    mock_platform.shutdown.assert_called_once()
