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

"""Tests for RPC error handling utilities."""

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.rpc.errors import (
    extract_error_details,
    rpc_error_handler,
)


def test_rpc_error_handler_preserves_connect_errors():
    """rpc_error_handler passes through existing ConnectErrors unchanged."""
    with pytest.raises(ConnectError) as exc_info:
        with rpc_error_handler("test operation"):
            raise ConnectError(Code.NOT_FOUND, "not found")

    error = exc_info.value
    assert error.code == Code.NOT_FOUND
    assert error.message == "not found"


def test_rpc_error_handler_includes_traceback_in_details():
    """rpc_error_handler attaches ErrorDetails with full traceback."""
    with pytest.raises(ConnectError) as exc_info:
        with rpc_error_handler("test operation"):
            raise ValueError("test error")

    error = exc_info.value
    details = extract_error_details(error)

    assert details is not None
    assert "ValueError" in details.exception_type
    assert "test_rpc_error_handler_includes_traceback_in_details" in details.traceback
    assert details.timestamp_ms > 0


def test_extract_error_details_returns_none_for_no_details():
    """extract_error_details returns None when no ErrorDetails attached."""
    error = ConnectError(Code.INTERNAL, "simple error")
    assert extract_error_details(error) is None
