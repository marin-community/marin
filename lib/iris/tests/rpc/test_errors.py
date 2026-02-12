# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from connectrpc.code import Code

from iris.rpc.errors import connect_error_with_traceback, extract_error_details


def test_connect_error_with_traceback_populates_timestamp() -> None:
    try:
        raise ValueError("boom")
    except ValueError as exc:
        err = connect_error_with_traceback(Code.INTERNAL, "Error launching job", exc=exc)

    details = extract_error_details(err)
    assert details is not None
    assert details.exception_type.endswith("ValueError")
    assert details.timestamp.epoch_ms > 0
