# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from forge._compat import is_text, recover


def test_recover_preserves_unhandled_group_members() -> None:
    def fail() -> str:
        raise ExceptionGroup("mixed", [RuntimeError("retry"), TypeError("bug")])

    with pytest.raises(ExceptionGroup) as caught:
        recover(fail, "fallback")
    assert [type(error) for error in caught.value.exceptions] == [TypeError]


def test_recover_returns_typed_result() -> None:
    value: object = recover(lambda: "completed", "fallback").value
    assert is_text(value)
    assert value.upper() == "COMPLETED"


def test_recover_returns_fallback_for_declared_runtime_failure() -> None:
    def fail() -> str:
        raise ExceptionGroup("recoverable", [RuntimeError("unavailable")])

    assert recover(fail, "fallback").value == "fallback"
