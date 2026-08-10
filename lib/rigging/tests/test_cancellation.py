# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from rigging.cancellation import CancellationToken, cancellation_scope, current_cancellation_token


def test_cancellation_token_calls_registered_callback_once():
    token = CancellationToken()
    reasons: list[str] = []
    token.add_callback(reasons.append)

    token.cancel("lease lost")
    token.cancel("second reason")

    assert token.cancelled
    assert token.reason == "lease lost"
    assert reasons == ["lease lost"]


def test_cancellation_token_removes_callback():
    token = CancellationToken()
    reasons: list[str] = []
    remove_callback = token.add_callback(reasons.append)

    remove_callback()
    token.cancel("lease lost")

    assert reasons == []


def test_cancellation_token_calls_callback_added_after_cancellation():
    token = CancellationToken()
    token.cancel("lease lost")
    reasons: list[str] = []

    token.add_callback(reasons.append)

    assert reasons == ["lease lost"]


def test_cancellation_scope_restores_prior_token():
    outer = CancellationToken()
    inner = CancellationToken()

    assert current_cancellation_token() is None
    with cancellation_scope(outer):
        assert current_cancellation_token() is outer
        with cancellation_scope(inner):
            assert current_cancellation_token() is inner
        assert current_cancellation_token() is outer
    assert current_cancellation_token() is None
