# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from levanter.callbacks import LambdaCallback


def test_lambda_callback_passes_force_when_supported():
    calls: list[bool] = []

    def fn(_info: Any, *, force: bool = False):
        calls.append(force)

    cb = LambdaCallback(fn)
    cb.on_step(None, force=True)

    assert calls == [True]


def test_lambda_callback_does_not_pass_force_when_not_supported():
    calls: list[Any] = []

    def fn(info: Any):
        calls.append(info)

    cb = LambdaCallback(fn)
    marker = object()
    cb.on_step(marker, force=True)

    assert calls == [marker]


def test_lambda_callback_passes_force_via_kwargs():
    calls: list[bool] = []

    def fn(_info: Any, **kwargs: Any):
        calls.append(bool(kwargs["force"]))

    cb = LambdaCallback(fn)
    cb.on_step(None, force=True)

    assert calls == [True]
