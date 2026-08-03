# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from marin.inference import iris_vllm


def test_interface_for_ipv4_finds_loopback():
    assert iris_vllm._interface_for_ipv4("127.0.0.1") == "lo"


def test_wait_until_retries_transient_resolver_unavailability(monkeypatch):
    attempts = 0

    def transient_outage() -> bool:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectError(Code.UNAVAILABLE, "resolver unavailable")
        return True

    monkeypatch.setattr(iris_vllm, "_POLL_SECONDS", 0.001)
    iris_vllm._wait_until(transient_outage, error_message="did not recover")
    assert attempts == 2
