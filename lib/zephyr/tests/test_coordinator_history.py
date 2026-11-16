# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The optional history endpoint must not prevent a pipeline from starting."""

from types import SimpleNamespace

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from zephyr import coordinator


class _UnavailableEndpointClient:
    def resolve_endpoint(self, _name: str) -> str:
        raise ConnectError(Code.UNAVAILABLE, "controller unavailable")


def test_history_endpoint_rpc_failure_is_optional(monkeypatch) -> None:
    monkeypatch.setattr(coordinator, "get_iris_ctx", lambda: SimpleNamespace(client=_UnavailableEndpointClient()))

    assert coordinator._resolve_execution_history_url() is None
