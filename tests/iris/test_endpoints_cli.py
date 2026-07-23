# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace

from click.testing import CliRunner

from iris.cli import endpoints as endpoints_cli


class _Client:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.minted = []

    def list_endpoints(self, _request):
        return SimpleNamespace(endpoints=[self.endpoint] if self.endpoint else [])

    def mint_endpoint_token(self, request):
        self.minted.append(request)
        return SimpleNamespace(token="scoped-token")


def _invoke(monkeypatch, endpoint, *args):
    client = _Client(endpoint)

    @contextmanager
    def rpc_client(_ctx):
        yield client

    monkeypatch.setattr(endpoints_cli, "rpc_client_for_ctx", rpc_client)
    result = CliRunner().invoke(
        endpoints_cli.endpoints,
        ["wait-and-mint", "/serve/test", "--poll-seconds", "0.01", *args],
        obj={"config": SimpleNamespace(dashboard_url="https://iris.oa.dev")},
    )
    return result, client


def test_wait_and_mint_prints_only_capability_url(monkeypatch):
    endpoint = SimpleNamespace(peer_id="cw-us-east-02a")
    result, client = _invoke(monkeypatch, endpoint, "--require-peer")

    assert result.exit_code == 0, result.output
    assert result.output == "https://iris.oa.dev/proxy/t/scoped-token/serve.test/\n"
    assert len(client.minted) == 1


def test_wait_and_mint_rejects_local_endpoint_when_peer_required(monkeypatch):
    endpoint = SimpleNamespace(peer_id="")
    result, client = _invoke(monkeypatch, endpoint, "--require-peer")

    assert result.exit_code != 0
    assert "local; expected a federated peer mirror" in result.output
    assert client.minted == []


def test_wait_and_mint_rejects_nonpositive_intervals(monkeypatch):
    endpoint = SimpleNamespace(peer_id="cw-us-east-02a")
    result, client = _invoke(monkeypatch, endpoint, "--timeout-seconds", "0")

    assert result.exit_code != 0
    assert "timeout and poll intervals must be positive" in result.output
    assert client.minted == []
