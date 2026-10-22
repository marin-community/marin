# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Private task credential files and the AWS credential_process entry point."""

from functools import cache
from pathlib import Path

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from rigging.auth import RefreshingTokenProvider
from rigging.credentials import ClientCredentials

from iris.cluster.credential_process import TaskCredentialFile, read_task_credential_file
from iris.rpc import controller_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync


def _renew(config: TaskCredentialFile) -> str:
    credentials = ClientCredentials(ca_bundle=Path(config.ca_file).read_bytes())
    with ControllerServiceClientSync(
        config.controller,
        http_client=credentials.http_client(),
        timeout_ms=30_000,
    ) as client:
        return client.renew_task_session(
            controller_pb2.Controller.RenewTaskSessionRequest(
                attempt_uid=config.attempt_uid,
                refresh_capability=config.refresh_capability,
            )
        ).token


class TaskSessionRolloverInterceptor:
    """Retry a rejected access token once using the separate refresh capability."""

    def __init__(self, tokens: RefreshingTokenProvider):
        self._tokens = tokens

    def _renew_rejected(self, ctx, error: ConnectError) -> None:
        if error.code != Code.UNAUTHENTICATED:
            raise error
        self._tokens.invalidate()
        ctx.request_headers()["authorization"] = f"Bearer {self._tokens.get_token()}"

    def intercept_unary_sync(self, call_next, request, ctx):
        try:
            return call_next(request, ctx)
        except ConnectError as error:
            self._renew_rejected(ctx, error)
        return call_next(request, ctx)

    async def intercept_unary(self, call_next, request, ctx):
        try:
            return await call_next(request, ctx)
        except ConnectError as error:
            self._renew_rejected(ctx, error)
        return await call_next(request, ctx)


@cache
def task_credentials(path: str) -> ClientCredentials:
    """Refresh control credentials independently of a training process's lifetime."""
    config = read_task_credential_file(path)
    tokens = RefreshingTokenProvider(_renew(config), lambda _old: _renew(config))
    return ClientCredentials(
        token_provider=tokens,
        ca_bundle=Path(config.ca_file).read_bytes(),
        extra_interceptors=(TaskSessionRolloverInterceptor(tokens),),
    )
