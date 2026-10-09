# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Renew and retain the worker's controller credential."""

from pathlib import Path

from rigging.auth import BearerTokenInjector, RefreshingTokenProvider, StaticTokenProvider

from iris.rpc import controller_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync

WORKER_TOKEN_REFRESH_MARGIN = 86400


def worker_token_provider(controller_address: str, token: str, cache_path: Path) -> RefreshingTokenProvider:
    def renew(current_token: str) -> str:
        # This exchange must use the current token directly, not recursively
        # invoke the provider which is waiting for the exchange to finish.
        client = ControllerServiceClientSync(
            address=controller_address,
            timeout_ms=10_000,
            interceptors=(BearerTokenInjector(StaticTokenProvider(current_token), "authorization"),),
        )
        try:
            return client.renew_worker_token(controller_pb2.Controller.RenewWorkerTokenRequest()).token
        finally:
            client.close()

    return RefreshingTokenProvider(
        token,
        renew,
        refresh_margin=WORKER_TOKEN_REFRESH_MARGIN,
        cache_path=cache_path,
    )
