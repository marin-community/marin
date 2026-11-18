# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Renew isolated proof identities without extending login or session policies."""

import tempfile
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from rigging.auth import RefreshingTokenProvider
from rigging.credentials import iap_edge_provider
from rigging.openbao import OpenBaoClient

from infra.auth.login import close_session, login, read_private, write_private


def google_identity(directory: Path, *, address: str, issuer: str, ca_file: str) -> str:
    """Use the enrolled desktop login, then close the temporary Bao session."""
    token = iap_edge_provider("marin").get_token()
    if token is None:
        raise ValueError("The enrolled desktop login is unavailable")
    with tempfile.TemporaryDirectory(prefix="auth-proof-login-") as scratch:
        temporary = Path(scratch)
        write_private(temporary / "enrollment.json", read_private(directory / "enrollment.json"))
        write_private(temporary / "google.jwt", token)
        try:
            login(temporary, address=address, issuer=issuer, ca_file=ca_file)
            return read_private(temporary / "identity.jwt")
        finally:
            if (temporary / "bao.token").exists():
                close_session(temporary, address=address, ca_file=ca_file, auth_url=None, grants=[])


@contextmanager
def refreshing_identity_file(path: Path, issue: Callable[[], str]) -> Iterator[None]:
    """Maintain an atomically replaced private JWT file during a long proof."""
    initial = issue()
    write_private(path, initial)
    provider = RefreshingTokenProvider(initial, lambda previous: issue(), refresh_margin=60, cache_path=path)
    stop = threading.Event()

    def renew() -> None:
        while not stop.wait(5):
            provider.get_token()

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="proof-identity") as executor:
        future = executor.submit(renew)
        try:
            yield
        finally:
            stop.set()
            future.result()


@contextmanager
def controller_identity_file(path: Path, *, address: str, token_file: Path, ca_file: str) -> Iterator[None]:
    """Refresh controller identity using its identity-only Bao session."""
    client = OpenBaoClient(address, token_file, ca_bundle=ca_file)
    with refreshing_identity_file(
        path, lambda: client.request("GET", "/identity/oidc/token/marin-auth")["data"]["token"]
    ):
        yield
