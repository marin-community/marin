# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Login orchestration: one command, one credential store.

This is the *orchestration* half of "one login" — the sequencing that the codex
re-review pulled up out of rigging into the umbrella. rigging supplies the
mechanism (the IAP desktop-OAuth flow, the opaque credential store); iris supplies
the token exchange (its ``Login`` RPC mints the app JWT). ``marin_cluster`` is the
only layer that may import both, so it owns the flow:

    desktop IAP OAuth  ->  id_token  ->  iris Login exchange  ->  app JWT
                           (+ refresh token cached)              (one store)

The two external steps are injectable seams (``desktop_login`` and ``exchange``)
so the flow is testable without a browser or a live controller; the defaults wire
the real rigging and iris implementations lazily, keeping this module importable
without the IAP extra or iris installed.
"""

from collections.abc import Callable
from dataclasses import dataclass

from rigging.cluster_manifest import AuthProvider, IapAuth
from rigging.credential_store import CredentialRecord, delete_credentials, save_credentials

from marin_cluster.config import ClusterConfig

# Seam: run the IAP desktop-OAuth flow, returning (id_token, refresh_token).
DesktopLogin = Callable[[IapAuth], tuple[str, str]]
# Seam: exchange an edge ``id_token`` at ``endpoint`` for an app token + identity.
Exchange = Callable[[str, str], "LoginResult"]


@dataclass(frozen=True)
class LoginResult:
    """The outcome of an app-token exchange."""

    app_token: str
    user_id: str


def login(
    cluster: str | None = None,
    *,
    desktop_login: DesktopLogin | None = None,
    exchange: Exchange | None = None,
) -> CredentialRecord:
    """Authenticate to ``cluster`` and cache one credential record.

    Raises:
        ValueError: if the cluster is not IAP-fronted, or its ``auth.iap`` is
            missing the desktop OAuth client id/secret needed to log in.
    """
    cfg = ClusterConfig.load(cluster)
    auth = cfg.auth
    if auth.provider is not AuthProvider.IAP or auth.iap is None:
        raise ValueError(
            f"`marin-cluster login` supports IAP clusters; {cfg.name!r} uses auth provider "
            f"{auth.provider.value!r}. (gcp/static clusters authenticate per-call.)"
        )
    iap = auth.iap
    if not iap.desktop_oauth_client_id or not iap.desktop_oauth_client_secret:
        raise ValueError(
            f"cluster {cfg.name!r} auth.iap has no desktop_oauth_client_id/secret; run "
            "`marin-cluster admin iap deploy` and copy its auth block into config/<cluster>.yaml"
        )

    run_desktop = desktop_login or _default_desktop_login
    run_exchange = exchange or _iris_exchange

    id_token, refresh_token = run_desktop(iap)
    result = run_exchange(iap.url, id_token)
    record = CredentialRecord(
        cluster=cfg.name,
        endpoint=iap.url,
        edge_refresh_token=refresh_token,
        app_token=result.app_token,
        metadata={"user_id": result.user_id},
    )
    save_credentials(record)
    return record


def logout(cluster: str | None = None) -> bool:
    """Drop cached credentials for ``cluster``. Returns True if any were removed."""
    cfg = ClusterConfig.load(cluster)
    return delete_credentials(cfg.name)


def _default_desktop_login(iap: IapAuth) -> tuple[str, str]:
    """Run rigging's IAP desktop-OAuth flow for ``iap``'s client credentials."""
    from rigging.auth import run_iap_desktop_login  # noqa: PLC0415  (optional IAP extra)

    assert iap.desktop_oauth_client_id is not None and iap.desktop_oauth_client_secret is not None
    return run_iap_desktop_login(iap.desktop_oauth_client_id, iap.desktop_oauth_client_secret)


def _iris_exchange(endpoint: str, id_token: str) -> LoginResult:
    """Exchange an IAP ``id_token`` for an Iris app JWT via the controller's Login RPC."""
    try:
        from iris.cli.connect import rpc_client  # noqa: PLC0415  (optional iris install)
        from iris.rpc import job_pb2  # noqa: PLC0415
        from iris.rpc.auth import ClientCredentials, StaticTokenProvider  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("exchanging an IAP token for an app token needs iris; install `marin-cluster[iris]`") from e

    credentials = ClientCredentials(iap_provider=StaticTokenProvider(id_token))
    with rpc_client(endpoint, credentials) as client:
        response = client.login(job_pb2.LoginRequest(identity_token=id_token))
    return LoginResult(app_token=response.token, user_id=response.user_id)
