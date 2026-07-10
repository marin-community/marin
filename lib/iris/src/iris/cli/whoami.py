# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``iris whoami`` — dump the caller's login status for one or all clusters.

Reports email, token expiry, and login method by minting the cluster's edge token
on demand (the identity a real RPC would present), degrading to a cached offline
view when minting fails. It never connects to a controller, so it works while a
cluster is unreachable or a login has lapsed.
"""

import json
import logging
from datetime import UTC, datetime

import click
from rigging.cluster_manifest import AuthProvider, ClusterAuth, IapAuth
from rigging.config_discovery import list_cluster_configs
from rigging.credential_store import list_cached_clusters, load_credentials
from rigging.login_status import LoginMethod, LoginState, LoginStatus, resolve_login_status

from iris.cli.connect import IRIS_CLUSTER_CONFIG_DIRS, cluster_auth_for
from iris.cluster.config import load_config

logger = logging.getLogger(__name__)

_METHOD_LABELS = {
    LoginMethod.NETWORK_TRUST: "network trust (no login required)",
    LoginMethod.IAP_BROWSER: "IAP (browser login)",
    LoginMethod.IAP_SERVICE_ACCOUNT: "IAP (service account)",
}
_STATE_LABELS = {
    LoginState.ACTIVE: "active",
    LoginState.OFFLINE: "offline — token mint failed",
    LoginState.NOT_LOGGED_IN: "not logged in",
    LoginState.NO_LOGIN_REQUIRED: "no login required",
}


def _status_for(cluster: str, config) -> LoginStatus:
    """Resolve one cluster's login status from its config, falling back to the cache.

    A cluster you logged into by raw ``--controller-url`` has a cached refresh token
    but no IAP config; a cached refresh token is only ever an IAP browser login, so
    synthesize a minimal IAP auth (default desktop client) to run the same mint path.
    """
    auth = cluster_auth_for(config)
    if auth.provider is not AuthProvider.IAP:
        record = load_credentials(cluster)
        if record is not None and record.edge_refresh_token is not None:
            auth = ClusterAuth(AuthProvider.IAP, iap=IapAuth(url=record.endpoint))
    return resolve_login_status(cluster, auth)


def _all_statuses() -> list[LoginStatus]:
    """Login status for every configured cluster plus any cached-only extras."""
    configured = sorted(list_cluster_configs(IRIS_CLUSTER_CONFIG_DIRS).items())
    statuses = [_status_for(name, load_config(str(path))) for name, path in configured]
    seen = {name for name, _ in configured}
    statuses.extend(_status_for(name, None) for name in list_cached_clusters() if name not in seen)
    return statuses


def _humanize_delta(seconds: int) -> str:
    """A compact ``2h5m`` / ``5m`` / ``30s`` rendering of a positive duration."""
    if seconds >= 3600:
        return f"{seconds // 3600}h{(seconds % 3600) // 60}m"
    if seconds >= 60:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def _format_expiry(expiry: datetime | None, now: datetime) -> str:
    stamp = expiry.strftime("%Y-%m-%d %H:%M:%S UTC") if expiry is not None else None
    if expiry is None:
        return "—"
    seconds = int((expiry - now).total_seconds())
    relative = f"in {_humanize_delta(seconds)}" if seconds >= 0 else f"expired {_humanize_delta(-seconds)} ago"
    return f"{stamp} ({relative})"


def _render_block(status: LoginStatus, now: datetime) -> str:
    lines = [f"Cluster:       {status.cluster}"]
    if status.endpoint:
        lines.append(f"Endpoint:      {status.endpoint}")
    lines.append(f"Login method:  {_METHOD_LABELS.get(status.method, '—')}")
    lines.append(f"State:         {_STATE_LABELS[status.state]}")
    if status.email:
        suffix = "" if status.state is LoginState.ACTIVE else "  (from last login)"
        lines.append(f"Email:         {status.email}{suffix}")
    if status.state is LoginState.ACTIVE:
        lines.append(f"Token expiry:  {_format_expiry(status.token_expiry, now)}")
    if status.detail:
        lines.append(f"Detail:        {status.detail}")
    return "\n".join(lines)


def _to_json(status: LoginStatus) -> dict:
    return {
        "cluster": status.cluster,
        "endpoint": status.endpoint,
        "method": status.method.value if status.method is not None else None,
        "state": status.state.value,
        "email": status.email,
        "token_expiry": status.token_expiry.isoformat() if status.token_expiry is not None else None,
        "detail": status.detail,
    }


@click.command("whoami")
@click.option(
    "--all", "all_clusters", is_flag=True, help="Report every configured and cached cluster, not just the current one."
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format (default: text).",
)
@click.pass_context
def whoami(ctx, all_clusters: bool, output_format: str):
    """Show the caller's login status: email, token expiry, and login method.

    Defaults to the current cluster (from ``--cluster`` / ``--config``); pass
    ``--all`` to dump every configured and cached cluster. Mints the edge token on
    demand to report the live identity, falling back to the cached login offline.
    """
    if all_clusters:
        statuses = _all_statuses()
    else:
        obj = ctx.obj or {}
        statuses = [_status_for(obj.get("cluster_name") or "default", obj.get("config"))]

    if output_format == "json":
        click.echo(json.dumps([_to_json(s) for s in statuses], indent=2))
        return

    if not statuses:
        click.echo("No clusters configured and no cached logins found.")
        return
    now = datetime.now(tz=UTC)
    click.echo("\n\n".join(_render_block(s, now) for s in statuses))
