# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cloudflare Tunnel integration for exposing the Iris controller publicly.

Creates a Cloudflare Tunnel and DNS CNAME so the controller dashboard is
accessible at ``marin-<nonce>.iris-ops.dev`` without a public IP address or
SSH port-forwarding.

Lifecycle:
    1. ``start_tunnel()`` — create tunnel, set DNS CNAME, launch ``cloudflared``.
    2. ``stop_tunnel()``  — kill ``cloudflared``, delete DNS record, delete tunnel.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

CLOUDFLARE_API_BASE = "https://api.cloudflare.com/client/v4"

# How long to wait for cloudflared to establish the tunnel connection.
TUNNEL_HEALTH_TIMEOUT_SECONDS = 30


@dataclass
class TunnelConfig:
    """Configuration for the Cloudflare Tunnel integration."""

    enabled: bool = False
    domain: str = "iris-ops.dev"
    cloudflare_account_id: str = ""
    cloudflare_zone_id: str = ""
    api_token: str = ""
    cluster_name: str = ""

    @property
    def subdomain(self) -> str:
        """Derive subdomain from cluster name + truncated hash for obscurity."""
        if not self.cluster_name:
            return "marin"
        nonce = hashlib.sha256(self.cluster_name.encode()).hexdigest()[:8]
        return f"marin-{nonce}"

    @property
    def fqdn(self) -> str:
        return f"{self.subdomain}.{self.domain}"

    @property
    def public_url(self) -> str:
        return f"https://{self.fqdn}"


@dataclass
class TunnelHandle:
    """Running tunnel state — returned by ``start_tunnel()``."""

    tunnel_id: str
    tunnel_token: str
    dns_record_id: str
    public_url: str
    process: subprocess.Popen | None = None


def _cf_headers(api_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }


def _cf_request(
    client: httpx.Client,
    method: str,
    path: str,
    api_token: str,
    json_body: dict | None = None,
) -> dict:
    """Make a Cloudflare API request and return the result dict.

    Raises ``RuntimeError`` on API errors.
    """
    url = f"{CLOUDFLARE_API_BASE}{path}"
    resp = client.request(method, url, headers=_cf_headers(api_token), json=json_body)
    data = resp.json()
    if not data.get("success", False):
        errors = data.get("errors", [])
        raise RuntimeError(f"Cloudflare API error ({method} {path}): {errors}")
    return data


def _find_existing_tunnel(
    client: httpx.Client,
    account_id: str,
    tunnel_name: str,
    api_token: str,
) -> dict | None:
    """Find a non-deleted tunnel by name, or return None."""
    data = _cf_request(
        client,
        "GET",
        f"/accounts/{account_id}/cfd_tunnel?name={tunnel_name}&is_deleted=false",
        api_token,
    )
    tunnels = data.get("result", [])
    return tunnels[0] if tunnels else None


def _create_tunnel(
    client: httpx.Client,
    account_id: str,
    tunnel_name: str,
    api_token: str,
) -> dict:
    """Create a new Cloudflare Tunnel. Returns the tunnel object."""
    # Generate a tunnel secret (32 bytes, base64-encoded by Cloudflare)
    import secrets

    tunnel_secret = secrets.token_urlsafe(32)
    data = _cf_request(
        client,
        "POST",
        f"/accounts/{account_id}/cfd_tunnel",
        api_token,
        json_body={"name": tunnel_name, "tunnel_secret": tunnel_secret},
    )
    return data["result"]


def _get_tunnel_token(
    client: httpx.Client,
    account_id: str,
    tunnel_id: str,
    api_token: str,
) -> str:
    """Fetch the token needed to run ``cloudflared tunnel run``."""
    data = _cf_request(
        client,
        "GET",
        f"/accounts/{account_id}/cfd_tunnel/{tunnel_id}/token",
        api_token,
    )
    return data["result"]


def _configure_tunnel_ingress(
    client: httpx.Client,
    account_id: str,
    tunnel_id: str,
    fqdn: str,
    controller_port: int,
    api_token: str,
) -> None:
    """Set the tunnel ingress rules to route traffic to the local controller."""
    _cf_request(
        client,
        "PUT",
        f"/accounts/{account_id}/cfd_tunnel/{tunnel_id}/configurations",
        api_token,
        json_body={
            "config": {
                "ingress": [
                    {
                        "hostname": fqdn,
                        "service": f"http://localhost:{controller_port}",
                    },
                    {
                        "service": "http_status:404",
                    },
                ]
            }
        },
    )


def _find_dns_record(
    client: httpx.Client,
    zone_id: str,
    fqdn: str,
    api_token: str,
) -> dict | None:
    """Find a DNS CNAME record for fqdn, or return None."""
    data = _cf_request(
        client,
        "GET",
        f"/zones/{zone_id}/dns_records?type=CNAME&name={fqdn}",
        api_token,
    )
    records = data.get("result", [])
    return records[0] if records else None


def _upsert_dns_record(
    client: httpx.Client,
    zone_id: str,
    fqdn: str,
    tunnel_id: str,
    api_token: str,
) -> str:
    """Create or update the DNS CNAME record. Returns the record ID."""
    cname_target = f"{tunnel_id}.cfargotunnel.com"
    existing = _find_dns_record(client, zone_id, fqdn, api_token)
    if existing:
        record_id = existing["id"]
        _cf_request(
            client,
            "PATCH",
            f"/zones/{zone_id}/dns_records/{record_id}",
            api_token,
            json_body={
                "type": "CNAME",
                "name": fqdn,
                "content": cname_target,
                "proxied": True,
            },
        )
        logger.info("Updated DNS CNAME %s → %s", fqdn, cname_target)
        return record_id
    else:
        data = _cf_request(
            client,
            "POST",
            f"/zones/{zone_id}/dns_records",
            api_token,
            json_body={
                "type": "CNAME",
                "name": fqdn,
                "content": cname_target,
                "proxied": True,
            },
        )
        record_id = data["result"]["id"]
        logger.info("Created DNS CNAME %s → %s", fqdn, cname_target)
        return record_id


def _delete_dns_record(
    client: httpx.Client,
    zone_id: str,
    record_id: str,
    api_token: str,
) -> None:
    """Delete a DNS record by ID."""
    _cf_request(
        client,
        "DELETE",
        f"/zones/{zone_id}/dns_records/{record_id}",
        api_token,
    )
    logger.info("Deleted DNS record %s", record_id)


def _delete_tunnel(
    client: httpx.Client,
    account_id: str,
    tunnel_id: str,
    api_token: str,
) -> None:
    """Delete a Cloudflare Tunnel."""
    _cf_request(
        client,
        "DELETE",
        f"/accounts/{account_id}/cfd_tunnel/{tunnel_id}",
        api_token,
    )
    logger.info("Deleted tunnel %s", tunnel_id)


def _launch_cloudflared(tunnel_token: str) -> subprocess.Popen:
    """Launch ``cloudflared`` as a background subprocess."""
    cmd = [
        "cloudflared",
        "tunnel",
        "--no-autoupdate",
        "run",
        "--token",
        tunnel_token,
    ]
    logger.info("Starting cloudflared tunnel connector")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Give cloudflared a moment to start and fail fast if binary is missing
    time.sleep(2)
    if proc.poll() is not None:
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        raise RuntimeError(f"cloudflared exited immediately (rc={proc.returncode}): {stderr}")
    logger.info("cloudflared started (pid=%d)", proc.pid)
    return proc


def start_tunnel(config: TunnelConfig, controller_port: int) -> TunnelHandle:
    """Create a Cloudflare Tunnel and DNS record, then launch cloudflared.

    Idempotent: reuses an existing tunnel with the same name if present.
    """
    if not config.api_token:
        raise ValueError("TunnelConfig.api_token is required")
    if not config.cloudflare_account_id:
        raise ValueError("TunnelConfig.cloudflare_account_id is required")
    if not config.cloudflare_zone_id:
        raise ValueError("TunnelConfig.cloudflare_zone_id is required")

    tunnel_name = f"iris-{config.subdomain}"
    fqdn = config.fqdn

    with httpx.Client(timeout=30) as client:
        # 1. Find or create tunnel
        existing = _find_existing_tunnel(client, config.cloudflare_account_id, tunnel_name, config.api_token)
        if existing:
            tunnel_id = existing["id"]
            logger.info("Reusing existing tunnel %s (%s)", tunnel_name, tunnel_id)
        else:
            tunnel = _create_tunnel(client, config.cloudflare_account_id, tunnel_name, config.api_token)
            tunnel_id = tunnel["id"]
            logger.info("Created tunnel %s (%s)", tunnel_name, tunnel_id)

        # 2. Configure tunnel ingress
        _configure_tunnel_ingress(
            client,
            config.cloudflare_account_id,
            tunnel_id,
            fqdn,
            controller_port,
            config.api_token,
        )

        # 3. Get tunnel token for cloudflared
        tunnel_token = _get_tunnel_token(client, config.cloudflare_account_id, tunnel_id, config.api_token)

        # 4. Upsert DNS CNAME
        dns_record_id = _upsert_dns_record(client, config.cloudflare_zone_id, fqdn, tunnel_id, config.api_token)

    # 5. Launch cloudflared
    proc = _launch_cloudflared(tunnel_token)

    public_url = config.public_url
    logger.info("Tunnel active: %s", public_url)

    return TunnelHandle(
        tunnel_id=tunnel_id,
        tunnel_token=tunnel_token,
        dns_record_id=dns_record_id,
        public_url=public_url,
        process=proc,
    )


def stop_tunnel(
    handle: TunnelHandle,
    config: TunnelConfig,
    delete_tunnel: bool = False,
) -> None:
    """Stop cloudflared and optionally clean up DNS/tunnel.

    By default, DNS records and the tunnel are preserved so the same URL keeps
    working across controller restarts. Pass ``delete_tunnel=True`` for full
    cleanup (e.g. on ``iris cluster stop``).
    """
    # Kill cloudflared process
    if handle.process and handle.process.poll() is None:
        logger.info("Stopping cloudflared (pid=%d)", handle.process.pid)
        handle.process.terminate()
        try:
            handle.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            handle.process.kill()
            handle.process.wait()
        logger.info("cloudflared stopped")

    if not delete_tunnel:
        return

    if not config.api_token:
        logger.warning("No API token — skipping DNS/tunnel cleanup")
        return

    with httpx.Client(timeout=30) as client:
        # Delete DNS record first (tunnel must have no routes to delete)
        if handle.dns_record_id:
            try:
                _delete_dns_record(client, config.cloudflare_zone_id, handle.dns_record_id, config.api_token)
            except RuntimeError:
                logger.warning("Failed to delete DNS record %s", handle.dns_record_id, exc_info=True)

        if handle.tunnel_id:
            try:
                _delete_tunnel(client, config.cloudflare_account_id, handle.tunnel_id, config.api_token)
            except RuntimeError:
                logger.warning("Failed to delete tunnel %s", handle.tunnel_id, exc_info=True)
