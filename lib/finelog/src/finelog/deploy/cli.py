# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""finelog deploy CLI — config-driven deployment management.

Each subcommand takes a logical config name (or path), loads it via
`load_finelog_config`, and dispatches to either the GCE or Kubernetes
backend based on which `deployment.*` block the config sets. The CLI
itself is platform-agnostic; finelog owns the platform decision via its
config schema, mirroring how `iris cluster start` decides backend from
cluster yaml.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from enum import StrEnum

import click
import pyarrow as pa
from rigging.log_setup import configure_logging
from rigging.tunnel import GcpSshForwardTarget, K8sPortForwardTarget, TunnelTarget, open_tunnel

from finelog.client.log_client import LogClient
from finelog.deploy import _gcp, _k8s
from finelog.deploy.build import build_image as build_finelog_image
from finelog.deploy.config import FinelogConfig, load_finelog_config


def _tunnel_target(cfg: FinelogConfig) -> TunnelTarget:
    """Translate a finelog deployment block into a rigging tunnel target.

    The GCP path forwards ``deployment.gcp.service_account`` as the SSH
    impersonation principal — matching the deploy CLI's own SSH calls
    (see ``_gcp._ssh_args``), so this command works wherever
    ``finelog deploy status`` does.
    """
    if cfg.deployment.gcp is not None:
        gcp = cfg.deployment.gcp
        return GcpSshForwardTarget(
            project=gcp.project,
            zone=gcp.zone,
            instance=cfg.name,
            port=cfg.port,
            impersonate_service_account=gcp.service_account,
        )
    assert cfg.deployment.k8s is not None
    k8s = cfg.deployment.k8s
    return K8sPortForwardTarget(namespace=k8s.namespace, service=cfg.name, port=cfg.port)


def _dispatch_up(cfg: FinelogConfig) -> None:
    if cfg.deployment.gcp is not None:
        _gcp.gcp_up(cfg)
    else:
        _k8s.k8s_up(cfg)


def _dispatch_down(cfg: FinelogConfig, *, yes: bool) -> None:
    if cfg.deployment.gcp is not None:
        _gcp.gcp_down(cfg, yes=yes)
    else:
        _k8s.k8s_down(cfg, yes=yes)


def _dispatch_restart(cfg: FinelogConfig) -> None:
    if cfg.deployment.gcp is not None:
        _gcp.gcp_restart(cfg)
    else:
        _k8s.k8s_restart(cfg)


def _dispatch_status(cfg: FinelogConfig) -> None:
    if cfg.deployment.gcp is not None:
        _gcp.gcp_status(cfg)
    else:
        _k8s.k8s_status(cfg)


def _dispatch_logs(cfg: FinelogConfig, *, tail: int, follow: bool) -> None:
    if cfg.deployment.gcp is not None:
        _gcp.gcp_logs(cfg, tail=tail, follow=follow)
    else:
        _k8s.k8s_logs(cfg, tail=tail, follow=follow)


@click.group()
def cli() -> None:
    """Manage finelog deployments."""


@cli.group("deploy")
def deploy() -> None:
    """Provision and manage a finelog deployment from a config file."""


@deploy.command("up")
@click.argument("name")
@click.option(
    "--build/--no-build",
    "build",
    default=True,
    show_default=True,
    help="Build and push the finelog image (using cfg.image as the tag) before provisioning.",
)
def up_cmd(name: str, build: bool) -> None:
    """Provision the finelog deployment described by `<name>` (idempotent)."""
    cfg = load_finelog_config(name)
    if build:
        build_finelog_image(image=cfg.image)
    _dispatch_up(cfg)


@deploy.command("down")
@click.argument("name")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation; for k8s also deletes the PVC.")
def down_cmd(name: str, yes: bool) -> None:
    """Tear down the finelog deployment described by `<name>`."""
    cfg = load_finelog_config(name)
    _dispatch_down(cfg, yes=yes)


@deploy.command("restart")
@click.argument("name")
@click.option(
    "--build/--no-build",
    "build",
    default=True,
    show_default=True,
    help="Build and push the finelog image (using cfg.image as the tag) before restarting.",
)
def restart_cmd(name: str, build: bool) -> None:
    """Restart the finelog deployment in place (refresh the container/image)."""
    cfg = load_finelog_config(name)
    if build:
        build_finelog_image(image=cfg.image)
    _dispatch_restart(cfg)


@deploy.command("status")
@click.argument("name")
def status_cmd(name: str) -> None:
    """Show status of the finelog deployment."""
    cfg = load_finelog_config(name)
    _dispatch_status(cfg)


class OutputFormat(StrEnum):
    TABLE = "table"
    JSON = "json"
    CSV = "csv"


def _print_table(table: pa.Table) -> None:
    """Render an Arrow table as fixed-width columns to stdout."""
    if table.num_rows == 0:
        click.echo(f"(0 rows; columns: {', '.join(table.schema.names)})")
        return
    rows = [[_format_cell(v) for v in row.values()] for row in table.to_pylist()]
    headers = list(table.schema.names)
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = " | "
    click.echo(sep.join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    click.echo("-+-".join("-" * w for w in widths))
    for row in rows:
        click.echo(sep.join(row[i].ljust(widths[i]) for i in range(len(headers))))
    click.echo(f"({table.num_rows} rows)")


def _format_cell(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, bytes):
        return v.hex()
    return str(v)


def _print_json(table: pa.Table) -> None:
    json.dump(table.to_pylist(), sys.stdout, default=str, indent=2)
    sys.stdout.write("\n")


def _print_csv(table: pa.Table) -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(table.schema.names)
    for row in table.to_pylist():
        writer.writerow([_format_cell(v) for v in row.values()])


_PRINTERS = {
    OutputFormat.TABLE: _print_table,
    OutputFormat.JSON: _print_json,
    OutputFormat.CSV: _print_csv,
}


@cli.command("query")
@click.argument("name")
@click.argument("sql")
@click.option(
    "--format",
    "output_format",
    type=click.Choice([f.value for f in OutputFormat]),
    default=OutputFormat.TABLE.value,
    show_default=True,
    help="Output format for the result.",
)
@click.option(
    "--max-rows",
    type=int,
    default=100_000,
    show_default=True,
    help="Reject results larger than this (use LIMIT or raise this cap).",
)
@click.option(
    "--tunnel-timeout",
    type=float,
    default=60.0,
    show_default=True,
    help="Seconds to wait for the local tunnel to become reachable.",
)
def query_cmd(name: str, sql: str, output_format: str, max_rows: int, tunnel_timeout: float) -> None:
    """Run SQL against the deployed finelog `<name>` via a tunnel.

    Opens an IAP tunnel (GCP) or `kubectl port-forward` (k8s) to the
    configured finelog server, runs `<sql>` through `StatsService.Query`,
    and prints results in `--format` (table/json/csv).
    """
    configure_logging(level=logging.INFO)
    cfg = load_finelog_config(name)
    target = _tunnel_target(cfg)
    with open_tunnel(target, timeout=tunnel_timeout) as url:
        client = LogClient.connect(url)
        try:
            table = client.query(sql, max_rows=max_rows)
        finally:
            client.close()
    _PRINTERS[OutputFormat(output_format)](table)


@deploy.command("logs")
@click.argument("name")
@click.option("--tail", type=int, default=200, show_default=True)
@click.option("-f", "--follow", is_flag=True, help="Stream logs")
def logs_cmd(name: str, tail: int, follow: bool) -> None:
    """Tail logs from the finelog deployment."""
    cfg = load_finelog_config(name)
    _dispatch_logs(cfg, tail=tail, follow=follow)


if __name__ == "__main__":
    cli()
