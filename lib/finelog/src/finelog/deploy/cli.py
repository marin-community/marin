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

import click

from finelog.deploy import _gcp, _k8s
from finelog.deploy.config import FinelogConfig, load_finelog_config


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
def up_cmd(name: str) -> None:
    """Provision the finelog deployment described by `<name>` (idempotent)."""
    cfg = load_finelog_config(name)
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
def restart_cmd(name: str) -> None:
    """Restart the finelog deployment in place (refresh the container/image)."""
    cfg = load_finelog_config(name)
    _dispatch_restart(cfg)


@deploy.command("status")
@click.argument("name")
def status_cmd(name: str) -> None:
    """Show status of the finelog deployment."""
    cfg = load_finelog_config(name)
    _dispatch_status(cfg)


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
