# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``marin-cluster`` — one front door for cluster config, login, and admin.

This umbrella sits above iris and finelog: it owns the top-level cluster config
and login orchestration, and *delegates* the day-to-day client verbs (job,
cluster, logs) to the library CLIs rather than reimplementing them. iris/finelog
are optional installs (``marin-cluster[iris]`` / ``[finelog]``); a verb whose
library is absent prints an install hint instead of failing at import.
"""

import os

import click
from rigging.config_discovery import list_cluster_configs
from rigging.filesystem import MARIN_CLUSTER_CONFIG_DIRS

from marin_cluster import config as cluster_config

_MARIN_CLUSTER_ENV = "MARIN_CLUSTER"


def _resolve_active_cluster(explicit: str | None) -> str | None:
    """Active cluster: ``--cluster`` > ``MARIN_CLUSTER`` env > ``config use`` pointer."""
    return explicit or os.environ.get(_MARIN_CLUSTER_ENV) or cluster_config.current_cluster()


@click.group()
@click.option("--cluster", default=None, help="Cluster name (overrides MARIN_CLUSTER and the pinned cluster).")
@click.pass_context
def main(ctx: click.Context, cluster: str | None) -> None:
    """Administer and connect to a Marin cluster."""
    active = _resolve_active_cluster(cluster)
    ctx.ensure_object(dict)
    ctx.obj["cluster"] = active
    # Export so delegated iris/finelog commands resolve the same cluster.
    if active:
        os.environ[_MARIN_CLUSTER_ENV] = active


@main.group()
def config() -> None:
    """Inspect and select cluster configuration."""


@config.command("list")
def config_list() -> None:
    """List discovered clusters and their manifest paths."""
    configs = list_cluster_configs(MARIN_CLUSTER_CONFIG_DIRS)
    if not configs:
        click.echo("No cluster configs found.")
        return
    pinned = cluster_config.current_cluster()
    for name, path in sorted(configs.items()):
        marker = " *" if name == pinned else ""
        click.echo(f"{name}{marker}\t{path}")


@config.command("use")
@click.argument("name")
def config_use(name: str) -> None:
    """Pin NAME as the current cluster (a secret-free pointer)."""
    cluster_config.set_current_cluster(name)
    click.echo(f"Current cluster set to '{name}'.")


@config.command("show")
@click.pass_context
def config_show(ctx: click.Context) -> None:
    """Show the resolved config for the active cluster."""
    cfg = cluster_config.ClusterConfig.load(ctx.obj.get("cluster"))
    click.echo(f"cluster:       {cfg.name}")
    click.echo(f"dashboard_url: {cfg.dashboard_url or '(none)'}")
    click.echo(f"auth:          {cfg.auth.provider.value}")
    if cfg.auth.iap is not None:
        click.echo(f"  iap.url:     {cfg.auth.iap.url}")
    if cfg.auth.admin_users:
        click.echo(f"  admin_users: {', '.join(cfg.auth.admin_users)}")
    prov = cfg.provisioning
    if prov is None:
        click.echo("provisioning:  (none)")
        return
    if prov.gcp is not None:
        click.echo(f"provisioning.gcp:      project={prov.gcp.project} zone={prov.gcp.default_zone}")
    if prov.iam is not None:
        click.echo(
            "provisioning.iam:      "
            f"controller={prov.iam.controller_service_account} worker={prov.iam.worker_service_account}"
        )
    if prov.iap_gclb is not None:
        click.echo(f"provisioning.iap_gclb: domain={prov.iap_gclb.domain} prefix={prov.iap_gclb.resource_prefix}")


def _mount_delegated(group: click.Group) -> None:
    """Mount the iris/finelog client verbs when their libraries are installed.

    Imports are guarded so an install without ``[iris]``/``[finelog]`` still loads
    the umbrella; the missing verbs simply don't appear.
    """
    try:
        from iris.cli.cluster import cluster as iris_cluster  # noqa: PLC0415
        from iris.cli.job import job as iris_job  # noqa: PLC0415

        group.add_command(iris_job, name="job")
        group.add_command(iris_cluster, name="cluster")
    except ImportError:
        pass

    try:
        from finelog.deploy.cli import cli as finelog_cli  # noqa: PLC0415

        group.add_command(finelog_cli, name="logs")
    except ImportError:
        pass


_mount_delegated(main)


if __name__ == "__main__":
    main()
