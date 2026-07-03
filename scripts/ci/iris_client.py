#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared Iris client helpers for CI scripts."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import click
from iris.cli.main import client_credentials, resolve_cluster_name
from iris.client import IrisClient
from iris.cluster.composer import provider_bundle
from iris.cluster.config import load_config
from iris.cluster.local_cluster import LocalCluster
from rigging.cluster_manifest import AuthProvider, ClusterAuth
from rigging.credential_store import cluster_name_from_url
from rigging.credentials import credentials_for


@contextmanager
def open_iris_client(
    *,
    iris_config: Path | None,
    repo_root: Path,
    controller_url: str | None = None,
) -> Iterator[IrisClient]:
    if controller_url is not None:
        credentials = credentials_for(cluster_name_from_url(controller_url), ClusterAuth(AuthProvider.NONE))
        with IrisClient.remote(controller_url, workspace=repo_root, credentials=credentials) as client:
            yield client
        return

    if iris_config is None:
        raise click.ClickException("No controller specified. Pass --iris-config or --controller-url.")

    config = load_config(iris_config)
    cluster_name = resolve_cluster_name(config, None, None)
    credentials = client_credentials(config, cluster_name)

    if config.controller.controller_kind() == "local":
        cluster = LocalCluster(config)
        try:
            with IrisClient.remote(cluster.start(), workspace=repo_root, credentials=credentials) as client:
                yield client
        finally:
            cluster.close()
        return

    bundle = provider_bundle(config)
    controller_address = config.controller_address() or bundle.controller.discover_controller(config.controller)
    with bundle.controller.tunnel(address=controller_address) as tunnel_url:
        with IrisClient.remote(tunnel_url, workspace=repo_root, credentials=credentials) as client:
            yield client
