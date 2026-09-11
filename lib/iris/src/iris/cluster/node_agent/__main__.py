# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for Iris node telemetry."""

import logging
import signal
import threading
from pathlib import Path

import click
from rigging.log_setup import configure_logging

from iris.cluster.node_agent import gcp, kubernetes
from iris.cluster.runtime.env import IRIS_NAMESPACE_ENV, IRIS_NODE_NAME_ENV


def _stop_event() -> threading.Event:
    stop = threading.Event()

    def handle_signal(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    return stop


@click.group()
def cli() -> None:
    """Run Iris physical node telemetry."""


@cli.command("gcp")
@click.option("--worker-config", type=click.Path(path_type=Path, exists=True), required=True)
def gcp_command(worker_config: Path) -> None:
    """Run beside an Iris worker on a GCP VM."""
    configure_logging(level=logging.INFO)
    gcp.run(worker_config, _stop_event())


@cli.command("k8s")
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--node-name", envvar=IRIS_NODE_NAME_ENV, required=True)
@click.option("--namespace", envvar=IRIS_NAMESPACE_ENV, required=True)
def k8s_command(config_path: Path, node_name: str, namespace: str) -> None:
    """Run once per Kubernetes node."""
    configure_logging(level=logging.INFO)
    kubernetes.run(config_path, node_name, namespace, _stop_event())


if __name__ == "__main__":
    cli()
