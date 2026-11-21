"""Select one Iris submission route for training, conversion, and publication."""

from collections.abc import Iterator
from contextlib import contextmanager
import os
from pathlib import Path

from iris.cli.connect import IRIS_CLUSTER_CONFIG_DIRS, open_iris_client
from iris.client.client import IrisClient
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, Constraint, ConstraintOp
from rigging.config_discovery import resolve_cluster_config


@contextmanager
def submission_client(
    cluster_config: str,
    target_cluster: str | None,
    parent_cluster_config: str | None,
    workspace: Path,
) -> Iterator[tuple[IrisClient, Constraint | None]]:
    """Use the ambient controller for children, or the correct root/federated route."""
    if controller_url := os.environ.get("IRIS_CONTROLLER_URL"):
        with IrisClient.in_cluster(controller_url, workspace=workspace) as client:
            yield client, None
        return

    cluster_pin = None
    submit_path = Path(cluster_config)
    if target_cluster:
        cluster_pin = Constraint.create(key=CLUSTER_CONSTRAINT_KEY, op=ConstraintOp.EQ, value=target_cluster)
        if parent_cluster_config:
            submit_path = Path(parent_cluster_config)
        else:
            sibling = submit_path.with_name("marin.yaml")
            submit_path = sibling if sibling.exists() else resolve_cluster_config("marin", dirs=IRIS_CLUSTER_CONFIG_DIRS)
    with open_iris_client(config_file=submit_path, workspace=workspace) as client:
        yield client, cluster_pin
