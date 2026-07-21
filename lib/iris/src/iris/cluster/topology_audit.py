# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit an admitted gang's topology using Iris cluster-status JSON."""

import json
from collections import Counter
from enum import StrEnum
from typing import Any, TextIO

import click


class TopologyExpectation(StrEnum):
    SINGLE_DOMAIN = "single-domain"
    BALANCED = "balanced"


def audit_topology(
    status: dict[str, Any],
    *,
    task_prefix: str,
    topology_key: str,
    expectation: TopologyExpectation,
    expected_domains: int | None,
) -> tuple[dict[str, Any], int]:
    """Return a machine-readable topology verdict and process exit code."""
    pods = [
        pod
        for pod in status.get("pod_statuses", [])
        if pod.get("node_name") and str(pod.get("task_id", "")).startswith(f"{task_prefix}.")
    ]
    if not pods:
        return {
            "status": "INCOMPLETE",
            "reason": "no admitted pods match the task prefix",
            "pod_count": 0,
            "node_count": 0,
        }, 3

    nodes = {node.get("name"): node for node in status.get("nodes", [])}
    node_names = {pod["node_name"] for pod in pods}
    missing = 0
    for node_name in node_names:
        topology_labels = nodes.get(node_name, {}).get("topology_labels", {})
        domain = topology_labels.get(topology_key)
        if not domain:
            missing += 1

    if missing:
        return {
            "status": "UNOBSERVABLE",
            "reason": f"topology key {topology_key!r} is absent for {missing} admitted node(s)",
            "pod_count": len(pods),
            "node_count": len(node_names),
        }, 2

    domain_by_node = {node_name: str(nodes[node_name]["topology_labels"][topology_key]) for node_name in node_names}
    counts = Counter(domain_by_node[pod["node_name"]] for pod in pods)
    matches = len(counts) == 1
    if expectation is TopologyExpectation.BALANCED:
        if expected_domains is None:
            raise click.UsageError("--expected-domains is required for expectation=balanced")
        sizes = counts.values()
        matches = len(counts) == expected_domains and max(sizes) - min(sizes) <= 1

    result = {
        "status": "MATCH" if matches else "VIOLATION",
        "expectation": expectation.value,
        "pod_count": len(pods),
        "node_count": len(node_names),
        "domain_counts": dict(sorted(counts.items())),
    }
    return result, 0 if matches else 1


@click.command()
@click.option("--task-prefix", required=True, help="Task ID prefix for the admitted gang.")
@click.option("--topology-key", required=True, help="Logical topology label to inspect.")
@click.option(
    "--expectation",
    required=True,
    type=click.Choice([expectation.value for expectation in TopologyExpectation]),
)
@click.option("--expected-domains", type=click.IntRange(min=1))
@click.argument("status_file", type=click.File("r"), default="-")
def main(
    task_prefix: str,
    topology_key: str,
    expectation: str,
    expected_domains: int | None,
    status_file: TextIO,
) -> None:
    """Audit JSON from Iris GetKubernetesClusterStatus; use '-' for stdin."""
    result, exit_code = audit_topology(
        json.load(status_file),
        task_prefix=task_prefix,
        topology_key=topology_key,
        expectation=TopologyExpectation(expectation),
        expected_domains=expected_domains,
    )
    click.echo(json.dumps(result, sort_keys=True))
    raise click.exceptions.Exit(exit_code)


if __name__ == "__main__":
    main()
