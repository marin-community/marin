# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from click.testing import CliRunner
from iris.cluster.topology_audit import main


def _status(*, labels: tuple[dict[str, str] | None, ...]) -> dict:
    pods = []
    nodes = []
    for index, topology_labels in enumerate(labels):
        node_name = f"node-{index}"
        pods.append(
            {
                "task_id": f"user.gang.{index}",
                "phase": "Running",
                "node_name": node_name,
            }
        )
        node = {"name": node_name}
        if topology_labels is not None:
            node["topology_labels"] = topology_labels
        nodes.append(node)
    return {"pod_statuses": pods, "nodes": nodes}


def _invoke(status: dict, *args: str):
    return CliRunner().invoke(
        main,
        ["--task-prefix", "user.gang", "--topology-key", "nvlink.domain", *args, "-"],
        input=json.dumps(status),
    )


def test_audit_reports_unobservable_when_iris_omits_topology_labels() -> None:
    result = _invoke(_status(labels=(None, None)), "--expectation", "single-domain")

    assert result.exit_code == 2
    assert json.loads(result.output) == {
        "status": "UNOBSERVABLE",
        "reason": "topology key 'nvlink.domain' is absent for 2 admitted node(s)",
        "pod_count": 2,
        "node_count": 2,
    }


def test_audit_accepts_single_domain_placement() -> None:
    result = _invoke(
        _status(labels=({"nvlink.domain": "domain-a"}, {"nvlink.domain": "domain-a"})),
        "--expectation",
        "single-domain",
    )

    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "status": "MATCH",
        "expectation": "single-domain",
        "pod_count": 2,
        "node_count": 2,
        "domain_counts": {"domain-a": 2},
    }


def test_audit_rejects_single_domain_skew() -> None:
    result = _invoke(
        _status(labels=({"nvlink.domain": "domain-a"}, {"nvlink.domain": "domain-b"})),
        "--expectation",
        "single-domain",
    )

    assert result.exit_code == 1
    assert json.loads(result.output) == {
        "status": "VIOLATION",
        "expectation": "single-domain",
        "pod_count": 2,
        "node_count": 2,
        "domain_counts": {"domain-a": 1, "domain-b": 1},
    }


def test_audit_accepts_balanced_domain_placement() -> None:
    result = _invoke(
        _status(
            labels=(
                {"nvlink.domain": "domain-a"},
                {"nvlink.domain": "domain-a"},
                {"nvlink.domain": "domain-b"},
            )
        ),
        "--expectation",
        "balanced",
        "--expected-domains",
        "2",
    )

    assert result.exit_code == 0
    assert json.loads(result.output)["status"] == "MATCH"
