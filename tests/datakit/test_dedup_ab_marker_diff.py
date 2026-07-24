# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.datakit.scripts.dedup_ab_marker_diff import marker_differences


def _marker(cluster_id: str, canonical: bool) -> dict:
    return {
        "dup_cluster_id": cluster_id,
        "is_cluster_canonical": canonical,
    }


def test_marker_differences_reports_every_presence_and_attribute_change() -> None:
    capped = {
        "capped-only": _marker("a", False),
        "cluster-changed": _marker("a", False),
        "canonical-changed": _marker("b", False),
        "same": _marker("c", True),
    }
    converged = {
        "cluster-changed": _marker("d", False),
        "canonical-changed": _marker("b", True),
        "converged-only": _marker("e", False),
        "same": _marker("c", True),
    }

    differences = list(marker_differences(capped, converged))

    assert [record["id"] for record in differences] == [
        "canonical-changed",
        "capped-only",
        "cluster-changed",
        "converged-only",
    ]
    assert [record["change_kind"] for record in differences] == [
        "attributes_changed",
        "capped_only",
        "attributes_changed",
        "converged_only",
    ]
    assert differences[0]["capped_cluster_id"] == differences[0]["converged_cluster_id"] == "b"
    assert differences[0]["capped_is_canonical"] is False
    assert differences[0]["converged_is_canonical"] is True
    assert differences[1]["converged_cluster_id"] is None
    assert differences[3]["capped_cluster_id"] is None
