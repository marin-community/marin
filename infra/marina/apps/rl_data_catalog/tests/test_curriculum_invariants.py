# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Release API invariants catch cross-source and display regressions without a browser matrix."""

from copy import deepcopy

import pytest

from infra.marina.apps.rl_data_catalog.release_curriculum import check_curriculum_document, count_projection


def measured_document():
    nodes = [
        dict(subject_id="D01", node_id="subject:D01", parent_id=None, kind="subject", name="Mathematics"),
        dict(subject_id="D01", node_id="add", parent_id="subject:D01", kind="skill", name="Addition"),
        dict(subject_id="D01", node_id="empty", parent_id="subject:D01", kind="skill", name="Empty skill"),
    ]
    values = {
        "train": dict(
            denominator=[3, 3], assigned=[2, 1], subject_only=[1, 1], out_of_scope=[0, 1], root=[3, 2], add=[2, 1]
        ),
        "eval": dict(
            denominator=[2, 2], assigned=[0, 0], subject_only=[1, 0], out_of_scope=[1, 2], root=[1, 0], add=[0, 0]
        ),
        "all": dict(
            denominator=[5, 5], assigned=[2, 1], subject_only=[2, 1], out_of_scope=[1, 3], root=[4, 2], add=[2, 1]
        ),
    }

    def statistics(value):
        return [
            [metric, "", "", value[metric]] for metric in ("denominator", "assigned", "subject_only", "out_of_scope")
        ] + [
            ["insufficient_context", "", "", [0, 0]],
            ["skipped", "", "", [0, 0]],
            ["placement", "D01", "subject:D01", value["root"]],
            ["placement", "D01", "add", value["add"]],
            ["placement", "D01", "empty", [0, 0]],
        ]

    views = {}
    for use, value in values.items():
        rows = statistics(value)
        views[use] = dict(
            release_id="measured",
            hierarchy=deepcopy(nodes),
            totals={metric: vector[1] for metric, subject, node, vector in rows if not subject},
            area_counts=[
                dict(subject_id=subject, node_id=node, count=vector[1])
                for metric, subject, node, vector in rows
                if subject
            ],
            confidence=dict(
                levels=[
                    dict(label="Broad", subject_min_similarity=0.3, capability_min_similarity=0.3),
                    dict(label="Original", subject_min_similarity=0.55, capability_min_similarity=0.75),
                ],
                statistics=[
                    dict(metric=metric, subject_id=subject, node_id=node, instances=vector)
                    for metric, subject, node, vector in rows
                ],
            ),
            thresholds=dict(subject_distance=0.45, capability_distance=0.25),
        )
    return dict(
        release_id="measured",
        taxonomy=nodes,
        views=views,
        sources=[
            dict(
                source_id=use,
                alignment=dict(basis="matched", intended_use=use, task_count=value["denominator"][0]),
                statistics=[[metric, subject, node, vector[1]] for metric, subject, node, vector in statistics(value)],
                confidence_statistics=statistics(value),
            )
            for use, value in values.items()
            if use != "all"
        ],
    )


def test_every_use_and_confidence_level_reconciles_sources_empty_nodes_and_remainders():
    document = measured_document()
    checked = check_curriculum_document(document)
    assert [(row["use"], row["level"], row["denominator"]) for row in checked["cases"]] == [
        ("all", 0, 5),
        ("all", 1, 5),
        ("train", 0, 3),
        ("train", 1, 3),
        ("eval", 0, 2),
        ("eval", 1, 2),
    ]
    assert all(row["domains"] == 1 and row["nodes"] == 3 for row in checked["cases"])
    broad = count_projection(document["views"]["all"], 0)
    strict = count_projection(document["views"]["all"], 1)
    assert broad.totals["assigned"] == 2 and strict.totals["assigned"] == 1
    assert broad.areas[("D01", "empty")] == strict.areas[("D01", "empty")] == 0


@pytest.mark.parametrize(
    "regression", ["source_breakdown", "empty_node", "subject_label", "displayed_count", "outcome_partition"]
)
def test_api_gate_rejects_count_and_taxonomy_regressions(regression):
    document = measured_document()
    data = document["views"]["all"]
    if regression == "source_breakdown":
        next(row for row in data["confidence"]["statistics"] if row["node_id"] == "subject:D01")["instances"][0] = 5
        message = "source breakdown differs"
    elif regression == "empty_node":
        data["hierarchy"] = [row for row in data["hierarchy"] if row["node_id"] != "empty"]
        message = "taxonomy inventory differs"
    elif regression == "subject_label":
        next(row for row in data["hierarchy"] if row["node_id"] == "add")["name"] = "Mathematics"
        message = "capability label equals"
    elif regression == "outcome_partition":
        next(row for row in data["confidence"]["statistics"] if row["metric"] == "assigned")["instances"][0] += 1
        message = "outcomes do not partition"
    else:
        data["totals"]["assigned"] = 2
        message = "displayed release counts differ"
    with pytest.raises(ValueError, match=message):
        check_curriculum_document(document)
