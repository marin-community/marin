# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Carry verified machine routes into exported task specifications."""

import json
from typing import Any

from shellbox.machine import Backend

from taskcompendium.models import TaskSpec
from taskcompendium.runtime.requirements import ROUTE_BACKENDS_LABEL


def export_machine_route(admitted: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Merge routing metadata after admission checks the original task identity."""
    # Routing metadata rides the row's labels past the oracle-filter integrity check,
    # then joins the task's own labels so every downstream reader sees it.
    labels = dict(row.get("labels") or ())
    route = labels.pop(ROUTE_BACKENDS_LABEL, None)
    if route is not None:
        task = TaskSpec.model_validate_json(admitted["task_json"])
        admitted = {
            **admitted,
            "task_json": (
                task.model_copy(update={"labels": {**task.labels, ROUTE_BACKENDS_LABEL: route}}).model_dump_json()
            ),
        }
    return admitted


def attach_machine_route(row: dict[str, Any], route: tuple[Backend, ...]) -> dict[str, Any]:
    """Retain the original task identity while attaching verified routing metadata."""
    if not route:
        return row
    return {
        **row,
        "labels": {**dict(row.get("labels") or ()), ROUTE_BACKENDS_LABEL: json.dumps(route, separators=(",", ":"))},
    }
