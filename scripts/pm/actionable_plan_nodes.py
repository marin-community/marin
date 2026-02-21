#!/usr/bin/env -S uv run
# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["PyYAML>=6.0"]
# ///
"""List actionable nodes from `.agents/the_plan.yaml`.

Actionable means:
- status is `planned` or `active`
- all dependencies exist and are `done`

Usage:
  ./scripts/pm/actionable_plan_nodes.py
  ./scripts/pm/actionable_plan_nodes.py --owner dlwh
  ./scripts/pm/actionable_plan_nodes.py --upstream-of milestone_2026_05_100b_moe
  ./scripts/pm/actionable_plan_nodes.py --json
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PLAN_PATH = Path(".agents/the_plan.yaml")
READY_STATUSES = {"planned", "active"}


def _parse_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _owner_set(node: dict[str, Any]) -> set[str]:
    owners = node.get("owners") or []
    owner_names = node.get("owner_names") or []
    merged: set[str] = set()
    for owner in owners:
        if isinstance(owner, str):
            merged.add(owner.lower())
    for owner in owner_names:
        if isinstance(owner, str):
            merged.add(owner.lower())
    return merged


def _is_done(node: dict[str, Any] | None) -> bool:
    if not isinstance(node, dict):
        return False
    return node.get("status") == "done"


def _split_csv_args(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if token:
                out.append(token)
    return out


def _upstream_closure(nodes: dict[str, dict[str, Any]], targets: list[str]) -> set[str]:
    """Return transitive dependency closure of targets (ancestors only)."""
    missing = sorted([target for target in targets if target not in nodes])
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Unknown target node(s): {missing_str}")

    keep: set[str] = set()
    stack: list[str] = list(targets)
    visited: set[str] = set()
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        deps = nodes[cur].get("dependencies") or []
        for dep in deps:
            if isinstance(dep, str) and dep in nodes and dep not in keep:
                keep.add(dep)
                stack.append(dep)
    return keep


def _is_actionable(
    *,
    node_id: str,
    node: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    include_milestones: bool,
    owner_filters: set[str],
    upstream_filter: set[str] | None,
) -> bool:
    status = node.get("status")
    if status not in READY_STATUSES:
        return False

    if not include_milestones and node.get("type") == "milestone":
        return False

    if upstream_filter is not None and node_id not in upstream_filter:
        return False

    if owner_filters:
        if _owner_set(node).isdisjoint(owner_filters):
            return False

    for dep in node.get("dependencies") or []:
        if not isinstance(dep, str):
            return False
        if dep not in nodes:
            return False
        if not _is_done(nodes[dep]):
            return False

    # Ignore malformed nodes with no meaningful payload.
    if not node_id or not isinstance(node.get("title"), str):
        return False

    return True


def _load_nodes(path: Path) -> dict[str, dict[str, Any]]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Expected top-level mapping, got: {type(raw).__name__}")

    nodes: dict[str, dict[str, Any]] = {}
    for node_id, payload in raw.items():
        if node_id == "meta":
            continue
        if payload is None:
            payload = {}
        if isinstance(payload, dict):
            nodes[node_id] = payload
    return nodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH, help="Path to the plan YAML")
    parser.add_argument(
        "--owner",
        action="append",
        default=[],
        help="Filter by owner handle/name (repeatable, case-insensitive)",
    )
    parser.add_argument(
        "--upstream-of",
        action="append",
        default=[],
        help=("Only show actionable nodes upstream of these target node IDs " "(repeatable or comma-separated)"),
    )
    parser.add_argument(
        "--include-milestones",
        action="store_true",
        help="Include milestone nodes in results (default: excluded)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max number of rows to print (0 = no limit)")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of table text")
    args = parser.parse_args()

    nodes = _load_nodes(args.plan)
    owner_filters = {owner.lower() for owner in args.owner}
    upstream_targets = _split_csv_args(args.upstream_of)
    upstream_filter: set[str] | None = None
    if upstream_targets:
        upstream_filter = _upstream_closure(nodes, upstream_targets)

    actionable: list[dict[str, Any]] = []
    for node_id, node in nodes.items():
        if not _is_actionable(
            node_id=node_id,
            node=node,
            nodes=nodes,
            include_milestones=args.include_milestones,
            owner_filters=owner_filters,
            upstream_filter=upstream_filter,
        ):
            continue

        actionable.append(
            {
                "id": node_id,
                "title": node.get("title"),
                "type": node.get("type"),
                "status": node.get("status"),
                "target_date": node.get("target_date"),
                "issue": node.get("issue"),
                "owners": node.get("owners"),
                "owner_names": node.get("owner_names"),
                "dependencies": node.get("dependencies") or [],
            }
        )

    actionable.sort(
        key=lambda node: (
            _parse_date(node.get("target_date")) is None,
            _parse_date(node.get("target_date")) or date.max,
            node.get("id") or "",
        )
    )

    if args.limit > 0:
        actionable = actionable[: args.limit]

    if args.json:
        print(json.dumps(actionable, indent=2, sort_keys=False))
        return

    if not actionable:
        print("No actionable nodes found.")
        return

    for node in actionable:
        owners = node.get("owners") or node.get("owner_names") or []
        owners_text = ",".join(owners) if owners else "-"
        issue = node.get("issue")
        issue_text = str(issue) if issue is not None else "-"
        target = node.get("target_date") or "-"
        print(
            f"{target}\t{node['id']}\t{node.get('type')}\towners={owners_text}\tissue={issue_text}\t{node.get('title')}"
        )

    print(f"\ncount={len(actionable)}")


if __name__ == "__main__":
    main()
