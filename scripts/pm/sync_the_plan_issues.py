#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate `.agents/the_plan.yaml` and sync nodes to GitHub issues.

This script has two responsibilities:

1) Validate the plan file:
   - schema validation (required/allowed fields, basic type checks)
   - dependency existence
   - DAG acyclicity (topological sort)

2) Sync plan nodes to GitHub issues:
   - create missing issues (default: epics only)
   - update issues when safe
   - detect text conflicts and write a reconciliation file for manual resolution
   - run a second pass to replace `#dag_id` references with `#<issue_number>` references

Usage (safe checks only):
  .venv/bin/python scripts/pm/sync_the_plan_issues.py check

Usage (dry-run sync):
  .venv/bin/python scripts/pm/sync_the_plan_issues.py sync --dry-run --token-env GITHUB_TOKEN

Usage (apply sync):
  .venv/bin/python scripts/pm/sync_the_plan_issues.py sync --apply --update-plan

Requires for sync:
  - `GITHUB_TOKEN` env var (or another env var via --token-env)
  - PyGithub installed in the environment
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml

try:
    from github import Github
    from github.Issue import Issue
    from github.Repository import Repository
except Exception:  # pragma: no cover
    Github = None  # type: ignore[assignment]
    Issue = None  # type: ignore[assignment]
    Repository = None  # type: ignore[assignment]


PLAN_PATH_DEFAULT = Path(".agents/the_plan.yaml")
RECONCILIATION_DEFAULT = Path(".agents/issue_reconciliation.yaml")

PlanType = Literal["epic", "task", "experiment", "milestone"]
PlanStatus = str

ALLOWED_NODE_KEYS: set[str] = {
    "issue",
    "candidate_id",
    "candidate_score",
    "title",
    "type",
    "status",
    "owners",
    "owner_names",
    "target_date",
    "labels",
    "description",
    "dependencies",
    "definition_of_done",
}


@dataclass(frozen=True)
class PlanNode:
    node_id: str
    issue: int | str | None
    candidate_id: int | None
    candidate_score: float | None
    title: str | None
    type: PlanType | None
    status: PlanStatus | None
    owners: list[str] | None
    owner_names: list[str] | None
    target_date: str | None
    labels: list[str] | None
    description: str | None
    dependencies: list[str] | None
    definition_of_done: str | None

    @property
    def is_issue_tracked(self) -> bool:
        return self.issue is not None

    def plan_text(self) -> str:
        parts: list[str] = []
        if self.description:
            parts.append(self.description.strip())
        if self.definition_of_done:
            parts.append(self.definition_of_done.strip())
        return "\n\n".join([p for p in parts if p]).strip()


def _as_str_list(value: Any, *, field: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ValueError(f"{field} must be list[str] or null, got: {value!r}")
    return value


def _as_optional_str(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be str or null, got: {value!r}")
    return value


def _as_optional_int(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"{field} must be int or null, got: {value!r}")
    return value


def _as_optional_float(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be float or null, got: {value!r}")
    return float(value)


def load_plan(path: Path) -> tuple[dict[str, Any], dict[str, PlanNode]]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping at top-level, got: {type(raw).__name__}")

    nodes: dict[str, PlanNode] = {}
    for node_id, payload in raw.items():
        if node_id == "meta":
            continue
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise ValueError(f"Node {node_id!r} must be a mapping, got: {type(payload).__name__}")

        unknown = set(payload.keys()) - ALLOWED_NODE_KEYS
        if unknown:
            raise ValueError(f"Node {node_id!r} has unknown keys: {sorted(unknown)}")

        issue = payload.get("issue", None)
        if issue is not None and not isinstance(issue, (int, str)):
            raise ValueError(f"Node {node_id!r} field issue must be int|str|null, got: {issue!r}")

        node_type = payload.get("type", None)
        if node_type is not None and node_type not in ("epic", "task", "experiment", "milestone"):
            raise ValueError(f"Node {node_id!r} field type invalid: {node_type!r}")

        status = payload.get("status", None)
        if status is not None and not isinstance(status, str):
            raise ValueError(f"Node {node_id!r} field status must be str|null, got: {status!r}")

        nodes[node_id] = PlanNode(
            node_id=node_id,
            issue=issue,
            candidate_id=_as_optional_int(payload.get("candidate_id"), field=f"{node_id}.candidate_id"),
            candidate_score=_as_optional_float(payload.get("candidate_score"), field=f"{node_id}.candidate_score"),
            title=_as_optional_str(payload.get("title"), field=f"{node_id}.title"),
            type=node_type,
            status=status,
            owners=_as_str_list(payload.get("owners"), field=f"{node_id}.owners"),
            owner_names=_as_str_list(payload.get("owner_names"), field=f"{node_id}.owner_names"),
            target_date=_as_optional_str(payload.get("target_date"), field=f"{node_id}.target_date"),
            labels=_as_str_list(payload.get("labels"), field=f"{node_id}.labels"),
            description=_as_optional_str(payload.get("description"), field=f"{node_id}.description"),
            dependencies=_as_str_list(payload.get("dependencies"), field=f"{node_id}.dependencies"),
            definition_of_done=_as_optional_str(
                payload.get("definition_of_done"), field=f"{node_id}.definition_of_done"
            ),
        )

    return raw, nodes


def validate_dependencies(nodes: dict[str, PlanNode]) -> None:
    node_ids = set(nodes.keys())
    missing: dict[str, list[str]] = {}
    for node in nodes.values():
        for dep in node.dependencies or []:
            if dep not in node_ids:
                missing.setdefault(node.node_id, []).append(dep)
    if missing:
        formatted = "\n".join(f"{k}: {v}" for k, v in sorted(missing.items()))
        raise ValueError(f"Missing dependencies:\n{formatted}")


def topo_sort(nodes: dict[str, PlanNode]) -> list[str]:
    """Topologically sort node IDs. Raises ValueError if a cycle is detected."""
    indegree: dict[str, int] = {k: 0 for k in nodes}
    outgoing: dict[str, list[str]] = {k: [] for k in nodes}
    for node in nodes.values():
        for dep in node.dependencies or []:
            outgoing[dep].append(node.node_id)
            indegree[node.node_id] += 1

    q = deque([k for k, d in indegree.items() if d == 0])
    ordered: list[str] = []
    while q:
        cur = q.popleft()
        ordered.append(cur)
        for nxt in outgoing[cur]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                q.append(nxt)

    if len(ordered) != len(nodes):
        stuck = sorted([k for k, d in indegree.items() if d > 0])
        raise ValueError(f"Cycle detected or disconnected subgraph; remaining nodes: {stuck[:50]}")

    return ordered


_ANCHOR_RE = re.compile(r"#([a-z0-9_]{2,})\b")


def replace_dag_anchors(text: str, *, id_to_issue: dict[str, int]) -> str:
    """Replace `#dag_id` anchors with `#<issue>` when mapping exists.

    This intentionally does *not* touch:
    - headings (`# Title` has a space)
    - numeric issue references (`#123`)
    """

    def repl(m: re.Match[str]) -> str:
        token = m.group(1)
        if token.isdigit():
            return m.group(0)
        issue = id_to_issue.get(token)
        if issue is None:
            return m.group(0)
        return f"#{issue}"

    return _ANCHOR_RE.sub(repl, text)


def check_anchors(nodes: dict[str, PlanNode]) -> None:
    node_ids = set(nodes.keys())
    problems: dict[str, list[str]] = defaultdict(list)
    for node in nodes.values():
        text = "\n\n".join([node.description or "", node.definition_of_done or ""])
        for m in _ANCHOR_RE.finditer(text):
            token = m.group(1)
            if token.isdigit():
                continue
            if token not in node_ids:
                continue
            # ok: references another node by dag id
        # Nothing to enforce yet; placeholder for future stricter checks.
    if problems:
        formatted = "\n".join(f"{k}: {v}" for k, v in sorted(problems.items()))
        raise ValueError(f"Anchor problems:\n{formatted}")


def plan_check(path: Path) -> None:
    _raw, nodes = load_plan(path)
    validate_dependencies(nodes)
    topo_sort(nodes)
    check_anchors(nodes)


def _require_pygithub() -> None:
    if Github is None:
        raise RuntimeError(
            "PyGithub is required for sync mode. Install it (e.g. `pip install PyGithub`) "
            "or use an environment where it is already available."
        )


def _gh_repo(gh: Github, repo_name: str) -> Repository:
    repo = gh.get_repo(repo_name)
    return repo


def _node_to_issue_body(node: PlanNode, *, id_to_issue: dict[str, int]) -> str:
    header = f"<!-- managed-by: the_plan.yaml id={node.node_id} -->"
    lines: list[str] = [header]

    if node.description:
        lines.append(node.description.rstrip())
        lines.append("")

    lines.append("**Plan Metadata**")
    if node.type:
        lines.append(f"- Type: `{node.type}`")
    if node.status:
        lines.append(f"- Status: `{node.status}`")
    if node.target_date:
        lines.append(f"- Target date: `{node.target_date}`")
    if node.owners:
        owners = " ".join([f"@{o.lstrip('@')}" for o in node.owners])
        lines.append(f"- Owners: {owners}")
    if node.labels:
        lines.append(f"- Labels: {', '.join([f'`{label}`' for label in node.labels])}")

    if node.dependencies:
        lines.append("")
        lines.append("**Dependencies**")
        for dep in node.dependencies:
            dep_issue = id_to_issue.get(dep)
            if dep_issue is not None:
                lines.append(f"- #{dep_issue} (`{dep}`)")
            else:
                lines.append(f"- #{dep} (`{dep}`)")

    if node.definition_of_done:
        lines.append("")
        lines.append("**Definition of Done**")
        lines.append(node.definition_of_done.rstrip())

    body = "\n".join(lines).strip() + "\n"
    return replace_dag_anchors(body, id_to_issue=id_to_issue)


def _issue_title(node: PlanNode) -> str:
    if node.title:
        return node.title.strip()
    return node.node_id


def _load_repo_labels(repo: Repository) -> set[str]:
    return {label.name for label in repo.get_labels()}


def _sync_labels(issue: Issue, *, desired: list[str], existing_labels: set[str], apply: bool) -> None:
    labels = [label for label in desired if label in existing_labels]
    if not apply:
        return
    issue.set_labels(*labels)


def _sync_assignees(issue: Issue, *, owners: list[str] | None, apply: bool) -> None:
    if not owners:
        return
    assignees = [o.lstrip("@") for o in owners]
    if not apply:
        return
    issue.edit(assignees=assignees)


@dataclass
class ReconciliationItem:
    node_id: str
    issue_number: int
    title: str
    plan_body: str
    github_body: str


def _write_reconciliation(path: Path, items: list[ReconciliationItem]) -> None:
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "items": [
            {
                "node_id": it.node_id,
                "issue_number": it.issue_number,
                "title": it.title,
                "plan_body": it.plan_body,
                "github_body": it.github_body,
            }
            for it in items
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _update_plan_issue_numbers(
    raw_plan: dict[str, Any], *, created: dict[str, int], plan_path: Path, apply: bool
) -> None:
    if not created:
        return
    if not apply:
        created_json = json.dumps(created, indent=2, sort_keys=True)
        print(f"[dry-run] Would update {plan_path} with issue numbers:\n{created_json}")
        return

    # We intentionally patch the file text in-place (not yaml.safe_dump), to preserve comments and formatting.
    text = plan_path.read_text()

    def find_block_span(node_id: str) -> tuple[int, int] | None:
        # Block starts at a top-level key line and ends before the next top-level key line (or EOF).
        m = re.search(rf"(?m)^{re.escape(node_id)}:\s*$", text)
        if m is None:
            return None
        start = m.start()
        next_m = re.search(r"(?m)^[a-zA-Z0-9_]+:\s*$", text[m.end() :])
        end = len(text) if next_m is None else m.end() + next_m.start()
        return start, end

    updated = False
    for node_id, issue_num in sorted(created.items()):
        span = find_block_span(node_id)
        if span is None:
            continue
        start, end = span
        block = text[start:end]

        issue_line_re = re.compile(r"(?m)^  issue:\s*(.*)\s*$")
        m_issue = issue_line_re.search(block)
        if m_issue:
            current = (m_issue.group(1) or "").strip()
            if current in ("", "null", "None"):
                block = issue_line_re.sub(f"  issue: {issue_num}", block, count=1)
                updated = True
        else:
            # Insert after `title:` if present, else right after header line.
            insert_re = re.compile(r"(?m)^  title:.*\n")
            m_title = insert_re.search(block)
            if m_title:
                insert_at = m_title.end()
            else:
                # After the node header line itself.
                header_end = block.find("\n") + 1
                insert_at = header_end if header_end > 0 else len(block)
            block = block[:insert_at] + f"  issue: {issue_num}\n" + block[insert_at:]
            updated = True

        text = text[:start] + block + text[end:]

    if updated:
        plan_path.write_text(text)


def sync_plan_to_issues(
    *,
    plan_path: Path,
    repo_name: str,
    create_mode: Literal["none", "epics", "all"],
    reconciliation_path: Path,
    apply: bool,
    update_plan: bool,
    token_env: str,
) -> None:
    _require_pygithub()

    token = os.environ.get(token_env)
    if not token:
        raise RuntimeError(f"Missing auth token: environment variable {token_env!r} is not set")

    raw_plan, nodes = load_plan(plan_path)
    validate_dependencies(nodes)
    order = topo_sort(nodes)

    gh = Github(token)
    repo = _gh_repo(gh, repo_name)
    repo_labels = _load_repo_labels(repo)

    # Build initial id->issue mapping from plan file.
    id_to_issue: dict[str, int] = {}
    for node in nodes.values():
        if isinstance(node.issue, int):
            id_to_issue[node.node_id] = node.issue

    def should_create(node: PlanNode) -> bool:
        if node.issue is not None:
            return False
        if create_mode == "none":
            return False
        if create_mode == "all":
            return True
        return node.type == "epic"

    # Pass 1: create missing issues in topo order (so deps usually exist by creation time).
    created: dict[str, int] = {}
    for node_id in order:
        node = nodes[node_id]
        if not should_create(node):
            continue

        title = _issue_title(node)
        body = _node_to_issue_body(node, id_to_issue=id_to_issue)
        if not apply:
            print(f"[dry-run] Would create issue for {node_id}: {title!r}")
            continue

        issue = repo.create_issue(title=title, body=body)
        created[node_id] = issue.number
        id_to_issue[node_id] = issue.number
        _sync_assignees(issue, owners=node.owners, apply=apply)
        if node.labels:
            _sync_labels(issue, desired=node.labels, existing_labels=repo_labels, apply=apply)

    if update_plan:
        _update_plan_issue_numbers(raw_plan, created=created, plan_path=plan_path, apply=apply)

    # Pass 1b: reconcile/update existing issues (only when safe).
    reconciliation: list[ReconciliationItem] = []
    managed_nodes: list[PlanNode] = []
    for node in nodes.values():
        if not isinstance(id_to_issue.get(node.node_id), int):
            continue
        managed_nodes.append(node)

    for node in managed_nodes:
        issue_num = id_to_issue[node.node_id]
        issue = repo.get_issue(number=issue_num)

        desired_title = _issue_title(node)
        desired_body = _node_to_issue_body(node, id_to_issue=id_to_issue)

        current_body = issue.body or ""
        plan_text = node.plan_text()
        gh_text = current_body.strip()

        if current_body.strip() != desired_body.strip():
            if not plan_text and gh_text:
                # Plan is empty; keep GitHub as-is.
                continue
            if gh_text == "" and desired_body.strip() != "":
                if apply:
                    issue.edit(title=desired_title, body=desired_body)
                    _sync_assignees(issue, owners=node.owners, apply=apply)
                    if node.labels:
                        _sync_labels(issue, desired=node.labels, existing_labels=repo_labels, apply=apply)
                else:
                    print(f"[dry-run] Would update empty issue #{issue_num} from plan ({node.node_id})")
                continue

            reconciliation.append(
                ReconciliationItem(
                    node_id=node.node_id,
                    issue_number=issue_num,
                    title=desired_title,
                    plan_body=desired_body,
                    github_body=current_body,
                )
            )
            continue

        # Titles/labels/assignees can still drift even when body matches.
        if apply and issue.title != desired_title:
            issue.edit(title=desired_title)
        _sync_assignees(issue, owners=node.owners, apply=apply)
        if node.labels:
            _sync_labels(issue, desired=node.labels, existing_labels=repo_labels, apply=apply)

    if reconciliation:
        _write_reconciliation(reconciliation_path, reconciliation)
        print(f"Wrote reconciliation file: {reconciliation_path} ({len(reconciliation)} items)")
        if apply:
            print("Skipped updating issues with conflicts; reconcile and re-run.")

    # Pass 2: once issue numbers exist, rewrite bodies again to replace `#dag_id` anchors.
    for node in managed_nodes:
        issue_num = id_to_issue[node.node_id]
        issue = repo.get_issue(number=issue_num)

        desired_body = _node_to_issue_body(node, id_to_issue=id_to_issue)
        if issue.body and issue.body.strip() == desired_body.strip():
            continue
        if reconciliation and any(it.issue_number == issue_num for it in reconciliation):
            continue
        if not apply:
            print(f"[dry-run] Would update anchors/body for issue #{issue_num} ({node.node_id})")
            continue
        issue.edit(body=desired_body)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--plan", type=Path, default=PLAN_PATH_DEFAULT, help="Path to plan YAML (default: .agents/the_plan.yaml)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="Validate the plan DAG and schema")

    sync = sub.add_parser("sync", help="Sync plan nodes to GitHub issues")
    sync.add_argument("--repo", default="marin-community/marin", help="GitHub repo (owner/name)")
    sync.add_argument(
        "--create",
        default="epics",
        choices=["none", "epics", "all"],
        help="Create issues for nodes with issue:null (default: epics)",
    )
    sync.add_argument("--dry-run", action="store_true", help="Do not create/update issues; print intended actions")
    sync.add_argument("--apply", action="store_true", help="Actually create/update issues (overrides --dry-run)")
    sync.add_argument(
        "--update-plan",
        action="store_true",
        help="Update .agents/the_plan.yaml with newly-created issue numbers (requires --apply)",
    )
    sync.add_argument(
        "--reconcile-out",
        type=Path,
        default=RECONCILIATION_DEFAULT,
        help="Where to write reconciliation info (default: .agents/issue_reconciliation.yaml)",
    )
    sync.add_argument(
        "--token-env",
        default="GITHUB_TOKEN",
        help="Environment variable name holding a GitHub token (default: GITHUB_TOKEN)",
    )

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "check":
        plan_check(args.plan)
        print(f"OK: {args.plan} is a valid DAG with valid schema")
        return

    if args.cmd == "sync":
        apply = bool(args.apply)
        if args.update_plan and not apply:
            raise ValueError("--update-plan requires --apply")
        sync_plan_to_issues(
            plan_path=args.plan,
            repo_name=args.repo,
            create_mode=args.create,
            reconciliation_path=args.reconcile_out,
            apply=apply,
            update_plan=bool(args.update_plan),
            token_env=args.token_env,
        )
        return

    raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
