#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize stored CPU profiles for an Iris job and its descendant sub-jobs.

Iris workers periodically capture py-spy CPU profiles (speedscope JSON) for
every running task and ship them to the cluster's finelog server under the
``iris.profile`` stats namespace (see ``lib/iris/OPS.md`` → "Stats Namespaces").
Each row carries a ``source`` (the task path ``/user/job/.../<index>``), a
``captured_at`` timestamp, and the raw speedscope ``profile_data`` blob.

This script:

1. Resolves the cluster's finelog deployment (``log_server_config`` in
   ``config/<cluster>.yaml``; defaults to the cluster name) and opens a tunnel
   to it, exactly as ``finelog query`` does.
2. Queries every CPU profile whose ``source`` is the given job or a descendant
   and downloads the speedscope blobs.
3. Parses the speedscope stacks, aggregates sample weights across captures, and
   reports the breakdown **per worker sub-job** (each task's immediate parent
   job), plus the merged top leaf frames for the whole job.
4. Optionally writes a merged folded-stack file (for ``flamegraph.pl`` /
   speedscope) and a flamegraph SVG.

Usage:
    uv run python scripts/job_profile_summary.py \\
        'https://iris.oa.dev/#/job/%2Frav%2Firis-run-job-20260601-054954'

    uv run python scripts/job_profile_summary.py /rav/iris-run-job-20260601-054954

    # Drill into a single worker sub-job and show its top stacks.
    uv run python scripts/job_profile_summary.py /rav/iris-run-job-... \\
        --subjob zephyr-fuzzy-dups-aa8bcf4c-p0-workers-a0 --show-stacks

    # Write a merged folded-stack file + flamegraph SVG.
    uv run python scripts/job_profile_summary.py <job> -o merged.folded --svg flame.svg
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import urllib.parse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml
from finelog.client.log_client import LogClient
from finelog.deploy.config import FinelogConfig, load_finelog_config
from iris.cluster.runtime.profile import PROFILE_NAMESPACE, ProfileType
from rigging.tunnel import GcpSshForwardTarget, K8sPortForwardTarget, TunnelTarget, open_tunnel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger("job-profile")

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def finelog_config_for_cluster(cluster: str) -> str:
    """Return the finelog deployment name for an Iris cluster.

    Reads ``log_server_config`` from ``config/<cluster>.yaml`` (the field that
    names the cluster's finelog deployment); falls back to the cluster name.
    """
    config_path = CONFIG_DIR / f"{cluster}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No cluster config at {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("log_server_config") or cluster


def tunnel_target(cfg: FinelogConfig) -> TunnelTarget:
    """Build a rigging tunnel target from a finelog deployment block.

    Mirrors ``finelog.deploy.cli._tunnel_target``: GCP forwards over IAP SSH
    impersonating the deployment service account; k8s uses ``kubectl
    port-forward``.
    """
    if cfg.deployment.gcp is not None:
        gcp = cfg.deployment.gcp
        return GcpSshForwardTarget(
            project=gcp.project,
            zone=gcp.zone,
            instance=cfg.name,
            port=cfg.port,
            impersonate_service_account=gcp.service_account,
        )
    assert cfg.deployment.k8s is not None
    k8s = cfg.deployment.k8s
    return K8sPortForwardTarget(namespace=k8s.namespace, service=cfg.name, port=cfg.port)


@dataclass(frozen=True)
class TaskProfile:
    source: str
    profile_bytes: bytes


# ---------------------------------------------------------------------------
# Job ID parsing
# ---------------------------------------------------------------------------


def parse_job_id(arg: str) -> str:
    """Accept either a raw ``/user/job/...`` path or a dashboard URL.

    Dashboard URLs look like ``https://iris.oa.dev/#/job/<percent-encoded path>``.
    """
    if arg.startswith(("http://", "https://")):
        if "#" not in arg:
            raise ValueError(f"URL has no fragment: {arg}")
        fragment = arg.split("#", 1)[1]
        if not fragment.startswith("/job/"):
            raise ValueError(f"Unexpected fragment in URL: {fragment!r}")
        decoded = urllib.parse.unquote(fragment[len("/job/") :])
        if not decoded.startswith("/"):
            raise ValueError(f"Decoded job id missing leading '/': {decoded!r}")
        return decoded.rstrip("/")
    if not arg.startswith("/"):
        raise ValueError(f"Job id must start with '/': {arg!r}")
    return arg.rstrip("/")


# ---------------------------------------------------------------------------
# finelog query
# ---------------------------------------------------------------------------


def fetch_task_profiles(client: LogClient, root_job_id: str, *, max_rows: int) -> list[TaskProfile]:
    """Pull every CPU profile capture under ``root_job_id`` from ``iris.profile``.

    A descendant task has a ``source`` of the form ``<root>/.../<index>``; the
    ``LIKE`` filter matches the root and all descendants. We keep every capture
    (not just the latest) so the aggregate reflects the whole run, not a single
    10-minute snapshot.
    """
    like = root_job_id + "/%"
    # SQL string literals: job ids are controller-generated (alnum/dash/slash),
    # so simple quoting is safe here. Single quotes cannot appear in a task id.
    sql = f"""
        SELECT source, profile_data
        FROM "{PROFILE_NAMESPACE}"
        WHERE (source = '{root_job_id}' OR source LIKE '{like}')
          AND type = '{ProfileType.CPU.value}'
        ORDER BY source, captured_at
    """
    table = client.query(sql, max_rows=max_rows)
    sources = table.column("source").to_pylist()
    blobs = table.column("profile_data").to_pylist()
    return [TaskProfile(source=src, profile_bytes=bytes(blob)) for src, blob in zip(sources, blobs, strict=True) if blob]


# ---------------------------------------------------------------------------
# Frame normalization
#
# py-spy frames carry per-task noise that prevents identical-by-meaning frames
# from merging: PIDs and unique tmp paths in the process header, raw native
# addresses, thread ids, and rust monomorphization hashes.
# ---------------------------------------------------------------------------

_TMPFILE_RE = re.compile(r"/tmp/tmp[A-Za-z0-9_]+(\.[A-Za-z0-9]+)?")
_HEX_ADDR_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
_TID_RE = re.compile(r"tid:\s*\d+")
_PID_RE = re.compile(r"\b(process|pid:)\s*\d+", re.IGNORECASE)
_RUST_HASH_RE = re.compile(r"::h[0-9a-f]{16}\b")


def normalize_frame(name: str, file: str | None) -> str:
    """Render one speedscope frame to a stable folded-stack token.

    Combines the function name with the file basename (matching py-spy's
    ``func (file.py)`` folded style) and scrubs per-task noise.
    """
    label = name
    if file:
        base = file.rsplit("/", 1)[-1]
        if base and base not in label:
            label = f"{label} ({base})"
    label = _TMPFILE_RE.sub(r"/tmp/<tmp>\1", label)
    label = _HEX_ADDR_RE.sub("<addr>", label)
    label = _TID_RE.sub("tid:N", label)
    label = _PID_RE.sub(lambda m: "process N" if m.group(1).lower() == "process" else "pid:N", label)
    label = _RUST_HASH_RE.sub("", label)
    return label


# ---------------------------------------------------------------------------
# speedscope parsing
# ---------------------------------------------------------------------------


@dataclass
class ParsedProfile:
    """Folded stacks parsed from one speedscope blob (root-first frame order)."""

    stacks: dict[str, float]  # "frame0;frame1;...;leaf" -> summed weight
    total: float


def parse_speedscope(data: bytes) -> ParsedProfile:
    """Parse a py-spy speedscope JSON blob into weighted folded stacks.

    speedscope ``profiles[*].samples`` are stacks of frame indices ordered
    root → leaf; ``weights`` is the parallel per-sample weight (py-spy emits
    sampling-interval seconds). We sum weights across all per-thread profiles.
    """
    doc = json.loads(data)
    frames = doc.get("shared", {}).get("frames", [])
    rendered = [normalize_frame(f.get("name", "?"), f.get("file")) for f in frames]

    stacks: dict[str, float] = defaultdict(float)
    total = 0.0
    for prof in doc.get("profiles", []):
        samples = prof.get("samples", [])
        weights = prof.get("weights", [])
        for sample, weight in zip(samples, weights, strict=False):
            if not sample:
                continue
            stack = ";".join(rendered[i] for i in sample if 0 <= i < len(rendered))
            if not stack:
                continue
            w = float(weight)
            stacks[stack] += w
            total += w
    return ParsedProfile(stacks=dict(stacks), total=total)


def leaf_of(stack: str) -> str:
    return stack.rsplit(";", 1)[-1]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class Aggregate:
    """Merged folded stacks plus bookkeeping for a set of tasks."""

    stacks: dict[str, float]
    leaves: dict[str, float]
    total: float
    tasks: set[str]

    @classmethod
    def empty(cls) -> Aggregate:
        return cls(stacks=defaultdict(float), leaves=defaultdict(float), total=0.0, tasks=set())

    def add(self, source: str, parsed: ParsedProfile) -> None:
        self.tasks.add(source)
        for stack, weight in parsed.stacks.items():
            self.stacks[stack] += weight
            self.leaves[leaf_of(stack)] += weight
            self.total += weight


def subjob_of(source: str, root_job_id: str) -> str:
    """The worker sub-job a task belongs to: its immediate parent job path.

    ``/root/sub/workers-a0/3`` → ``sub/workers-a0`` (relative to the root job).
    Coordinator tasks like ``/root/sub/0`` group under ``sub``; the root job's
    own tasks (``/root/0``) group under ``(root)``.
    """
    parent = source.rsplit("/", 1)[0]
    if parent == root_job_id:
        return "(root)"
    prefix = root_job_id + "/"
    return parent[len(prefix) :] if parent.startswith(prefix) else parent


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _md_cell(s: str) -> str:
    """Escape one GitHub-markdown table cell; wrap code-ish text in backticks.

    Frame/stack text contains ``<>`` and ``;`` that would otherwise render as
    HTML, so cells carrying those are rendered as inline code.
    """
    if any(c in s for c in "<>`;"):
        return "`" + s.replace("`", "'").replace("|", "\\|") + "`"
    return s.replace("|", "\\|")


def render_table(headers: list[str], rows: list[list[str]], *, markdown: bool = False) -> str:
    if markdown:
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(_md_cell(c) for c in row) + " |" for row in rows],
        ]
        return "\n".join(lines)
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers), "  ".join("-" * w for w in widths)]
    lines += [fmt.format(*row) for row in rows]
    return "\n".join(lines)


def print_table(
    title: str, headers: list[str], rows: list[list[str]], *, markdown: bool = False, level: int = 3
) -> None:
    if markdown:
        print(f"\n{'#' * level} {title}\n")
        print(render_table(headers, rows, markdown=True))
    else:
        print(f"\n{title}")
        print("-" * len(title))
        print(render_table(headers, rows))


def fmt_int(n: int) -> str:
    return f"{n:,}"


def fmt_pct(part: float, total: float) -> str:
    return f"{(100.0 * part / total) if total else 0.0:5.1f}%"


# ---------------------------------------------------------------------------
# Flame SVG (self-contained: no flamegraph.pl dependency)
# ---------------------------------------------------------------------------


def _xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _frame_color(name: str) -> str:
    """FlameGraph-style warm palette (orange/red/yellow), seeded by frame name."""
    h = abs(hash(name))
    r = 200 + (h % 56)
    g = 60 + ((h >> 6) % 130)
    b = 40 + ((h >> 13) % 50)
    return f"rgb({r},{g},{b})"


def render_flame_svg(
    merged_stacks: dict[str, float],
    total: float,
    out_path: Path,
    *,
    width: int = 1400,
    row_height: int = 16,
    title: str = "",
) -> None:
    """Write a basic flamegraph SVG from merged folded stacks (root at bottom)."""
    root: dict = {"name": "all", "value": 0.0, "kids": {}}
    for stack, count in merged_stacks.items():
        node = root
        node["value"] += count
        for frame in stack.split(";"):
            node = node["kids"].setdefault(frame, {"name": frame, "value": 0.0, "kids": {}})
            node["value"] += count

    def depth(n: dict) -> int:
        return 1 + max((depth(c) for c in n["kids"].values()), default=0)

    max_depth = depth(root)
    header = 30
    height = max_depth * row_height + header + 10
    min_visible = 0.15  # px; skip slivers

    rects: list[tuple[float, float, float, str, float]] = []

    def visit(node: dict, x: float, level: int) -> None:
        w = width * node["value"] / total
        if w < min_visible:
            return
        y = height - 5 - (level + 1) * row_height
        rects.append((x, y, w, node["name"], node["value"]))
        cx = x
        for child in sorted(node["kids"].values(), key=lambda c: -c["value"]):
            visit(child, cx, level + 1)
            cx += width * child["value"] / total

    cx = 0.0
    for child in sorted(root["kids"].values(), key=lambda c: -c["value"]):
        visit(child, cx, 0)
        cx += width * child["value"] / total

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
        ' font-family="Verdana, sans-serif" font-size="11">',
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text x="{width / 2}" y="18" text-anchor="middle" font-size="14"'
        f' font-weight="bold">{_xml_escape(title)}</text>',
    ]
    for x, y, w, name, val in rects:
        clipped = name
        max_chars = max(0, int((w - 4) / 6.5))
        if max_chars < len(name):
            clipped = name[: max_chars - 1] + "…" if max_chars > 1 else ""
        title_text = f"{name} ({val:,.1f}, {100 * val / total:.2f}%)"
        parts.append(
            f"<g><title>{_xml_escape(title_text)}</title>"
            f'<rect x="{x:.2f}" y="{y:.0f}" width="{w:.2f}" height="{row_height - 1}"'
            f' fill="{_frame_color(name)}" stroke="#000" stroke-width="0.2"/>'
        )
        if clipped:
            parts.append(f'<text x="{x + 3:.2f}" y="{y + row_height - 4:.0f}">{_xml_escape(clipped)}</text>')
        parts.append("</g>")
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("job", help="Job ID (e.g. /user/job/sub) or iris dashboard URL")
    p.add_argument(
        "--cluster",
        default="marin",
        help="Cluster name; resolves the finelog deployment via config/<cluster>.yaml (default: marin)",
    )
    p.add_argument("--finelog-config", help="finelog deployment name (default: resolved from the cluster config)")
    p.add_argument("--tunnel-timeout", type=float, default=60.0, help="Seconds to wait for the tunnel (default 60)")
    p.add_argument(
        "--subjob",
        help="Restrict to one worker sub-job (its label from the per-sub-job table, or any source substring)",
    )
    p.add_argument("--top", type=int, default=20, help="Top-N leaves/stacks/tasks to print (default 20)")
    p.add_argument("--show-stacks", action="store_true", help="Print the top merged stacks table")
    p.add_argument("--show-tasks", action="store_true", help="Print the top tasks table")
    p.add_argument(
        "--per-subjob",
        action="store_true",
        help="Emit a full section (top leaves, and stacks with --show-stacks) for every worker sub-job",
    )
    p.add_argument("--markdown", action="store_true", help="Render the report as GitHub markdown")
    p.add_argument("-o", "--output", type=Path, help="Write merged folded-stack profile to this path")
    p.add_argument("--svg", type=Path, help="Write a flamegraph SVG to this path")
    p.add_argument("--max-rows", type=int, default=500_000, help="Reject results larger than this (default 500k)")
    args = p.parse_args()

    job_id = parse_job_id(args.job)
    logger.info("Job ID: %s", job_id)

    finelog_name = args.finelog_config or finelog_config_for_cluster(args.cluster)
    cfg = load_finelog_config(finelog_name)
    logger.info("finelog deployment: %s (%s)", finelog_name, cfg.name)

    with open_tunnel(tunnel_target(cfg), timeout=args.tunnel_timeout) as url:
        client = LogClient.connect(url)
        try:
            profiles = fetch_task_profiles(client, job_id, max_rows=args.max_rows)
        finally:
            client.close()

    if args.subjob:
        profiles = [pr for pr in profiles if args.subjob in pr.source]
    if not profiles:
        logger.error("No CPU profiles found under %s%s", job_id, f" matching {args.subjob!r}" if args.subjob else "")
        return 2

    overall = Aggregate.empty()
    per_subjob: dict[str, Aggregate] = defaultdict(Aggregate.empty)
    per_task: dict[str, float] = defaultdict(float)
    n_captures = 0

    for prof in profiles:
        try:
            parsed = parse_speedscope(prof.profile_bytes)
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning("Skipping unparseable profile for %s: %s", prof.source, e)
            continue
        if parsed.total <= 0:
            continue
        n_captures += 1
        overall.add(prof.source, parsed)
        per_subjob[subjob_of(prof.source, job_id)].add(prof.source, parsed)
        per_task[prof.source] += parsed.total

    if overall.total <= 0:
        logger.error("Profiles contained no samples")
        return 2

    md = args.markdown

    def leaf_rows(agg: Aggregate) -> list[list[str]]:
        top = sorted(agg.leaves.items(), key=lambda kv: -kv[1])[: args.top]
        return [[f"{w:,.1f}", fmt_pct(w, agg.total), s] for s, w in top]

    def stack_rows(agg: Aggregate) -> list[list[str]]:
        top = sorted(agg.stacks.items(), key=lambda kv: -kv[1])[: args.top]
        return [[f"{w:,.1f}", fmt_pct(w, agg.total), s] for s, w in top]

    # --- Summary header ---------------------------------------------------
    summary = [
        f"**Job:** `{job_id}`",
        f"**finelog:** {finelog_name} ({cfg.name})",
        f"**CPU captures:** {fmt_int(n_captures)} across {fmt_int(len(overall.tasks))} tasks",
        f"**Aggregate weight:** {overall.total:,.1f} py-spy speedscope sample-seconds",
        f"**Distinct stacks:** {fmt_int(len(overall.stacks))}",
    ]
    if args.subjob:
        summary.insert(2, f"**Filter:** source contains `{args.subjob}`")
    if md:
        print(f"# CPU profile — `{job_id}`\n")
        for line in summary:
            print(f"- {line}")
    else:
        for line in summary:
            print(line.replace("**", "").replace("`", ""))

    # --- Per worker sub-job overview -------------------------------------
    sub_rows = []
    for label, agg in sorted(per_subjob.items(), key=lambda kv: -kv[1].total):
        top_leaf, top_leaf_w = max(agg.leaves.items(), key=lambda kv: kv[1], default=("-", 0.0))
        sub_rows.append(
            [
                label,
                fmt_int(len(agg.tasks)),
                f"{agg.total:,.1f}",
                fmt_pct(agg.total, overall.total),
                f"{top_leaf} [{fmt_pct(top_leaf_w, agg.total)}]",
            ]
        )
    print_table(
        "CPU by worker sub-job (share of job total; hottest leaf within the sub-job)",
        ["sub-job", "tasks", "weight", "job share", "hottest leaf"],
        sub_rows,
        markdown=md,
        level=2,
    )

    # --- Per-task (optional) ---------------------------------------------
    if args.show_tasks:
        sorted_tasks = sorted(per_task.items(), key=lambda kv: -kv[1])[: args.top]
        prefix = job_id + "/"
        rows = [
            [src[len(prefix) :] if src.startswith(prefix) else src, f"{w:,.1f}", fmt_pct(w, overall.total)]
            for src, w in sorted_tasks
        ]
        print_table(f"Top {len(rows)} tasks by CPU weight", ["task", "weight", "share"], rows, markdown=md, level=2)

    # --- Full section per worker sub-job (optional) ----------------------
    if args.per_subjob:
        if md:
            print("\n## Per worker sub-job")
        for label, agg in sorted(per_subjob.items(), key=lambda kv: -kv[1].total):
            heading = f"{label} — {agg.total:,.1f} ({fmt_pct(agg.total, overall.total)} of job, {len(agg.tasks)} tasks)"
            if md:
                print(f"\n### {heading}")
            else:
                print(f"\n{'=' * len(heading)}\n{heading}\n{'=' * len(heading)}")
            print_table(
                f"Top {len(leaf_rows(agg))} leaf frames",
                ["weight", "share", "frame"],
                leaf_rows(agg),
                markdown=md,
                level=4,
            )
            if args.show_stacks:
                print_table(
                    f"Top {len(stack_rows(agg))} stacks",
                    ["weight", "share", "stack"],
                    stack_rows(agg),
                    markdown=md,
                    level=4,
                )

    # --- Whole-job top stacks (optional) ---------------------------------
    if args.show_stacks and not args.per_subjob:
        print_table(
            f"Top {len(stack_rows(overall))} stacks (merged across captures)",
            ["weight", "share", "stack"],
            stack_rows(overall),
            markdown=md,
            level=2,
        )

    # --- Whole-job top leaves --------------------------------------------
    print_table(
        f"Top {len(leaf_rows(overall))} leaf frames (where CPU is actually spent)",
        ["weight", "share", "frame"],
        leaf_rows(overall),
        markdown=md,
        level=2,
    )

    # --- Optional folded-stack output ------------------------------------
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for stack, count in sorted(overall.stacks.items(), key=lambda kv: -kv[1]):
                f.write(f"{stack} {round(count)}\n")
        logger.info("Wrote merged folded profile to %s (%d stacks)", args.output, len(overall.stacks))

    # --- Flame SVG --------------------------------------------------------
    if args.svg:
        title = f"{job_id}  ({n_captures} captures, {overall.total:,.0f} weight)"
        render_flame_svg(overall.stacks, overall.total, args.svg, title=title)
        logger.info("Wrote flame SVG to %s", args.svg)
        print(f"\nFlame SVG: {args.svg}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
