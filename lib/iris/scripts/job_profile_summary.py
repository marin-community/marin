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

1. Resolves the cluster's finelog deployment (``finelog.config`` in
   ``config/<cluster>.yaml``; defaults to the cluster name) and opens a tunnel
   to it, exactly as ``finelog query`` does.
2. Queries every CPU profile whose ``source`` is the given job or a descendant
   and downloads the speedscope blobs.
3. Parses the speedscope stacks, aggregates sample weights across captures, and
   reports the breakdown **per worker sub-job** (each task's immediate parent
   job): CPU rolled up by library/binary, plus a hot-path Mermaid call graph
   that renders inline in GitHub markdown.
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

# In --per-subjob mode, only emit a full detail section for sub-jobs at or above
# this share of total job CPU; smaller ones stay in the overview table only.
SUBJOB_DETAIL_MIN_SHARE = 0.01


def finelog_config_for_cluster(cluster: str) -> str:
    """Return the finelog deployment name for an Iris cluster.

    Reads ``finelog.config`` from ``config/<cluster>.yaml`` (the field that
    names the cluster's finelog deployment); falls back to the cluster name.
    """
    config_path = CONFIG_DIR / f"{cluster}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No cluster config at {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("finelog", {}).get("config") or cluster


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


# A frame is "native" if py-spy couldn't symbolize it (``<addr>``) or it lives in
# a shared object (``*.so``/``*.dylib``). The interpreter binary (``python3.12``)
# and ``.py`` frames are kept — they carry meaningful symbols (e.g. gc).
_NATIVE_LIB_RE = re.compile(r"\.so(\.\d+)*$|\.dylib$")


def _is_native_frame(frame: str) -> bool:
    name, _, lib = frame.partition(" (")
    if name.startswith("<addr>"):
        return True
    return bool(_NATIVE_LIB_RE.search(lib.rstrip(")")))


def meaningful_leaf(stack: str) -> str:
    """The deepest frame worth naming: roll the leaf up past native frames.

    A stack often tops out in ``<addr> (libc.so.6)`` or a ``*.abi3.so`` symbol,
    which says little about *what* is running. Walk from the leaf toward the root
    and return the deepest non-native frame (the application/Python operation
    responsible); fall back to the deepest symbolized frame, then the raw leaf.
    The full stack is left untouched — native frames still drive the library
    rollup and the call graph.
    """
    frames = stack.split(";")
    for frame in reversed(frames):
        if not _is_native_frame(frame):
            return frame
    for frame in reversed(frames):
        if not frame.startswith("<addr>"):
            return frame
    return frames[-1]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class Aggregate:
    """Merged folded stacks plus bookkeeping for a set of tasks."""

    stacks: dict[str, float]
    leaves: dict[str, float]  # true topmost frame — drives the library rollup
    app_leaves: dict[str, float]  # leaf rolled up past native frames (meaningful_leaf)
    total: float
    tasks: set[str]

    @classmethod
    def empty(cls) -> "Aggregate":
        return cls(
            stacks=defaultdict(float),
            leaves=defaultdict(float),
            app_leaves=defaultdict(float),
            total=0.0,
            tasks=set(),
        )

    def add(self, source: str, parsed: ParsedProfile) -> None:
        self.tasks.add(source)
        for stack, weight in parsed.stacks.items():
            self.stacks[stack] += weight
            self.leaves[leaf_of(stack)] += weight
            self.app_leaves[meaningful_leaf(stack)] += weight
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
# Library rollup — group self time by the binary/file a leaf frame lives in.
# This is what makes unsymbolized native time legible: every `<addr> (libc.so.6)`
# / `free (libc.so.6)` / `malloc (libc.so.6)` leaf rolls into one `libc.so.6` row.
# ---------------------------------------------------------------------------

_LIBRARY_RE = re.compile(r"\(([^()]*)\)\s*$")


def library_of(leaf_label: str) -> str:
    m = _LIBRARY_RE.search(leaf_label)
    return m.group(1) if m and m.group(1) else "(builtin)"


def library_rollup_rows(agg: Aggregate, top: int) -> list[list[str]]:
    by_lib: dict[str, float] = defaultdict(float)
    for leaf, weight in agg.leaves.items():
        by_lib[library_of(leaf)] += weight
    ranked = sorted(by_lib.items(), key=lambda kv: -kv[1])[:top]
    return [[f"{w:,.1f}", fmt_pct(w, agg.total), lib] for lib, w in ranked]


# ---------------------------------------------------------------------------
# Call-tree → Mermaid flowchart
#
# Build a weighted call tree (root → leaf), keep only nodes above a cumulative
# share, collapse single-child chains into one node, and render the result as a
# Mermaid flowchart (renders inline in GitHub markdown). This shows the call
# path responsible for CPU, not a flat list of leaves.
# ---------------------------------------------------------------------------


@dataclass
class CallNode:
    label: str
    value: float
    children: "dict[str, CallNode]"


def build_call_tree(stacks: dict[str, float]) -> CallNode:
    root = CallNode("(all)", 0.0, {})
    for stack, weight in stacks.items():
        root.value += weight
        node = root
        for frame in stack.split(";"):
            child = node.children.get(frame)
            if child is None:
                child = CallNode(frame, 0.0, {})
                node.children[frame] = child
            child.value += weight
            node = child
    return root


def _short_frame(label: str, maxlen: int = 38) -> str:
    """A compact node label: function name only (drop the ``(file)`` suffix)."""
    name = label.split(" (", 1)[0]
    if name.startswith("process N"):
        name = "process"
    if len(name) > maxlen:
        name = name[: maxlen - 1] + "…"
    return name


def _mermaid_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("(", "&#40;")
        .replace(")", "&#41;")
    )


def render_mermaid_calltree(root: CallNode, total: float, *, min_share: float = 0.05, max_nodes: int = 18) -> str:
    """Render the hot-path call tree as a fenced Mermaid flowchart block."""
    min_value = total * min_share
    node_lines: list[str] = []
    edge_lines: list[str] = []
    counter = {"n": 0}

    def emit(node: CallNode) -> str:
        # Collapse a run of single-significant-child frames into one chain node.
        chain = [node]
        cur = node
        while True:
            kids = [c for c in cur.children.values() if c.value >= min_value]
            if len(kids) == 1:
                cur = kids[0]
                chain.append(cur)
            else:
                break
        nid = f"N{counter['n']}"
        counter["n"] += 1
        shown: list[str] = []
        for name in (_short_frame(n.label) for n in chain if n.label != "(all)"):
            if not shown or shown[-1] != name:
                shown.append(name)
        shown = shown or ["(all)"]
        label = ("… → " if len(shown) > 2 else "") + " → ".join(shown[-2:])
        node_lines.append(f'  {nid}["{_mermaid_escape(label)}<br/>{100.0 * cur.value / total:.0f}%"]')
        for kid in sorted((c for c in cur.children.values() if c.value >= min_value), key=lambda c: -c.value):
            if counter["n"] >= max_nodes:
                break
            edge_lines.append(f"  {nid} --> {emit(kid)}")
        return nid

    emit(root)
    return "\n".join(["```mermaid", "flowchart TD", *node_lines, *edge_lines, "```"])


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

    def print_detail(agg: Aggregate, *, level: int) -> None:
        """Rollup-by-library table + hot-path Mermaid call graph for one aggregate."""
        print_table(
            "CPU by library (self time)",
            ["weight", "share", "library"],
            library_rollup_rows(agg, min(args.top, 10)),
            markdown=md,
            level=level,
        )
        if md:
            print(f"\n{'#' * level} Hot call paths\n")
        else:
            print("\nHot call paths")
        print(render_mermaid_calltree(build_call_tree(agg.stacks), agg.total))

    def print_stacks(agg: Aggregate, *, level: int) -> None:
        top = sorted(agg.stacks.items(), key=lambda kv: -kv[1])[: args.top]
        print_table(
            f"Top {len(top)} raw stacks",
            ["weight", "share", "stack"],
            [[f"{w:,.1f}", fmt_pct(w, agg.total), s] for s, w in top],
            markdown=md,
            level=level,
        )

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
        top_fn, top_fn_w = max(agg.app_leaves.items(), key=lambda kv: kv[1], default=("-", 0.0))
        sub_rows.append(
            [
                label,
                fmt_int(len(agg.tasks)),
                f"{agg.total:,.1f}",
                fmt_pct(agg.total, overall.total),
                f"{top_fn} [{fmt_pct(top_fn_w, agg.total)}]",
            ]
        )
    print_table(
        "CPU by worker sub-job (share of job total; hottest function, native rolled up to nearest named frame)",
        ["sub-job", "tasks", "weight", "job share", "hottest fn"],
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
        ranked = sorted(per_subjob.items(), key=lambda kv: -kv[1].total)
        detailed = [(lbl, agg) for lbl, agg in ranked if agg.total >= overall.total * SUBJOB_DETAIL_MIN_SHARE]
        if md:
            print("\n## Per worker sub-job")
        for label, agg in detailed:
            heading = f"{label} — {agg.total:,.1f} ({fmt_pct(agg.total, overall.total)} of job, {len(agg.tasks)} tasks)"
            if md:
                print(f"\n### {heading}")
            else:
                print(f"\n{'=' * len(heading)}\n{heading}\n{'=' * len(heading)}")
            print_detail(agg, level=4)
            if args.show_stacks:
                print_stacks(agg, level=4)
        omitted = len(ranked) - len(detailed)
        if omitted:
            note = f"{omitted} sub-jobs below {SUBJOB_DETAIL_MIN_SHARE:.0%} of job CPU omitted (see overview table)."
            print(f"\n_{note}_" if md else f"\n{note}")

    # --- Whole-job rollup + hot paths ------------------------------------
    if md:
        print("\n## Whole job")
    print_detail(overall, level=2)
    if args.show_stacks:
        print_stacks(overall, level=2)

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
