#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize stored CPU profiles for an Iris job and its descendants.

The Iris controller periodically captures py-spy ``--format raw`` (folded-stack)
profiles for every running task and stores them in the ``profiles`` SQLite DB
attached to the controller DB. The DBs are checkpointed under
``{remote_state_dir}/controller-state/{epoch_ms}/`` (zstd-compressed).

This script:

1. Locates the latest checkpoint under the configured cluster's state dir
   (default ``gs://marin-us-central2/iris/marin/state``).
2. Downloads + decompresses ``controller.sqlite3`` and ``profiles.sqlite3``
   into a local cache (``~/.cache/iris-job-profile/<cluster>/<epoch_ms>``).
3. Selects the latest CPU profile for every task whose task_id is the given
   job_id or a descendant, parses the folded stacks, and merges them.
4. Prints summary tables (per-task sample counts, top stacks, top leaf
   frames) and optionally writes the merged folded-stack file for piping
   into ``flamegraph.pl`` or speedscope.

Usage:
    uv run python scripts/job_profile_summary.py \\
        'https://iris.oa.dev/#/job/%2Frav%2Firis-run-tokenize.../zephyr.../zephyr-...workers-a0'

    uv run python scripts/job_profile_summary.py /rav/iris-run-tokenize.../...

    # Write a merged folded-stack file for flamegraph.pl
    uv run python scripts/job_profile_summary.py <job> -o merged.folded
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sqlite3
import subprocess
import sys
import urllib.parse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml
import zstandard

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger("job-profile")

CONTROLLER_DB = "controller.sqlite3"
PROFILES_DB = "profiles.sqlite3"

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def remote_state_dir_for_cluster(cluster: str) -> str:
    """Look up ``storage.remote_state_dir`` from ``examples/<cluster>.yaml``."""
    config_path = EXAMPLES_DIR / f"{cluster}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No cluster config at {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    state_dir = cfg.get("storage", {}).get("remote_state_dir")
    if not state_dir:
        raise ValueError(f"{config_path} has no storage.remote_state_dir")
    return state_dir


# Active task states in the iris controller schema. See lib/iris/OPS.md.
ACTIVE_STATES = {2, 3, 9}  # BUILDING, RUNNING, ASSIGNED
STATE_NAMES = {
    1: "PENDING",
    2: "BUILDING",
    3: "RUNNING",
    4: "SUCCEEDED",
    5: "FAILED",
    6: "KILLED",
    7: "WORKER_FAILED",
    8: "UNSCHEDULABLE",
    9: "ASSIGNED",
    10: "PREEMPTED",
}


@dataclass(frozen=True)
class TaskProfile:
    task_id: str
    job_id: str
    state: int
    captured_at_ms: int
    profile_bytes: bytes


# ---------------------------------------------------------------------------
# Job ID parsing
# ---------------------------------------------------------------------------


def parse_job_id(arg: str) -> str:
    """Accept either a raw ``/user/job/...`` path or a dashboard URL.

    Dashboard URLs look like ``https://iris.oa.dev/#/job/<percent-encoded path>``.
    """
    if arg.startswith(("http://", "https://")):
        # Strip the URL fragment and percent-decode.
        if "#" not in arg:
            raise ValueError(f"URL has no fragment: {arg}")
        fragment = arg.split("#", 1)[1]
        if not fragment.startswith("/job/"):
            raise ValueError(f"Unexpected fragment in URL: {fragment!r}")
        encoded = fragment[len("/job/") :]
        decoded = urllib.parse.unquote(encoded)
        if not decoded.startswith("/"):
            raise ValueError(f"Decoded job id missing leading '/': {decoded!r}")
        return decoded
    if not arg.startswith("/"):
        raise ValueError(f"Job id must start with '/': {arg!r}")
    return arg


# ---------------------------------------------------------------------------
# Checkpoint download
# ---------------------------------------------------------------------------


def find_latest_checkpoint(remote_state_dir: str) -> str:
    """Return the gs:// URI of the most recent timestamped checkpoint dir."""
    prefix = remote_state_dir.rstrip("/") + "/controller-state/"
    result = subprocess.run(
        ["gcloud", "storage", "ls", prefix],
        capture_output=True,
        text=True,
        check=True,
    )
    timestamps: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        line = line.strip().rstrip("/")
        if not line:
            continue
        basename = line.rsplit("/", 1)[-1]
        if basename.isdigit():
            timestamps.append((int(basename), line))
    if not timestamps:
        raise RuntimeError(f"No timestamped checkpoint directories under {prefix}")
    timestamps.sort(reverse=True)
    return timestamps[0][1] + "/"


def download_checkpoint(remote_dir: str, local_dir: Path) -> None:
    """rsync the checkpoint dir locally; idempotent (skips up-to-date files)."""
    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Syncing %s -> %s", remote_dir, local_dir)
    subprocess.run(
        ["gcloud", "storage", "rsync", remote_dir, str(local_dir) + "/"],
        check=True,
    )


def decompress_zst(src: Path, dst: Path) -> None:
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    logger.info("Decompressing %s", src.name)
    dctx = zstandard.ZstdDecompressor()
    with open(src, "rb") as f_in, open(dst, "wb") as f_out:
        dctx.copy_stream(f_in, f_out)


# ---------------------------------------------------------------------------
# SQLite queries
# ---------------------------------------------------------------------------


def open_db(controller_path: Path, profiles_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{controller_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute(f"ATTACH DATABASE 'file:{profiles_path}?mode=ro' AS profiles")
    return conn


def fetch_task_profiles(conn: sqlite3.Connection, root_job_id: str) -> list[TaskProfile]:
    """Pull the latest cpu profile per task under ``root_job_id``.

    A descendant task has a task_id of the form ``<root>/<...>/<index>``; the
    LIKE filter matches both direct tasks and sub-job tasks because sub-job
    rows are stored as separate jobs whose task_ids share the prefix.
    """
    like = root_job_id.rstrip("/") + "/%"
    rows = conn.execute(
        """
        SELECT
            t.task_id   AS task_id,
            t.job_id    AS job_id,
            t.state     AS state,
            p.captured_at_ms AS captured_at_ms,
            p.profile_data   AS profile_data
        FROM tasks AS t
        JOIN profiles.task_profiles AS p
          ON p.task_id = t.task_id
        WHERE (t.task_id = ? OR t.task_id LIKE ?)
          AND p.profile_kind = 'cpu'
          AND p.id = (
              SELECT id FROM profiles.task_profiles
              WHERE task_id = t.task_id AND profile_kind = 'cpu'
              ORDER BY id DESC LIMIT 1
          )
        ORDER BY t.task_id
        """,
        (root_job_id, like),
    ).fetchall()
    return [
        TaskProfile(
            task_id=r["task_id"],
            job_id=r["job_id"],
            state=r["state"],
            captured_at_ms=r["captured_at_ms"],
            profile_bytes=r["profile_data"],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Folded-stack parsing
# ---------------------------------------------------------------------------


def parse_folded(data: bytes) -> list[tuple[str, int]]:
    """Parse py-spy ``--format raw`` output into ``[(stack, count), ...]``.

    Each line is ``frame1;frame2;...;frameN <count>``. Frames may contain
    spaces (e.g. ``foo (file.py)``), so we split on the last whitespace.
    Lines that cannot be parsed are skipped with a debug log.
    """
    out: list[tuple[str, int]] = []
    text = data.decode("utf-8", errors="replace")
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            continue
        idx = line.rfind(" ")
        if idx < 0:
            continue
        stack, count_str = line[:idx], line[idx + 1 :]
        try:
            count = int(count_str)
        except ValueError:
            continue
        out.append((stack, count))
    return out


def leaf_of(stack: str) -> str:
    """The last frame of a semicolon-delimited stack."""
    return stack.rsplit(";", 1)[-1]


# ---------------------------------------------------------------------------
# Frame normalization
#
# py-spy raw frames carry per-task noise that prevents identical-by-meaning
# frames from merging in a shared-parent tree:
#   - process headers contain a PID and unique tmp paths:
#       process 87:"/app/.venv/bin/python -u -m zephyr.subprocess_worker
#                   /tmp/tmpML_bq91r.pkl /tmp/tmp5dblzotf.pkl"
#   - native frames sometimes appear as raw addresses:
#       0x7fb890d30b7b (libc.so.6)
#   - thread ids: "(tid: 8730996992)"
#   - rust monomorphization hashes: "::h3d171a55bcef49b9"
# ---------------------------------------------------------------------------

_TMPFILE_RE = re.compile(r"/tmp/tmp[A-Za-z0-9_]+(\.[A-Za-z0-9]+)?")
_HEX_ADDR_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
_TID_RE = re.compile(r"tid:\s*\d+")
_PID_RE = re.compile(r"\b(process|pid:)\s*\d+", re.IGNORECASE)
_RUST_HASH_RE = re.compile(r"::h[0-9a-f]{16}\b")


def normalize_frame(frame: str) -> str:
    frame = _TMPFILE_RE.sub(r"/tmp/<tmp>\1", frame)
    frame = _HEX_ADDR_RE.sub("<addr>", frame)
    frame = _TID_RE.sub("tid:N", frame)
    frame = _PID_RE.sub(lambda m: f"{m.group(1).lower()} N" if m.group(1).lower() == "process" else "pid:N", frame)
    frame = _RUST_HASH_RE.sub("", frame)
    return frame


def normalize_stack(stack: str) -> str:
    return ";".join(normalize_frame(f) for f in stack.split(";"))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_table(title: str, headers: list[str], rows: list[list[str]], stream=sys.stdout) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(f"\n{title}", file=stream)
    print("-" * len(title), file=stream)
    print(fmt.format(*headers), file=stream)
    print("  ".join("-" * w for w in widths), file=stream)
    for row in rows:
        print(fmt.format(*row), file=stream)


def fmt_int(n: int) -> str:
    return f"{n:,}"


def fmt_pct(part: int, total: int) -> str:
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
    merged_stacks: dict[str, int],
    total: int,
    out_path: Path,
    *,
    width: int = 1400,
    row_height: int = 16,
    title: str = "",
) -> None:
    """Write a basic flamegraph SVG from merged folded stacks (root at bottom)."""
    root: dict = {"name": "all", "value": 0, "kids": {}}
    for stack, count in merged_stacks.items():
        node = root
        node["value"] += count
        for frame in stack.split(";"):
            node = node["kids"].setdefault(frame, {"name": frame, "value": 0, "kids": {}})
            node["value"] += count

    def depth(n: dict) -> int:
        return 1 + max((depth(c) for c in n["kids"].values()), default=0)

    max_depth = depth(root)
    header = 30
    height = max_depth * row_height + header + 10
    min_visible = 0.15  # px; skip slivers

    rects: list[tuple[float, int, float, str, int]] = []

    def visit(node: dict, x: float, level: int) -> None:
        w = width * node["value"] / total
        if w < min_visible:
            return
        # Conventional flamegraph: deeper frames on top, root rows at bottom.
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
        # ~6.5 px/char in Verdana 11 — clip to avoid overflowing the rect.
        max_chars = max(0, int((w - 4) / 6.5))
        if max_chars < len(name):
            clipped = name[: max_chars - 1] + "…" if max_chars > 1 else ""
        title_text = f"{name} ({val:,} samples, {100 * val / total:.2f}%)"
        parts.append(
            f"<g><title>{_xml_escape(title_text)}</title>"
            f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="{row_height - 1}"'
            f' fill="{_frame_color(name)}" stroke="#000" stroke-width="0.2"/>'
        )
        if clipped:
            parts.append(f'<text x="{x + 3:.2f}" y="{y + row_height - 4}">{_xml_escape(clipped)}</text>')
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
        help="Cluster name; resolves storage.remote_state_dir from lib/iris/examples/<cluster>.yaml (default: marin)",
    )
    p.add_argument(
        "--remote-state-dir",
        help="Override the state dir inferred from --cluster (e.g. gs://...).",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "iris-job-profile",
        help="Local cache directory for downloaded checkpoints",
    )
    p.add_argument("--top", type=int, default=20, help="Top-N stacks/leaves/tasks to print (default 20)")
    p.add_argument("-o", "--output", type=Path, help="Write merged folded-stack profile to this path")
    p.add_argument(
        "--svg",
        type=Path,
        help="Write a basic flamegraph SVG to this path (default: <cache>/<cluster>/<epoch>/flame.svg)",
    )
    p.add_argument(
        "--checkpoint-dir",
        help="Use this checkpoint directory (gs:// or local) instead of finding the latest",
    )
    p.add_argument("--refresh", action="store_true", help="Force re-download even if cache is present")
    args = p.parse_args()

    job_id = parse_job_id(args.job)
    logger.info("Job ID: %s", job_id)

    state_dir = args.remote_state_dir or remote_state_dir_for_cluster(args.cluster)
    logger.info("State dir: %s", state_dir)

    if args.checkpoint_dir:
        remote_dir = args.checkpoint_dir.rstrip("/") + "/"
    else:
        remote_dir = find_latest_checkpoint(state_dir)
    epoch_label = remote_dir.rstrip("/").rsplit("/", 1)[-1]
    logger.info("Checkpoint: %s", remote_dir)

    local_dir = args.cache_dir / args.cluster / epoch_label
    if args.refresh and local_dir.exists():
        shutil.rmtree(local_dir)
    download_checkpoint(remote_dir, local_dir)

    for name in (CONTROLLER_DB, PROFILES_DB):
        zst = local_dir / f"{name}.zst"
        if zst.exists():
            decompress_zst(zst, local_dir / name)

    controller_path = local_dir / CONTROLLER_DB
    profiles_path = local_dir / PROFILES_DB
    if not controller_path.exists():
        logger.error("Missing %s in %s", CONTROLLER_DB, local_dir)
        return 1
    if not profiles_path.exists():
        logger.error("Missing %s in %s (controller may not have written profiles yet)", PROFILES_DB, local_dir)
        return 1

    conn = open_db(controller_path, profiles_path)
    profiles = fetch_task_profiles(conn, job_id)
    if not profiles:
        logger.error("No CPU profiles found under %s in checkpoint %s", job_id, epoch_label)
        return 2

    merged_stacks: dict[str, int] = defaultdict(int)
    merged_leaves: dict[str, int] = defaultdict(int)
    per_task_samples: dict[str, int] = {}
    per_task_state: dict[str, int] = {}
    per_task_captured: dict[str, int] = {}

    for prof in profiles:
        parsed = parse_folded(prof.profile_bytes)
        task_total = 0
        for stack, count in parsed:
            normalized = normalize_stack(stack)
            merged_stacks[normalized] += count
            merged_leaves[leaf_of(normalized)] += count
            task_total += count
        per_task_samples[prof.task_id] = task_total
        per_task_state[prof.task_id] = prof.state
        per_task_captured[prof.task_id] = prof.captured_at_ms

    grand_total = sum(per_task_samples.values())

    # --- Summary header ---------------------------------------------------
    print(f"Job:        {job_id}")
    print(f"Checkpoint: {remote_dir}")
    print(f"Tasks with profiles: {len(profiles)}")
    print(f"Merged samples:      {fmt_int(grand_total)}")
    print(f"Distinct stacks:     {fmt_int(len(merged_stacks))}")

    # --- Per-task table (top N only; full job paths are long) ------------
    sorted_tasks = sorted(per_task_samples.items(), key=lambda kv: -kv[1])
    rows = []
    job_prefix = job_id.rstrip("/") + "/"
    for task_id, samples in sorted_tasks[: args.top]:
        state = per_task_state[task_id]
        # Strip the common job prefix so per-task rows stay readable.
        short = task_id[len(job_prefix) :] if task_id.startswith(job_prefix) else task_id
        rows.append(
            [
                short,
                STATE_NAMES.get(state, str(state)),
                fmt_int(samples),
                fmt_pct(samples, grand_total),
                str(per_task_captured[task_id]),
            ]
        )
    print_table(
        f"Top {len(rows)} of {len(sorted_tasks)} tasks by samples (relative to '{job_prefix}')",
        ["task", "state", "samples", "share", "captured_at_ms"],
        rows,
    )

    # --- Top stacks -------------------------------------------------------
    top_stacks = sorted(merged_stacks.items(), key=lambda kv: -kv[1])[: args.top]
    print_table(
        f"Top {len(top_stacks)} stacks (merged across tasks)",
        ["samples", "share", "stack"],
        [[fmt_int(c), fmt_pct(c, grand_total), s] for s, c in top_stacks],
    )

    # --- Top leaves -------------------------------------------------------
    top_leaves = sorted(merged_leaves.items(), key=lambda kv: -kv[1])[: args.top]
    print_table(
        f"Top {len(top_leaves)} leaf frames (merged across tasks)",
        ["samples", "share", "frame"],
        [[fmt_int(c), fmt_pct(c, grand_total), s] for s, c in top_leaves],
    )

    # --- Optional folded-stack output ------------------------------------
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for stack, count in sorted(merged_stacks.items(), key=lambda kv: -kv[1]):
                f.write(f"{stack} {count}\n")
        logger.info("Wrote merged folded profile to %s (%d stacks)", args.output, len(merged_stacks))

    # --- Flame SVG --------------------------------------------------------
    svg_path = args.svg or (local_dir / "flame.svg")
    svg_title = f"{job_id}  ({len(profiles)} tasks, {grand_total:,} samples, ckpt {epoch_label})"
    render_flame_svg(merged_stacks, grand_total, svg_path, title=svg_title)
    logger.info("Wrote flame SVG to %s", svg_path)
    print(f"\nFlame SVG: {svg_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
