# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Track committed 140B-pipeline tokens and regenerate a lightweight SVG plot.

The tracker samples the currently saved rollout count in GCS, converts it to
committed tokens using the experiment's 11.4k-token estimate per rollout, and
appends one row every N minutes to a CSV. After each sample it rewrites:

* `progress_140b_latest.json`
* `progress_140b.csv`
* `progress_140b.svg`

This script intentionally avoids external plotting dependencies.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import subprocess
import time
from pathlib import Path

OUT = "gs://marin-us-central2/experiments/swe_zero_100b"
TARGET_ROLLOUTS = 12_291_000
TOKENS_PER_ROLLOUT = 11_400


BYTES_PER_ROLLOUT = 48_500
"""Empirical avg bytes per rollout entry in the JSON files (measured from
shard 053: 6667 rollouts / 325 MB ≈ 48.7 KB/rollout). Used as a fast,
race-free estimator when the exact count from ``gsutil cat`` would race
with concurrent checkpoint writes."""

_GS_LS_RE = re.compile(r"^\s*(\d+)\s+\S+\s+(gs://\S+)$")


def measure_rollouts(output_root: str) -> int:
    """Estimate total rollouts from file sizes (gsutil ls -l).

    This avoids the read-race that plagued the old gsutil-cat counter:
    when workers checkpoint, the file is momentarily empty/partial during
    upload, causing the cat-based counter to undercount by 10-30k.
    Metadata (file size) is always consistent.
    """
    proc = subprocess.run(
        ["gsutil", "ls", "-l", f"{output_root.rstrip('/')}/shard_*/rollouts*.json"],
        capture_output=True,
        text=True,
        check=False,
    )
    total_bytes = 0
    for line in proc.stdout.splitlines():
        m = _GS_LS_RE.match(line)
        if m:
            total_bytes += int(m.group(1))
    return int(total_bytes / BYTES_PER_ROLLOUT)


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    high_water_rollouts = 0
    normalized = []
    for row in rows:
        rollouts = int(row["rollouts"])
        high_water_rollouts = max(high_water_rollouts, rollouts)
        row = dict(row)
        row.setdefault("high_water_rollouts", str(high_water_rollouts))
        row.setdefault("high_water_pct_complete", f"{100 * high_water_rollouts / TARGET_ROLLOUTS:.3f}")
        row.setdefault("high_water_tokens_b", f"{high_water_rollouts * TOKENS_PER_ROLLOUT / 1e9:.3f}")
        normalized.append(row)
    return normalized


def append_row(csv_path: Path, row: dict[str, str]) -> list[dict[str, str]]:
    rows = load_rows(csv_path)
    if rows and rows[-1]["timestamp_utc"] == row["timestamp_utc"]:
        return rows
    last_high_water = int(rows[-1]["high_water_rollouts"]) if rows else 0
    high_water_rollouts = max(last_high_water, int(row["rollouts"]))
    row = dict(row)
    row["high_water_rollouts"] = str(high_water_rollouts)
    row["high_water_pct_complete"] = f"{100 * high_water_rollouts / TARGET_ROLLOUTS:.3f}"
    row["high_water_tokens_b"] = f"{high_water_rollouts * TOKENS_PER_ROLLOUT / 1e9:.3f}"
    fieldnames = [
        "timestamp_utc",
        "rollouts",
        "target_rollouts",
        "pct_complete",
        "tokens_b",
        "high_water_rollouts",
        "high_water_pct_complete",
        "high_water_tokens_b",
    ]
    rows.append(row)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_latest_json(path: Path, row: dict[str, str]) -> None:
    path.write_text(json.dumps(row, indent=2) + "\n")


def _svg_polyline_points(values: list[float], x0: int, y0: int, width: int, height: int) -> str:
    if not values:
        return ""
    ymin = 0.0
    ymax = max(values) if max(values) > 0 else 1.0
    if len(values) == 1:
        return f"{x0},{y0 + height}"
    pts = []
    for i, val in enumerate(values):
        x = x0 + (width * i / (len(values) - 1))
        frac = (val - ymin) / (ymax - ymin) if ymax > ymin else 0.0
        y = y0 + height - (frac * height)
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def write_svg(path: Path, rows: list[dict[str, str]]) -> None:
    width = 1000
    height = 420
    margin_l = 90
    margin_r = 30
    margin_t = 40
    margin_b = 70
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    token_vals = [float(r["tokens_b"]) for r in rows]
    high_water_vals = [float(r["high_water_tokens_b"]) for r in rows]
    ymax_tokens = max(high_water_vals or token_vals) if (high_water_vals or token_vals) else 1.0
    if ymax_tokens <= 0:
        ymax_tokens = 1.0

    token_points = _svg_polyline_points(token_vals, margin_l, margin_t, plot_w, plot_h)
    high_water_points = _svg_polyline_points(high_water_vals, margin_l, margin_t, plot_w, plot_h)

    y_grid = []
    y_labels = []
    for i in range(6):
        frac = i / 5
        y = margin_t + plot_h - (plot_h * frac)
        val = ymax_tokens * frac
        x2 = margin_l + plot_w
        xl = margin_l - 12
        y_grid.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{x2}" y2="{y:.1f}" stroke="#d7dee8" stroke-width="1"/>')
        y_labels.append(
            f'<text x="{xl}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#425466">{val:.2f}B</text>'
        )

    x_labels = []
    if rows:
        label_idx = sorted(set([0, len(rows) - 1] + [i * (len(rows) - 1) // 3 for i in range(4)]))
        for idx in label_idx:
            x = margin_l if len(rows) == 1 else margin_l + (plot_w * idx / (len(rows) - 1))
            ts = rows[idx]["timestamp_utc"][11:16]
            x_labels.append(
                f'<text x="{x:.1f}" y="{height - 22}" text-anchor="middle" font-size="12" fill="#425466">{ts}</text>'
            )

    latest = rows[-1] if rows else None
    summary = ""
    if latest:
        summary = (
            f'current {latest["tokens_b"]}B ({latest["pct_complete"]}%)'
            f' | high-water {latest["high_water_tokens_b"]}B ({latest["high_water_pct_complete"]}%)'
        )

    x_mid = margin_l + plot_w / 2
    y_mid = margin_t + plot_h / 2
    x2 = margin_l + plot_w
    y2 = margin_t + plot_h
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="#f7f4ec"/>',
        f'<text x="{margin_l}" y="24" font-size="22" fill="#17212b">SWE-ZERO 140B Progress</text>',
        f'<text x="{margin_l}" y="48" font-size="13" fill="#425466">{summary}</text>',
        f'<rect x="{margin_l}" y="{margin_t}" width="{plot_w}" height="{plot_h}" fill="#fffdf8" stroke="#9fb3c8"/>',
        "".join(y_grid),
        "".join(y_labels),
        f'<line x1="{margin_l}" y1="{y2}" x2="{x2}" y2="{y2}" stroke="#506274" stroke-width="1.5"/>',
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{y2}" stroke="#506274" stroke-width="1.5"/>',
        f'<polyline fill="none" stroke="#d17b0f" stroke-width="3" points="{high_water_points}"/>',
        f'<polyline fill="none" stroke="#126e82" stroke-width="3" points="{token_points}"/>',
        f'<text x="{margin_l + 16}" y="{margin_t + 16}" font-size="12" fill="#126e82">current</text>',
        f'<text x="{margin_l + 90}" y="{margin_t + 16}" font-size="12" fill="#d17b0f">high-water</text>',
        "".join(x_labels),
        f'<text x="{x_mid:.1f}" y="{height - 6}" text-anchor="middle" font-size="12" fill="#425466">UTC</text>',
        f'<text transform="translate(20 {y_mid:.1f}) rotate(-90)" text-anchor="middle" font-size="12">'
        "Committed tokens (B)</text>",
        "</svg>",
    ]
    svg = "\n".join(svg_lines) + "\n"
    path.write_text(svg)


def sample(output_dir: Path, output_root: str) -> dict[str, str]:
    ts = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
    rollouts = measure_rollouts(output_root)
    pct = 100 * rollouts / TARGET_ROLLOUTS
    tokens_b = rollouts * TOKENS_PER_ROLLOUT / 1e9
    row = {
        "timestamp_utc": ts.isoformat().replace("+00:00", "Z"),
        "rollouts": str(rollouts),
        "target_rollouts": str(TARGET_ROLLOUTS),
        "pct_complete": f"{pct:.3f}",
        "tokens_b": f"{tokens_b:.3f}",
    }
    csv_path = output_dir / "progress_140b.csv"
    latest_json = output_dir / "progress_140b_latest.json"
    svg_path = output_dir / "progress_140b.svg"
    rows = append_row(csv_path, row)
    write_latest_json(latest_json, rows[-1])
    write_svg(svg_path, rows)
    return rows[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Track committed 140B pipeline tokens")
    parser.add_argument("--output-dir", default="experiments/swe_zero/progress")
    parser.add_argument("--output-root", default=OUT)
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    while True:
        row = sample(output_dir, args.output_root)
        print(json.dumps(row))
        if args.once:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
