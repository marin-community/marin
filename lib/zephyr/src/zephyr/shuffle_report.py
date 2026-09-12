# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline charts of reducer input sizes read from Finelog."""

from collections import defaultdict
from collections.abc import Sequence
from html import escape
from pathlib import Path
from statistics import median

from zephyr.stats import ZephyrShuffleStat

_HISTOGRAM_BINS = 10
_STYLE = """
body { font: 16px system-ui, sans-serif; color: #183043; background: #f4f6f8; margin: 0; }
main { max-width: 1100px; margin: auto; padding: 24px; }
section { background: white; padding: 24px; margin: 24px 0; border: 1px solid #d5dfe6; border-radius: 8px; }
h1, h2, h3 { line-height: 1.2; } h2, p { overflow-wrap: anywhere; }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 400px), 1fr)); gap: 24px; }
.scroll { max-height: 480px; overflow: auto; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
th { text-align: left; } th, td { padding: 6px; border-bottom: 1px solid #e7edf1; }
td:last-child { text-align: right; white-space: nowrap; }
.track { min-width: 80px; background: #edf1f5; height: 16px; }
.bar { background: #157b86; height: 100%; }
.histogram .bar { background: #7357ac; }
.coverage { margin: 12px 0; height: 12px; background: #d5dfe6; }
.coverage .bar { background: #157b86; }
.unreported { color: #687581; background: #edf1f5; }
.note { color: #4c6071; } code { font-size: 0.9em; }
"""


def _rank_chart(values: list[tuple[int, int | None]], label: str, metric: str) -> str:
    maximum = max((value for _, value in values if value is not None), default=0)
    rows = []
    for target, value in sorted(values, key=lambda item: (item[1] is None, -(item[1] or 0), item[0])):
        if value is None:
            rows.append(
                f'<tr class="unreported" data-target="{target}" data-status="UNREPORTED">'
                f'<td>{target}</td><td colspan="2">UNREPORTED</td></tr>'
            )
            continue
        width = 100 * value / maximum if maximum else 0
        rows.append(
            f'<tr data-target="{target}" data-value="{value}"><td>{target}</td>'
            f'<td><div class="track"><div class="bar" style="width:{width:.4f}%"></div></div></td>'
            f"<td>{value:,}</td></tr>"
        )
    return (
        f'<div><h3>{label} by target reducer</h3><div class="scroll"><table data-metric="{metric}">'
        f"<thead><tr><th>Target</th><th>Relative size</th><th>{label}</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div></div>"
    )


def _histogram(values: list[int], label: str, metric: str) -> str:
    if not values:
        return f'<div data-distribution="{metric}"><h3>{label} distribution</h3><p>No measurements yet.</p></div>'
    maximum = max(values)
    bin_width = max(1, (maximum + _HISTOGRAM_BINS) // _HISTOGRAM_BINS)
    counts = [0] * (maximum // bin_width + 1)
    for value in values:
        counts[value // bin_width] += 1
    largest_count = max(counts)
    rows = []
    for index, count in enumerate(counts):
        lower = index * bin_width
        upper = min(maximum, lower + bin_width - 1)
        width = 100 * count / largest_count
        rows.append(
            f'<tr data-lower="{lower}" data-upper="{upper}" data-count="{count}">'
            f'<td>{lower:,}-{upper:,}</td><td><div class="track">'
            f'<div class="bar" style="width:{width:.4f}%"></div></div></td><td>{count}</td></tr>'
        )
    return (
        f'<div class="histogram"><h3>{label} distribution</h3><table data-metric="{metric}">'
        f"<thead><tr><th>{label} (inclusive)</th><th>Frequency</th><th>Reducers</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _summary(values: list[int], label: str) -> str:
    if not values:
        return f"{label}: unknown (no measurements)."
    middle = median(values)
    maximum = max(values)
    ratio = f"{maximum / middle:,.2f}x" if middle else "undefined (median is zero)"
    return f"{label}: median {middle:,.1f}; max {maximum:,}; max/median {ratio}."


def render_shuffle_report(records: Sequence[ZephyrShuffleStat], output_path: Path) -> None:
    """Write observed input-size charts grouped by execution and reduce stage.

    Args:
        records: Placeholders and observations. Keep the greatest attempt, then
            prefer a measurement over a placeholder, then the latest timestamp.
            Resolve timestamp ties by Finelog sequence in the query; otherwise
            the last supplied tied row wins. Infer unreported IDs from num_targets.
        output_path: Destination HTML file in an existing directory.
    """
    groups: dict[tuple[str, str], list[ZephyrShuffleStat]] = defaultdict(list)
    latest_targets: dict[tuple[str, str, int], ZephyrShuffleStat] = {}
    for record in records:
        identity = (record.execution_id, record.stage_name, record.target_shard)
        if not 0 <= record.target_shard < record.num_targets:
            raise ValueError(f"Shuffle target outside expected range: {identity}")
        sizes = (record.input_rows, record.payload_bytes, record.num_sources)
        if any(value is None for value in sizes) and not all(value is None for value in sizes):
            raise ValueError(f"Incomplete shuffle measurement: {identity}")
        if record.attempt < 0 or any(value < 0 for value in sizes if value is not None):
            raise ValueError(f"Negative shuffle observation: {identity}")
        previous = latest_targets.get(identity)
        priority = (record.attempt, record.input_rows is not None, record.ts)
        if previous is None or priority >= (previous.attempt, previous.input_rows is not None, previous.ts):
            latest_targets[identity] = record
    for identity, record in latest_targets.items():
        groups[identity[:2]].append(record)

    sections = []
    for (execution_id, stage_name), group in sorted(groups.items()):
        expected = group[0].num_targets
        if any(record.num_targets != expected for record in group):
            raise ValueError(f"Conflicting target counts for {execution_id}, {stage_name}")
        input_rows = [record.input_rows for record in group if record.input_rows is not None]
        payload_bytes = [record.payload_bytes for record in group if record.payload_bytes is not None]
        observed = len(input_rows)
        unreported = expected - observed
        coverage = "partial" if unreported else "complete"
        rows_by_target = {record.target_shard: record.input_rows for record in group}
        bytes_by_target = {record.target_shard: record.payload_bytes for record in group}
        empty_count = sum(value == 0 for value in input_rows)
        jobs = ", ".join(sorted({record.job_id for record in group if record.job_id})) or "local / unspecified"
        latest = max(record.ts for record in group).isoformat()
        sections.append(
            f'<section data-execution="{escape(execution_id)}" data-stage="{escape(stage_name)}">'
            f"<h2>Reduce stage: {escape(stage_name)}</h2>"
            f"<p>Execution: <code>{escape(execution_id)}</code><br>Job: <code>{escape(jobs)}</code>"
            f"<br>Latest record timestamp: {escape(latest)}</p>"
            f'<p data-targets="{observed}" data-expected-targets="{expected}" '
            f'data-unreported-targets="{unreported}" data-empty-targets="{empty_count}" data-coverage="{coverage}">'
            f"{observed:,} / {expected:,} targets reported ({100 * observed / expected:.1f}% coverage); "
            f"{unreported:,} UNREPORTED; {empty_count:,} reported empty. "
            "Coverage describes available telemetry, not task completion.</p>"
            f'<div class="coverage"><div class="bar" style="width:{100 * observed / expected:.4f}%"></div></div>'
            "<p>Observed targets only:<br>"
            f"{_summary(input_rows, 'Input rows')}<br>{_summary(payload_bytes, 'Encoded payload bytes')}</p>"
            '<div class="charts">'
            + _rank_chart(
                [(target, rows_by_target.get(target)) for target in range(expected)], "Input rows", "input_rows"
            )
            + _rank_chart(
                [(target, bytes_by_target.get(target)) for target in range(expected)], "Encoded bytes", "payload_bytes"
            )
            + _histogram(input_rows, "Input rows", "input_rows_histogram")
            + _histogram(payload_bytes, "Encoded bytes", "payload_bytes_histogram")
            + "</div></section>"
        )
    content = "".join(sections) or (
        '<p data-coverage="unknown">No shuffle observations were returned. Expected target count and coverage '
        "are unknown. No distribution can be shown.</p>"
    )
    output_path.write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Zephyr shuffle inputs</title><style>{_STYLE}</style></head><body><main>"
        '<h1>Zephyr shuffle inputs</h1><p class="note">Each reducer reports its input sizes after reading mapper '
        "metadata. Transport can delay publication. Encoded bytes exclude Parquet overhead and do not measure "
        "RAM or network traffic. "
        "Partition totals cannot identify a single hot key or explain runtime on their own.</p>"
        '<p class="note">Charts use linear scales. Rank charts sort each metric independently. '
        "Histograms and summaries include reported targets only, including explicitly reported empty targets. "
        "UNREPORTED targets are unknown, never zero: they may be queued, loading, or missing telemetry. "
        "The coordinator announces targets with null sizes before they run; missing announcements are inferred "
        "from the expected target count. Time filters can omit records. Each target uses its greatest attempt, "
        "preferring a measurement over a placeholder before comparing timestamps. "
        "This does not establish which attempt succeeded.</p>"
        f"{content}</main></body></html>",
        encoding="utf-8",
    )
