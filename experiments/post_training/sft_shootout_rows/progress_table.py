# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render one progress.py summary as a dated section of the progress table, with projections and flags (task #218).

Projections scale each family's measured value per attempted task to the family's not-held-out pool tasks (pool v5:
TMax 1,914, SWE-smith 1,995). Relay parity is the projected relay pool (every usable relay episode, before the
subsample to B) over the projected plain B, per family and in total.

Flags (printed as FLAG lines and listed in the section):
- projected B above 29,626,824 (4 x the frozen replay slice; continuation v2 would be needed),
- relay parity below 1.0 (meaningful once a family's follow-up rounds are done),
- a drop reason outside KNOWN_DROPS, or any capability-token drop.

Usage:
    python -m experiments.post_training.sft_shootout_rows.progress_table <progress log> <progress.md> <label>
"""

import datetime
import json
import sys
from pathlib import Path

POOL_NOT_HELD_OUT = {"tmax": 1914, "swesmith": 1995}
REPLAY_CEILING_B = 4 * 7_406_706
KNOWN_DROPS = {
    "not_passed",
    "heldout_task",
    "over_context",
    "summarized_trial",
    "touches_agent_tool_env",
    "touches_library_copy_of_repo",
    "contains_proxy_capability_token",
    "harness_exception",
    "task_not_in_pool",
    "trace_problems",
    "relay_run",
    "plain_run",
    "not_heldout",
    "family_mismatch",
    "no_teacher_turn",
    "student_turn_in_plain_run",
}  # build.py candidates(); 'render_error: ...' is deliberately not listed, so it is flagged.


def fmt(value: float) -> str:
    return f"{value:,.0f}"


def main() -> None:
    log, table, label = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    line = next(line for line in log.read_text().splitlines() if "PROGRESS_SUMMARY " in line)
    summary = json.loads(line.split("PROGRESS_SUMMARY ", 1)[1])
    plain, rs, relay = (summary["arms"][arm] for arm in ("plain", "rs-multi", "relay"))
    flags = []
    families = sorted(POOL_NOT_HELD_OUT)
    out = [f"## {label} ({datetime.date.today().isoformat()})", ""]
    out.append(f"Finished shards ({len(summary['shards'])}): " + ", ".join(f"`{s}`" for s in summary["shards"]) + ".")
    out.append(f"Pool manifest `{plain['pool_manifest_sha256'][:12]}…`.")
    out.append("")
    out += [
        "| | " + " | ".join(families) + " | total |",
        "|---|" + "---:|" * (len(families) + 1),
    ]
    rows: dict[str, dict[str, str]] = {}
    projected_b, projected_relay = {}, {}
    for family in families:
        eligibility = plain["eligibility"].get(family, {})
        attempted = eligibility.get("tasks_attempted", 0)
        eligible = eligibility.get("tasks_eligible", 0)
        b = plain["splits"]["train"]["supervised_tokens"].get(family, 0)
        projected_b[family] = b / attempted * POOL_NOT_HELD_OUT[family] if attempted else 0
        pool = (relay.get("relay_pool") or {}).get(family, {})
        relay_attempted = relay["eligibility"].get(family, {}).get("tasks_attempted", 0)
        projected_relay[family] = (
            pool.get("supervised_tokens", 0) / relay_attempted * POOL_NOT_HELD_OUT[family] if relay_attempted else 0
        )
        values = {
            "plain tasks attempted / eligible": f"{attempted:,} / {eligible:,}",
            "plain eligibility": f"{eligible / attempted:.3f}" if attempted else "-",
            "plain B (supervised)": fmt(b),
            "plain context tokens": fmt(plain["splits"]["train"]["context_tokens"].get(family, 0)),
            "projected B": fmt(projected_b[family]),
            "RS-multi epoch supervised": " / ".join(
                fmt(rs["splits"][f"epoch-{e}"]["supervised_tokens"].get(family, 0)) for e in range(3)
            ),
            "relay tasks attempted / with an episode": f"{relay_attempted:,} / {pool.get('tasks', 0):,}",
            "relay pool (episodes, supervised)": f"{pool.get('episodes', 0):,}, {fmt(pool.get('supervised_tokens', 0))}",
            "relay pool / B (measured)": f"{pool.get('supervised_tokens', 0) / b:.2f}" if b else "-",
            "projected relay pool / projected B": (
                f"{projected_relay[family] / projected_b[family]:.2f}" if projected_b[family] else "-"
            ),
        }
        for key, value in values.items():
            rows.setdefault(key, {})[family] = value
        for arm_name, arm in (("plain", plain), ("relay", relay)):
            for reason, count in sorted(arm["candidates"]["dropped_by_family"].get(family, {}).items()):
                rows.setdefault(f"{arm_name} drop: {reason}", {})[family] = fmt(count)
    total_b = sum(projected_b.values())
    total_relay = sum(projected_relay.values())
    totals = {
        "plain B (supervised)": fmt(plain["B_supervised_tokens"]),
        "projected B": fmt(total_b),
        "projected relay pool / projected B": f"{total_relay / total_b:.2f}" if total_b else "-",
    }
    for key, by_family in rows.items():
        cells = [by_family.get(family, "0" if " drop: " in key else "-") for family in families]
        out.append(f"| {key} | " + " | ".join(cells) + f" | {totals.get(key, '')} |")
    epochs = {
        "plain": [plain["splits"]["train"]],
        "rs-multi": [rs["splits"][f"epoch-{e}"] for e in range(3)],
        "relay": [relay["splits"]["train"]],
    }
    padded = "; ".join(
        f"{arm} " + " / ".join(f"{s.get('padded_rows', s['rows'] + s['padding_rows']):,}" for s in splits)
        for arm, splits in epochs.items()
    )
    out += ["", f"Rows per epoch (padded, multiple of 16): {padded}."]
    replay = plain["replay"]
    out += [
        "",
        f"Replay cut (plain's B {fmt(plain['B_supervised_tokens'])}): R = {fmt(replay['target_supervised_tokens'])}, "
        f"achieved {fmt(replay['supervised_tokens'])} in {replay['rows']} of {replay['slice_rows']} rows, share "
        f"{replay['share_of_supervised']['train']:.3f}; rs-multi and relay `rows_sha256` "
        + (
            "equal plain's"
            if rs["replay"]["rows_sha256"] == relay["replay"]["rows_sha256"] == replay["rows_sha256"]
            else "DIFFER"
        )
        + f". At the projected B, R = {fmt(total_b / 4)}.",
    ]
    if total_b > REPLAY_CEILING_B:
        flags.append(f"projected B {fmt(total_b)} exceeds {fmt(REPLAY_CEILING_B)}: replay continuation v2 needed")
    for family in families:
        if projected_b[family] and projected_relay[family] < projected_b[family]:
            flags.append(
                f"{family} relay pool projects to {projected_relay[family] / projected_b[family]:.2f} of B "
                "(below parity; check whether its follow-up rounds are done)"
            )
    for arm, data in summary["arms"].items():
        for family, drops in data["candidates"]["dropped_by_family"].items():
            for reason, count in drops.items():
                if reason not in KNOWN_DROPS:
                    flags.append(f"new drop reason in {arm}/{family}: {reason} ({count})")
                if reason == "contains_proxy_capability_token" and count:
                    flags.append(f"capability token drops in {arm}/{family}: {count}")
    out += ["", "Flags: " + ("; ".join(flags) if flags else "none") + ".", ""]
    with table.open("a") as stream:
        stream.write("\n".join(out) + "\n")
    for flag in flags:
        print("FLAG " + flag)
    print(f"appended {label} to {table}")


if __name__ == "__main__":
    main()
