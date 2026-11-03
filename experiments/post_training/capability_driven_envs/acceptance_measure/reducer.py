# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recompute the measurement's funnel, costs and attribution from the allocations' ``records.jsonl`` (no model calls).

``reduce_rows(rows, declared)`` is pure: ``declared`` is every allocation key the sample and arms declare, so an
allocation with no terminal stays in the denominator (acceptance per declared allocation, with the missing outcomes
as a bound). Outputs, per arm:

* the funnel: declared, terminal, accepted (and how many under the ``saturated`` label), rejected, abandoned by
  budget, abandoned by infrastructure retries, held or missing; acceptance per declared (with bounds) and per terminal;
* tokens per stage (raw input, cached input, output, reasoning, and uncached input + output) and per accept, with
  the missing allocations' tokens counted because they are paid;
* the (a)-(e) attribution: (a) decisive claims the shadow judge supported, (b) claims it contradicted, (c) typed
  holds and shadow errors (charged components must hold none), (d) saturated rounds on otherwise passing versions,
  (e) claims in the strictness classes and probe-battery misses;
* per-class repair convergence (a fed-back class whose next round's FA and FR both pass) and the regression count
  (a component that passed in a round and failed in the next);
* the shadow decisions at the terminal (L3, L4, judge, realistic-only) against the actual decision;

and, per pair of arms, the paired table of accepts on capabilities both arms brought to a terminal (McNemar's b, c).

    uv run --frozen python -m experiments.post_training.capability_driven_envs.acceptance_measure.reducer \\
        --output s3://marin-us-east-02a/marin/taskforge/acceptance-measure/<run> --sample <sample.json>
"""

import itertools
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence

import click

from experiments.post_training.capability_driven_envs.acceptance_measure.driver import (
    ALLOCATIONS_DIR,
    SEAL_FILE,
    Store,
    read_source,
)
from experiments.post_training.capability_driven_envs.acceptance_measure.records import RECORDS_FILE
from experiments.post_training.capability_driven_envs.acceptance_measure.runtime import TAXONOMY
from experiments.post_training.capability_driven_envs.acceptance_measure.shadows import read_taxonomy
from experiments.post_training.capability_driven_envs.acceptance_measure.treatment import ARM_NAMES

STRICTNESS = frozenset(name for name, entry in read_taxonomy(TAXONOMY).items() if entry["strictness"])
INFRA_REASON = "infrastructure retry budget exhausted"
TOKEN_FIELDS = ("tokens_in", "tokens_cached", "tokens_out", "tokens_reasoning")


def outcome(terminal: Mapping | None) -> str:
    """The funnel bucket of an allocation's terminal row (``missing`` when it has none)."""
    if terminal is None or terminal.get("decision") is None:
        return "missing"
    decision = terminal["decision"]
    if decision == "abandoned":
        return "abandoned_infra" if INFRA_REASON in (terminal.get("reason") or "") else "abandoned_budget"
    return {"accepted": "accepted", "rejected": "rejected"}.get(decision, "held")


def _tokens(rows: Iterable[Mapping]) -> dict[str, dict[str, int]]:
    by_stage: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        if row["row"] == "cost" and row.get("kind") == "llm_call":
            for field in TOKEN_FIELDS:
                by_stage[row["stage"]][field] += row.get(field) or 0
        if row["row"] == "claim" and row.get("judge") and "tokens_in" in row["judge"]:
            for field in ("tokens_in", "tokens_cached", "tokens_out"):
                by_stage["shadow_judge"][field] += row["judge"][field]
    for counts in by_stage.values():
        counts["uncached_plus_output"] = counts["tokens_in"] - counts["tokens_cached"] + counts["tokens_out"]
    return {stage: dict(counts) for stage, counts in sorted(by_stage.items())}


def _convergence_and_regressions(rows: Sequence[Mapping]) -> tuple[Counter, Counter, int]:
    status: dict[tuple[int, str], list[str]] = defaultdict(list)
    for row in rows:
        if row["row"] == "component":
            status[(row["round"], row["component"])].append(row["status"])
    passed = {key: all(s == "pass" for s in values) for key, values in status.items()}
    fed, converged = Counter(), Counter()
    for row in rows:
        if row["row"] != "feedback":
            continue
        after = row["round"] + 1
        fixed = all(passed.get((after, c), False) for c in ("fa", "fr")) and any(
            (after, c) in passed for c in ("fa", "fr")
        )
        for cls in set(row["classes"]):
            fed[cls] += 1
            converged[cls] += fixed
    regressions = sum(
        1 for (round_no, component), ok in passed.items() if ok and passed.get((round_no + 1, component)) is False
    )
    return fed, converged, regressions


def arm_summary(allocations: Mapping[str, Sequence[Mapping]], declared: Sequence[str]) -> dict:
    funnel = Counter()
    saturated = 0
    tokens_rows: list[Mapping] = []
    attribution = Counter()
    fed, converged, regressions = Counter(), Counter(), 0
    shadows: dict[str, Counter] = defaultdict(Counter)
    for key in declared:
        rows = allocations.get(key, ())
        terminal = next((r for r in rows if r["row"] == "terminal"), None)
        bucket = outcome(terminal)
        funnel[bucket] += 1
        if bucket == "accepted" and terminal is not None and terminal["band_label"] == "saturated":
            saturated += 1
        tokens_rows += rows
        for row in rows:
            if row["row"] == "claim":
                verdict = (row.get("judge") or {}).get("verdict")
                attribution["a_supported"] += verdict == "finder_correct"
                attribution["b_contradicted"] += verdict == "finder_false"
                attribution["e_strictness_claims"] += row["class"] in STRICTNESS
                attribution["l4_precondition_fail"] += row["l4"].get("precondition") == "fail"
            elif row["row"] == "component":
                attribution["c_typed_holds"] += row["status"] == "hold"
                attribution["e_probe_battery_misses"] += row["component"] == "probe_battery" and row["status"] == "fail"
            elif row["row"] == "claim_error":
                attribution["c_shadow_errors"] += 1
            elif row["row"] == "submission" and not row["charged"]:
                attribution[f"refused_{row['refusal']}"] += 1
            elif row["row"] == "cost" and row.get("kind") == "detector":
                attribution[f"detector_{row['event']}"] += 1
        if terminal is not None and terminal.get("l3"):
            attribution["d_saturated_last_round"] += bool(terminal["l3"].get("accept"))
        if terminal is not None and terminal.get("shadow_decisions"):
            for rule, decision in terminal["shadow_decisions"].items():
                shadows[rule][str(decision)] += 1
        f, c, r = _convergence_and_regressions(rows)
        fed, converged, regressions = fed + f, converged + c, regressions + r
    n = len(declared)
    terminal_count = funnel["accepted"] + funnel["rejected"] + funnel["abandoned_budget"]
    unresolved = n - terminal_count
    tokens = _tokens(tokens_rows)
    total = sum(stage["uncached_plus_output"] for stage in tokens.values())
    accepted = funnel["accepted"]
    return {
        "declared": n,
        "funnel": dict(funnel),
        "accepted_saturated": saturated,
        "acceptance_per_declared": None if n == 0 else accepted / n,
        "acceptance_per_declared_bounds": None if n == 0 else [accepted / n, (accepted + unresolved) / n],
        "acceptance_per_terminal": None if terminal_count == 0 else accepted / terminal_count,
        "tokens_by_stage": tokens,
        "tokens_uncached_plus_output": total,
        "tokens_per_accept": None if accepted == 0 else total / accepted,
        "attribution": dict(attribution),
        "repair_convergence": {cls: {"fed_back": fed[cls], "converged": converged[cls]} for cls in sorted(fed)},
        "regressions": regressions,
        "shadow_decisions": {rule: dict(counts) for rule, counts in shadows.items()},
    }


def paired(allocations: Mapping[str, Sequence[Mapping]], capabilities: Sequence[str], a: str, b: str) -> dict:
    """McNemar's table of accepts for capabilities that reached a terminal in both arms."""
    table = Counter()
    for capability in capabilities:
        ends = []
        for arm in (a, b):
            rows = allocations.get(f"{arm}/{capability}", ())
            ends.append(outcome(next((r for r in rows if r["row"] == "terminal"), None)))
        if any(e in ("missing", "held", "abandoned_infra") for e in ends):
            table["unpaired"] += 1
            continue
        table[f"{a}_{'acc' if ends[0] == 'accepted' else 'rej'}__{b}_{'acc' if ends[1] == 'accepted' else 'rej'}"] += 1
    return dict(table)


def reduce_rows(rows: Iterable[Mapping], capabilities: Sequence[str], arms: Sequence[str] = ARM_NAMES) -> dict:
    by_allocation: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        by_allocation[row["allocation"]].append(row)
    return {
        "arms": {arm: arm_summary(by_allocation, [f"{arm}/{c}" for c in capabilities]) for arm in arms},
        "pairs": {f"{b}-{a}": paired(by_allocation, capabilities, a, b) for a, b in itertools.pairwise(arms)},
    }


def load_rows(store: Store, keys: Sequence[str]) -> list[dict]:
    rows = []
    for key in keys:
        prefix = f"{ALLOCATIONS_DIR}/{key}"
        if not store.exists(f"{prefix}/{SEAL_FILE}"):
            continue
        rows += [
            json.loads(line) for line in store.read(f"{prefix}/{RECORDS_FILE}").decode().splitlines() if line.strip()
        ]
    return rows


@click.command()
@click.option("--output", required=True, help="The measurement's output root.")
@click.option("--sample", "sample_path", required=True)
@click.option("--wave", type=click.Choice(["1", "2", "all"]), default="all", show_default=True)
@click.option("--arms", default=",".join(ARM_NAMES), show_default=True)
def main(output: str, sample_path: str, wave: str, arms: str) -> None:
    """Print the reduction of every sealed allocation as JSON; unsealed ones count as missing."""
    sample = json.loads(read_source(sample_path))
    entries = (
        sample["sample"]
        if wave == "all"
        else sample["sample"][: sample["wave_size"]] if wave == "1" else sample["sample"][sample["wave_size"] :]
    )
    capabilities = [e["capability"] for e in entries]
    names = tuple(arms.split(","))
    rows = load_rows(Store(output), [f"{arm}/{c}" for arm in names for c in capabilities])
    click.echo(json.dumps(reduce_rows(rows, capabilities, names), indent=2))


if __name__ == "__main__":
    main()
