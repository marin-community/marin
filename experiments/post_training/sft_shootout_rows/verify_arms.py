# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify production arm prefixes against their manifests and against plain.

For each prefix: _SUCCESS present and no DRY_RUN_NOT_FOR_TRAINING marker; every file's sha256 equals the manifest's
output_hashes; each epoch file's decompressed sha256 equals splits[epoch].rows_sha256; rows plus padding equal
padded_rows, a multiple of 16, with steps = padded_rows / 16; padding rows are pad-only with zero loss; per-family
supervised and context tokens equal the split's totals; no training row comes from a held-out task (pool v5 ids in the
manifest). Across arms: replay rows in every epoch file are the same multiset as plain's epoch 0. Plain: the three
epoch files hold the same rows (as a multiset) in different orders. RS-multi: each epoch's agentic supervised tokens
are within 1% of plain's B, on plain's task slots. Recovery: plain's trials and B exactly, every row trains plain's
target tokens, and rows not marked injected are byte-identical to plain's. Relay: each family's episodes reach plain's
B for it to whole-episode granularity (dropping the family's last chosen episode would fall below it), or the family
took every usable episode and its shortfall is reported; every epoch holds the same rows, each trial once. A relay
component (COMPONENT_NOT_AN_ARM, some families, no replay) gets the same family checks; given with a full relay arm
(components listed first), the arm's rows for the component's families must be byte-identical to the component's. Any
other arm is refused.

--dry-run checks dry builds (build.py --dry-run) instead: each prefix must carry the DRY_RUN_NOT_FOR_TRAINING marker
and no _SUCCESS; every other check is the same.

    python -m experiments.post_training.sft_shootout_rows.verify_arms [--dry-run] <plain prefix> [<other arm prefix> ...]
"""

import argparse
import collections
import gzip
import hashlib
import json
from typing import NamedTuple

from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.sft_shootout_rows.build import COMPONENT_MARKER, DRY_RUN_MARKER, EPOCHS, plain_family_b
from experiments.post_training.sweep_data.prepare_agentic import BATCH, PAD_TOKEN, read_component
from experiments.post_training.sweep_data.select import CONTEXT


class ArmCheck(NamedTuple):
    manifest: dict
    index: list[dict]
    # Per epoch: {"replay": Counter of row hashes, "agentic": Counter of row hashes, "supervised": per-family tokens}.
    epochs: dict[int, dict]
    report: dict
    selection: list[dict]


def check_prefix(uri: str, dry_run: bool = False) -> ArmCheck:
    """Check one production (or, with dry_run, one dry-build) prefix against its own manifest; raise on any mismatch."""
    root = StoragePath(uri)
    if dry_run:
        if (root / "_SUCCESS").exists() or not (root / DRY_RUN_MARKER).exists():
            raise ValueError(f"{uri}: a dry build has a {DRY_RUN_MARKER} marker and no _SUCCESS")
    elif not (root / "_SUCCESS").exists():
        raise ValueError(f"{uri}: no _SUCCESS marker")
    elif (root / DRY_RUN_MARKER).exists():
        raise ValueError(f"{uri}: carries a dry-run marker")
    manifest, _, files = read_component(root)
    if (COMPONENT_MARKER in files) != bool(manifest.get("relay_component")):
        raise ValueError(f"{uri}: component marker and manifest disagree")
    index = [json.loads(line) for line in files["train-index.jsonl"].decode().splitlines()]
    heldout = {task for tasks in manifest["heldout_task_ids"].values() for task in tasks}
    leaked = sorted({item["task_id"] for item in index if item.get("task_id") in heldout})
    if leaked:
        raise ValueError(f"{uri}: held-out tasks in training: {leaked[:5]}")
    epochs = {}
    for epoch in range(EPOCHS):
        split = manifest["splits"][f"epoch-{epoch}"]
        data = gzip.decompress(files[f"train-epoch-{epoch}-prebuilt.jsonl.gz"])
        if hashlib.sha256(data).hexdigest() != split["rows_sha256"]:
            raise ValueError(f"{uri}: epoch {epoch} rows_sha256 differs")
        lines = data.decode().splitlines()
        entries = [item for item in index if item["epoch"] == epoch]
        if len(lines) != split["padded_rows"] or split["padded_rows"] % BATCH or len(entries) != split["rows"]:
            raise ValueError(f"{uri}: epoch {epoch} row counts differ")
        if split["steps"] * BATCH != split["padded_rows"]:
            raise ValueError(f"{uri}: epoch {epoch} steps differ")
        supervised, context = collections.Counter(), collections.Counter()
        replay = collections.Counter()
        agentic = collections.Counter()
        # Per agentic trial: (row sha256, sha256 of the trained target tokens), for row-level comparison with plain.
        by_trial = {}
        for number, line in enumerate(lines):
            row = json.loads(line)
            ids, weights = row["input_ids"], row["loss_weights"]
            if len(ids) != CONTEXT or len(weights) != CONTEXT or not set(weights) <= {0, 1}:
                raise ValueError(f"{uri}: epoch {epoch} row {number} malformed")
            if number >= len(entries):
                if any(weights) or set(ids) != {PAD_TOKEN}:
                    raise ValueError(f"{uri}: epoch {epoch} padding row {number} not empty")
                continue
            entry = entries[number]
            if entry["row"] != number or sum(weights) != entry["supervised_tokens"]:
                raise ValueError(f"{uri}: epoch {epoch} row {number} differs from its index entry")
            if any(weights[entry["context_tokens"] - 1 :]):
                raise ValueError(f"{uri}: epoch {epoch} row {number} has loss past its content")
            supervised[entry["family"]] += entry["supervised_tokens"]
            context[entry["family"]] += entry["context_tokens"]
            digest = hashlib.sha256(line.encode()).hexdigest()
            (replay if entry["family"] == "replay" else agentic)[digest] += 1
            if entry["family"] != "replay":
                # loss_weights[i] scores the prediction of input_ids[i + 1].
                targets = [ids[i + 1] for i, weight in enumerate(weights) if weight]
                by_trial[entry["trial_id"]] = (digest, hashlib.sha256(json.dumps(targets).encode()).hexdigest())
        if dict(supervised) != split["supervised_tokens"] or dict(context) != split["context_tokens"]:
            raise ValueError(f"{uri}: epoch {epoch} token totals differ from the manifest")
        epochs[epoch] = {
            "replay": replay,
            "agentic": agentic,
            "supervised": dict(supervised),
            "context": dict(context),
            "by_trial": by_trial,
        }
    report = {
        "prefix": uri,
        "arm": manifest["arm"],
        "B_supervised_tokens": manifest["B_supervised_tokens"],
        "epochs": {
            e: {
                "rows": manifest["splits"][f"epoch-{e}"]["rows"],
                "padded_rows": manifest["splits"][f"epoch-{e}"]["padded_rows"],
                "steps": manifest["splits"][f"epoch-{e}"]["steps"],
                "supervised_tokens": epochs[e]["supervised"],
                "rows_sha256": manifest["splits"][f"epoch-{e}"]["rows_sha256"],
            }
            for e in epochs
        },
        "heldout_rows": 0,
    }
    selection = [json.loads(line) for line in files["selection.jsonl"].decode().splitlines()]
    return ArmCheck(manifest, index, epochs, report, selection)


def check_recovery(uri: str, manifest: dict, index: list[dict], arm: dict, plain_manifest: dict, plain: dict) -> dict:
    """Recovery: plain's rows with masked injections. Same trials and B as plain; every row trains exactly plain's
    target tokens; a row not marked injected is byte-identical to plain's; an injected row only adds context."""
    b = plain_manifest["B_supervised_tokens"]
    if manifest["B_supervised_tokens"] != b or "handoff" in manifest["recovery"]:
        raise ValueError(f"{uri}: recovery B differs from plain's or carries handoff rows")
    injected = {item["trial_id"] for item in index if item.get("injected")}
    growth = {}
    for epoch in range(EPOCHS):
        rows, plain_rows = arm[epoch]["by_trial"], plain[epoch]["by_trial"]
        if set(rows) != set(plain_rows):
            raise ValueError(f"{uri}: epoch {epoch} trials differ from plain's")
        if sum(v for family, v in arm[epoch]["supervised"].items() if family != "replay") != b:
            raise ValueError(f"{uri}: epoch {epoch} agentic supervised tokens differ from B {b}")
        for trial, (digest, targets) in rows.items():
            if targets != plain_rows[trial][1]:
                raise ValueError(f"{uri}: epoch {epoch} trial {trial} trains different tokens than plain")
            if (digest == plain_rows[trial][0]) == (trial in injected):
                raise ValueError(f"{uri}: epoch {epoch} trial {trial} injected flag disagrees with its bytes")
        growth[epoch] = {
            family: tokens / plain[epoch]["context"][family] - 1 for family, tokens in arm[epoch]["context"].items()
        }
    families = collections.Counter(item["family"] for item in index if item["epoch"] == 0 and item["family"] != "replay")
    shares = collections.Counter(item["family"] for item in index if item["epoch"] == 0 and item.get("injected"))
    return {
        "injected_share": {family: shares[family] / rows for family, rows in sorted(families.items())},
        "context_growth": growth,
        "per_family": {
            family: {
                key: stats[key] for key in ("rows", "selected", "injected", "reduced_to_fit", "skipped_over_context")
            }
            for family, stats in manifest["recovery"]["families"].items()
        },
    }


def check_relay(uri: str, arm: ArmCheck, plain_manifest: dict) -> dict:
    """Relay arm or component: per-family totals to plain's B at whole-episode granularity, same rows every epoch."""
    budgets = plain_family_b(plain_manifest)
    families = sorted(arm.manifest["relay_families"])
    if arm.manifest["relay_component"] != (families != sorted(budgets)):
        raise ValueError(f"{uri}: relay_component disagrees with its families")
    if (
        not arm.manifest["relay_component"]
        and arm.manifest["B_supervised_tokens"] != plain_manifest["B_supervised_tokens"]
    ):
        raise ValueError(f"{uri}: relay B differs from plain's")
    if not (arm.epochs[0]["agentic"] == arm.epochs[1]["agentic"] == arm.epochs[2]["agentic"]):
        raise ValueError(f"{uri}: relay epochs do not hold the same rows")
    trials = [item["trial_id"] for item in arm.index if item["epoch"] == 0 and item["family"] != "replay"]
    if len(trials) != len(set(trials)) or set(trials) != {item["trial_id"] for item in arm.selection}:
        raise ValueError(f"{uri}: relay trials repeat or differ from the selection")
    supervised = {
        item["trial_id"]: (item["family"], item["supervised_tokens"]) for item in arm.index if item["epoch"] == 0
    }
    report = {}
    for family in families:
        chosen = [supervised[item["trial_id"]][1] for item in arm.selection if supervised[item["trial_id"]][0] == family]
        total, budget = sum(chosen), budgets[family]
        if (
            total != arm.epochs[0]["supervised"].get(family, 0)
            or total != arm.manifest["relay_families"][family]["achieved"]
        ):
            raise ValueError(f"{uri}: {family} relay total differs between rows and manifest")
        if total >= budget and chosen and total - chosen[-1] >= budget:
            raise ValueError(f"{uri}: {family} took episodes past plain's B {budget}")
        pool = arm.manifest["relay_pool"][family]["supervised_tokens"]
        if total < budget and total != pool:
            raise ValueError(f"{uri}: {family} is short of plain's B but did not take its whole pool")
        report[family] = {"plain_b": budget, "achieved": total, "delta": total - budget, "episodes": len(chosen)}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("plain")
    parser.add_argument("others", nargs="*")
    args = parser.parse_args()
    plain_manifest, plain_index, plain, plain_report, _ = check_prefix(args.plain, args.dry_run)
    if plain_manifest["arm"] != "plain":
        raise ValueError("The first prefix must be plain")
    if not (plain[0]["agentic"] == plain[1]["agentic"] == plain[2]["agentic"]):
        raise ValueError("Plain epoch files do not hold the same agentic rows")
    orders = [
        [item.get("trial_id") or item.get("byte_offset") for item in plain_index if item["epoch"] == e]
        for e in range(EPOCHS)
    ]
    if len({tuple(order) for order in orders}) != EPOCHS:
        raise ValueError("Plain epochs share an order")
    reports = [plain_report | {"epochs_same_rows_different_order": True}]
    b = plain_manifest["B_supervised_tokens"]
    components = []
    for uri in args.others:
        check = check_prefix(uri, args.dry_run)
        manifest, index, arm, report = check.manifest, check.index, check.epochs, check.report
        if manifest.get("relay_component"):
            if any(arm[epoch]["replay"] for epoch in range(EPOCHS)):
                raise ValueError(f"{uri}: relay component holds replay rows")
            report["relay"] = check_relay(uri, check, plain_manifest)
            components.append((uri, check))
            reports.append(report)
            continue
        if manifest["replay"]["rows_sha256"] != plain_manifest["replay"]["rows_sha256"]:
            raise ValueError(f"{uri}: replay cut differs from plain's")
        for epoch in range(EPOCHS):
            if arm[epoch]["replay"] != plain[0]["replay"]:
                raise ValueError(f"{uri}: epoch {epoch} replay rows differ from plain's")
        if manifest["arm"] == "rs-multi":
            slots = set(plain_manifest["task_slots"])
            if {item["task_id"] for item in index if item.get("task_id")} - slots:
                raise ValueError(f"{uri}: rows outside plain's task slots")
            deviations = {}
            for epoch in range(EPOCHS):
                agentic = sum(v for family, v in arm[epoch]["supervised"].items() if family != "replay")
                deviations[epoch] = agentic / b - 1
                if abs(deviations[epoch]) > 0.01:
                    raise ValueError(f"{uri}: epoch {epoch} agentic supervised {agentic} is not within 1% of B {b}")
            report["agentic_supervised_vs_B"] = deviations
        elif manifest["arm"] == "relay":
            report["relay"] = check_relay(uri, check, plain_manifest)
            for component_uri, component in components:
                families = set(component.manifest["relay_families"])
                trials = {item["trial_id"] for item in index if item["family"] in families}
                if {t: v for t, v in arm[0]["by_trial"].items() if t in trials} != component.epochs[0]["by_trial"]:
                    raise ValueError(f"{uri}: rows of {sorted(families)} differ from the component {component_uri}")
                report.setdefault("matches_components", []).append(component_uri)
        elif manifest["arm"] == "recovery":
            report["recovery"] = check_recovery(uri, manifest, index, arm, plain_manifest, plain)
        else:
            raise ValueError(f"{uri}: no verification rules for arm {manifest['arm']!r}")
        reports.append(report)
    print("ARMS_VERIFIED " + json.dumps(reports), flush=True)


if __name__ == "__main__":
    main()
