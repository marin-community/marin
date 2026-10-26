# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""TP050: exact and roundoff-equivalent minimizers of all saved TP046 menus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    diagnose_tpp40_whole_response_noisy_pairs_20261015 as prior,
)

BASE, PARENT, INPUT = prior.BASE, prior.PARENT, prior.INPUT
OUTPUT = PARENT / "menu_minimizer_diagnostic"
PRIOR_SHA = "6118777151932aa14149226ccd513d8ff6ecd000c108bd34e204ab4321d382bc"
SEAL_SHA = "24466a9ea9d9e014279edd03801a163424f97dca9f06d8f1b201af53d53eb36e"
CONTRACT = {
    "candidate_id": "TP40-050",
    "scope": "all30_TP046_saved34row_menu_minimizers",
    "heads": 30,
    "menu_rows": 34,
    "set_definitions": ["exact", "roundoff"],
    "exact": "predicted == minimum predicted",
    "roundoff": "abs(predicted-minimum)<=64*float64_epsilon*max(1,abs(minimum))",
    "roundoff_interpretation": "numerical equivalence only; no statistical indifference or gate tolerance",
    "official_order": "prediction, role string, row_id",
    "regrets": "best, worst and uniform-average true regret over each predicted minimizer set",
    "pair_gap": "matched-tied truth minus asymmetric truth for each same-source pair contained in the set",
    "query_failures": "preserve unavailable sets; no reconstructed or replacement means",
    "official_regret_changed": False,
    "new_queries": False,
    "new_fits": False,
    "empirical_outcomes": False,
    "scientific_promotion": False,
    "scientific_gate": None,
}
read, digest, write = prior.read, prior.digest, prior.write


def registration_contents() -> dict[str, Any]:
    if digest(Path(prior.__file__)) != PRIOR_SHA or digest(prior.SEAL) != SEAL_SHA:
        raise ValueError("Adopt the exact TP047 utilities and TP046 result seal")
    sources = {
        str(p): digest(p)
        for p in (
            Path(__file__),
            BASE / "test_tpp40_menu_minimizers_20261015.py",
            OUTPUT / "proposal_v1.md",
            Path(prior.__file__),
        )
    }
    sources.update(prior.inherited_sources())
    inputs = [prior.SEAL, INPUT / "complete.json", prior.AUDIT]
    return {
        "contract": CONTRACT,
        "runtime": prior.pairs.runtime(),
        "source_hashes": sources,
        "input_hashes": {str(p): digest(p) for p in inputs},
    }


def identity() -> dict:
    registration = read(OUTPUT / "registration.json")
    current = registration_contents()
    if registration.get("status") != "registered" or any(registration.get(k) != v for k, v in current.items()):
        raise ValueError("Require exact registered no-refit scope")
    prior.verify_seal(prior.SEAL)
    if read(prior.AUDIT)["status"] != "author_saved_output_replay_passed":
        raise ValueError("Require complete TP046 arithmetic replay")
    if read(INPUT / "complete.json")["completed_heads"] != 30:
        raise ValueError("Require all thirty completed heads")
    return {"registration_sha256": digest(OUTPUT / "registration.json"), **current}


def evaluate_sets(frame: pd.DataFrame) -> tuple[list[dict], list[dict], list[dict]]:
    """Retain every minimizer; compare their true losses on the unchanged menu."""
    predicted, truth = frame.predicted.to_numpy(float), frame.truth.to_numpy(float)
    if not len(frame) or not np.isfinite(np.r_[predicted, truth]).all():
        raise ValueError("The saved finite menu is required")
    labels = list(zip(frame.role, frame.row_id, frame.source_index, strict=True))
    if len(labels) != len(set(labels)):
        raise ValueError("Menu action identities must be unique")
    minimum, optimum = float(np.min(predicted)), float(np.min(truth))
    tolerance = 64 * np.finfo(float).eps * max(1.0, abs(minimum))
    order = sorted(range(len(frame)), key=lambda i: (predicted[i], str(frame.iloc[i].role), str(frame.iloc[i].row_id)))
    selected = order[0]
    records, members, pairs = [], [], []
    for name in CONTRACT["set_definitions"]:
        indices = np.flatnonzero(predicted == minimum if name == "exact" else abs(predicted - minimum) <= tolerance)
        regrets = truth[indices] - optimum
        included = frame.iloc[indices]
        current_pairs = []
        for source, group in included.groupby("source_index", sort=True):
            asymmetric, tied = group.loc[group.role == "asymmetric"], group.loc[group.role == "matched_tied"]
            if len(asymmetric) > 1 or len(tied) > 1:
                raise ValueError("Ambiguous aggregate-matched pair")
            if len(asymmetric) and len(tied):
                a, t = asymmetric.iloc[0], tied.iloc[0]
                if a.row_id != t.row_id:
                    raise ValueError("Matched counterpart row identities differ")
                current_pairs.append(
                    {
                        "set": name,
                        "source_index": int(source),
                        "row_id": str(a.row_id),
                        "true_gap_tied_minus_asymmetric": float(t.truth - a.truth),
                        "predicted_gap_tied_minus_asymmetric": float(t.predicted - a.predicted),
                    }
                )
        records.append(
            {
                "set": name,
                "status": "finite",
                "set_size": len(indices),
                "prediction_minimum": minimum,
                "roundoff_tolerance": tolerance if name == "roundoff" else 0.0,
                "best_true_regret": float(min(regrets)),
                "worst_true_regret": float(max(regrets)),
                "uniform_true_regret": float(np.mean(regrets)),
                "official_selected_role": str(frame.iloc[selected].role),
                "official_selected_row_id": str(frame.iloc[selected].row_id),
                "official_selected_source_index": int(frame.iloc[selected].source_index),
                "official_regret_at_1": float(truth[selected] - optimum),
                "asymmetric_and_matched_tied_pairs": len(current_pairs),
                "role_counts": json.dumps(included.role.value_counts().to_dict(), sort_keys=True),
            }
        )
        members.extend(
            {
                "set": name,
                "menu_index": int(i),
                "role": str(frame.iloc[i].role),
                "row_id": str(frame.iloc[i].row_id),
                "source_index": int(frame.iloc[i].source_index),
                "predicted": float(predicted[i]),
                "truth": float(truth[i]),
                "true_regret": float(truth[i] - optimum),
                "official_selected": bool(i == selected),
            }
            for i in indices
        )
        pairs.extend(current_pairs)
    return records, members, pairs


def unavailable_sets(failure: dict) -> list[dict]:
    return [
        {"set": name, "status": "original_query_failure", "original_failure": json.dumps(failure, sort_keys=True)}
        for name in CONTRACT["set_definitions"]
    ]


def run() -> None:
    frozen = identity()
    if frozen != read(OUTPUT / "execution_identity.json"):
        raise ValueError("Frozen identity changed")
    approval = read(OUTPUT / "execution_approval.json")
    for key, value in {
        "scope": CONTRACT["scope"],
        "registration_sha256": frozen["registration_sha256"],
        "execution_identity_sha256": digest(OUTPUT / "execution_identity.json"),
    }.items():
        if approval.get(key) != value:
            raise ValueError("Require exact saved-menu execution approval")
    complete = OUTPUT / "complete.json"
    if complete.exists():
        old = read(complete)
        if old["identity"] != frozen or any(digest(OUTPUT / p) != sha for p, sha in old["files"].items()):
            raise ValueError("Preserve existing output")
        return
    tables: dict[str, list] = {k: [] for k in ("sets", "members", "paired_gaps")}
    for noise, world, candidate in prior.inventory():
        folder = INPUT / "cells/outer3" / noise / world / candidate
        fitted = read(folder / "fit.json")
        common = {
            "world": world,
            "noise": noise,
            "candidate": candidate,
            "outer_fold": 3,
            "source_fit_path": str(folder / "fit.json"),
            "source_fit_sha256": digest(folder / "fit.json"),
            "selected_mu": fitted["selected_candidate"]["mu"],
            "native_full_fit_reused": fitted["native_full_fit_reused"],
        }
        if (folder / "query_failure.json").exists():
            records, members, pairs = unavailable_sets(read(folder / "query_failure.json")), [], []
            common.update(
                source_query_path=str(folder / "query_failure.json"),
                source_query_sha256=digest(folder / "query_failure.json"),
            )
        else:
            frame = pd.read_csv(folder / "predictions.csv", float_precision="round_trip")
            if len(frame) != CONTRACT["menu_rows"]:
                raise ValueError("Frozen menu size changed")
            records, members, pairs = evaluate_sets(frame)
            score = read(folder / "scores.json")
            for record in records:
                if (
                    record["official_regret_at_1"] != score["regret_at_1"]
                    or record["official_selected_role"] != score["selected_role"]
                    or record["official_selected_row_id"] != score["selected_row_id"]
                ):
                    raise ValueError("Official regret or canonical selected action changed")
            common.update(
                source_query_path=str(folder / "predictions.csv"), source_query_sha256=digest(folder / "predictions.csv")
            )
        for key, values in zip(tables, (records, members, pairs), strict=True):
            tables[key].extend(common | value for value in values)
    if len(tables["sets"]) != 60:
        raise ValueError("All thirty heads and both sets are required")
    files = {}
    for name, rows in tables.items():
        path = OUTPUT / f"{name}.csv"
        value = pd.DataFrame(rows).to_csv(index=False)
        if path.exists() and path.read_text() != value:
            raise ValueError("Preserve prior output table")
        path.write_text(value)
        files[path.name] = digest(path)
    write(
        complete,
        {
            "identity": frozen,
            "files": files,
            "heads": 30,
            "set_records": 60,
            "scientific_gate": None,
            "scientific_promotion": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("draft", "freeze", "run"))
    command = parser.parse_args().command
    if command == "draft":
        write(OUTPUT / "registration.draft.json", {"status": "draft_not_authorized", **registration_contents()})
    elif command == "freeze":
        write(OUTPUT / "execution_identity.json", identity())
    else:
        run()


if __name__ == "__main__":
    main()
