# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Independently audit archived Figure 3 values without changing source artifacts."""

import hashlib
import json
import math
from collections import Counter
from itertools import pairwise
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
LIVE = HERE.parents[1]
REPO = HERE.parents[5]
PAPER = Path(
    "/Users/calvinxu/Library/CloudStorage/GoogleDrive-pinlinxu@stanford.edu/My Drive/Research/Marin/"
    "data_mixing_paper_one_phase"
)
OLD = LIVE / "target_jump_audit_20260912"
CORRECTED = LIVE / "consistent_bpb_20260912"
BUILDER = PAPER / "revision_notes/20260912_outline_figures"
TAG = "paloma/dolma_100_programing_languages-tpp10"
LOSS = f"eval/{TAG}/loss"
BPB = f"eval/{TAG}/bpb"
NEIGHBORHOOD = (50, 55, 60, 65, 70)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        assert set(left) == set(right)
        return [change for key in sorted(left) for change in differences(left[key], right[key], f"{prefix}.{key}")]
    return [] if left == right else [[prefix, left, right]]


def main():
    normalized = read(CORRECTED / "analysis.json")
    provenance = read(CORRECTED / "metric_provenance.json")
    plans = [read(LIVE / name) for name in ("pilot_plan.json", "refinement_plan.json")]
    requests = {row["run_name"]: row for plan in plans for row in plan["runs"]}
    assert len(requests) == 102
    for key in ("design_sha256", "primary_metric", "runtime_versions", "code_sha256"):
        assert plans[0][key] == plans[1][key]
    records = read(OLD / "all_102_final_metric_records.json")
    assert len(records) == len({r["run_name"] for r in records}) == 102
    assert set(requests) == {r["run_name"] for r in records}
    population_path = LIVE.parent / "domain_sweeps/repairs_20260911/population_counts.json"
    populations = read(population_path)
    assert canonical_sha(populations) == normalized["metric_definition"]["population_counts_sha256"]
    population = populations[TAG]
    factor = population["tokens"] / (population["bytes"] * math.log(2))
    source_hash_checks = []
    for name, expected in normalized["sources_sha256"].items():
        path = Path(name)
        actual = sha(path)
        assert actual == expected, path
        source_hash_checks.append({"path": name, "sha256": actual, "matches_recorded_hash": True})
    rows = []
    history = {}
    provenance_by_name = {r["run_name"]: r for r in provenance["rows"]}
    for record in records:
        request = requests[record["run_name"]]
        assert all(record[k] == v for k, v in request.items())
        path = REPO / record["source"]
        assert sha(path) == record["source_sha256"]
        events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        final = [e for e in events if e.get("step") == request["total_steps"] - 1 and BPB in e]
        assert len(final) == record["final_records"] > 0
        assert all(all(e.get(k) == v for k, v in record["final"].items()) for e in final)
        value = final[0][LOSS] * factor
        p = provenance_by_name[record["run_name"]]
        assert value == p["normalized_bpb"]
        assert final[0][LOSS] == p["token_average_loss"]
        assert final[0][BPB] == p["reported_bpb"]
        assert final[0].get("eval/bpb_schema_version") == p["reported_schema"]
        row = {
            **request,
            "step": final[0]["step"],
            "token_loss": final[0][LOSS],
            "normalized_bpb": value,
            "reported_bpb": final[0][BPB],
            "reported_schema": p["reported_schema"],
            "raw_source": str(path),
            "raw_sha256": record["source_sha256"],
            "final_duplicate_count": len(final),
        }
        if p["reported_schema"] == 2:
            assert abs(value - final[0][BPB]) < 5e-5
        rows.append(row)
        if request["arm"] == "target" and request["percent"] in NEIGHBORHOOD:
            by_step = {}
            for e in events:
                if LOSS in e:
                    if e["step"] in by_step:
                        assert e[LOSS] == by_step[e["step"]]["token_loss"]
                    by_step[e["step"]] = {
                        "step": e["step"],
                        "token_loss": e[LOSS],
                        "bpb_from_token_loss": e[LOSS] * factor,
                        "reported_schema": e.get("eval/bpb_schema_version"),
                    }
            history[request["percent"]] = by_step

    curve_checks, local_curves = [], []
    for curve in normalized["curves"]:
        arm, subset = curve["arm"], curve["subset_seed"]
        local = []
        for point in curve["points"]:
            members = [
                r
                for r in rows
                if r["percent"] == point["percent"]
                and (
                    (arm == "matched" and point["percent"] == 0 and r["arm"] == "unmatched")
                    or (r["arm"] == arm and r["subset_seed"] == subset)
                )
            ]
            assert len(members) == (1 if arm == "target" else 2)
            assert {r["run_name"] for r in members} == set(point["run_names"])
            actual = mean(r["normalized_bpb"] for r in members)
            assert abs(actual - point["value"]) < 1e-12
            curve_checks.append(abs(actual - point["value"]))
            if point["percent"] in NEIGHBORHOOD:
                local.append(
                    {
                        "percent": point["percent"],
                        "bpb": actual,
                        "token_loss_mean": mean(r["token_loss"] for r in members),
                        "run_names": point["run_names"],
                    }
                )
        values = {p["percent"]: p["bpb"] for p in local}
        local_curves.append(
            {
                "arm": arm,
                "subset_seed": subset,
                "points": local,
                "adjacent_deltas_bpb": {f"{a}_to_{b}": values[b] - values[a] for a, b in pairwise(NEIGHBORHOOD)},
                "p060_above_p055_p065_chord_bpb": values[60] - (values[55] + values[65]) / 2,
            }
        )

    per_seed = []
    for arm, subset, seed in sorted({(r["arm"], r["subset_seed"], r["trainer_seed"]) for r in rows}):
        selected = {
            r["percent"]: r for r in rows if (r["arm"], r["subset_seed"], r["trainer_seed"]) == (arm, subset, seed)
        }
        values = {p: selected[p]["normalized_bpb"] for p in NEIGHBORHOOD}
        per_seed.append(
            {
                "arm": arm,
                "subset_seed": subset,
                "trainer_seed": seed,
                "p055": values[55],
                "p060": values[60],
                "p065": values[65],
                "p070": values[70],
                "p055_to_p060_bpb": values[60] - values[55],
                "p060_to_p065_bpb": values[65] - values[60],
                "p060_above_chord_bpb": values[60] - (values[55] + values[65]) / 2,
            }
        )

    history_rows = []
    for step in sorted(set.intersection(*(set(x) for x in history.values()))):
        current = {p: history[p][step] for p in NEIGHBORHOOD}
        deviation_loss = current[60]["token_loss"] - (current[55]["token_loss"] + current[65]["token_loss"]) / 2
        history_rows.append(
            {
                "step": step,
                "percent_values": current,
                "p060_above_chord_token_loss": deviation_loss,
                "p060_above_chord_bpb": deviation_loss * factor,
            }
        )

    checkpoints = {r["run_name"]: r for r in read(LIVE / "completion_20260912/verified_endpoints.json")["rows"]}
    pilot_checkpoints = {
        r["run_name"]: r for r in read(LIVE / "pilot_results/target_checkpoint_validation.json")["targets"]
    }
    target_rows = []
    for row in rows:
        if row["arm"] != "target" or row["percent"] not in NEIGHBORHOOD:
            continue
        name = row["run_name"]
        wandb = read(OLD / f"{name}_wandb.json")
        assert wandb["state"] == "finished"
        wandb_loss_difference = wandb["summary"][LOSS] - row["token_loss"]
        # W&B's decimal JSON rounds p050 at the last double-precision digit.
        assert abs(wandb_loss_difference) < 5e-15
        assert wandb["summary"][BPB] == row["reported_bpb"]
        assert wandb["summary"]["global_step"] == row["step"]
        if name in checkpoints:
            saved = checkpoints[name]
            assert saved["artifact_status"] == "SUCCESS" and saved["runtime_verified"]
            cp = saved["checkpoint"]
            assert saved["value"] == row["reported_bpb"]
            assert all(saved[k] == v for k, v in requests[name].items())
        else:
            saved = pilot_checkpoints[name]
            cp = {"path": saved["uri"].removesuffix("/metadata.json"), "metadata": saved["metadata"]}
        assert cp["metadata"]["step"] == 11490 and cp["metadata"]["is_temporary"] is False
        target_rows.append(
            {
                **row,
                "checkpoint": cp,
                "archived_wandb_loss_bpb_step_match": True,
                "archived_wandb_loss_difference": wandb_loss_difference,
            }
        )

    previous_checkpoint_checks = []
    for path in sorted(OLD.glob("*_previous_checkpoint_audit.json")):
        audit = read(path)
        assert audit["verified"] and audit["counts_sha256"] == canonical_sha(populations)
        metrics = audit["metrics"]
        legacy = metrics["legacy_survey" if "legacy_survey" in metrics else "legacy_alone"]
        error = abs(legacy[LOSS] * factor - metrics["corrected_expanded"][BPB])
        assert error < 5e-5
        previous_checkpoint_checks.append(
            {
                "source": str(path),
                "sha256": sha(path),
                "run_name": audit["request"]["run_name"],
                "bpb_reconstruction_error": error,
            }
        )

    baseline_config = read(OLD / "tpp10_target_p070_s20260910_wandb.json")["config"]
    expected_config_diffs = {
        row["percent"]: row["diffs_from_p70"] for row in read(OLD / "config_wandb_target_diffs.json")
    }
    config_checks = []
    for percent in (55, 60, 65):
        current_config = read(OLD / f"tpp10_target_p{percent:03d}_s20260910_wandb.json")["config"]
        changes = differences(baseline_config, current_config)
        assert changes == expected_config_diffs[percent]
        assert len(changes) == 12
        config_checks.append({"percent": percent, "matches_archived_comparison": True, "differences_from_p070": changes})

    paper_input = BUILDER / "data/epoch_matching_analysis.json"
    assert sha(paper_input) == sha(CORRECTED / "analysis.json")
    receipt = read(BUILDER / "figure5_receipt.json")
    assert receipt["source_sha256"] == sha(paper_input)
    assert receipt["builder_sha256"] == sha(BUILDER / "build_epoch_matching.py")
    assert receipt["allocation_source_sha256"] == sha(BUILDER / "data/allocation_audit.json")
    final_hashes = read(PAPER / "revision_notes/20260912_tpp10_consistent_bpb/final_sha256.json")
    paper_hashes = {}
    for name in ("epoch_matching_tpp10.pdf", "epoch_matching_tpp10.png"):
        path = PAPER / "figures" / name
        assert sha(path) == final_hashes[str(path)]
        paper_hashes[str(path)] = sha(path)

    allocations = []
    for a in read(BUILDER / "data/allocation_audit.json")["coordinates"]:
        if a["percent"] not in NEIGHBORHOOD:
            continue
        allocations.append(
            {
                **a,
                "realized_target_fraction": a["target_allocation"]["starcoder"] / sum(a["target_allocation"].values()),
                "realized_proxy_fraction": a["proxy_allocation"]["starcoder"] / sum(a["proxy_allocation"].values()),
            }
        )
    ax = {a["percent"]: a for a in allocations}
    tv = {r["percent"]: r["normalized_bpb"] for r in target_rows}
    blend = (ax[60]["realized_target_fraction"] - ax[55]["realized_target_fraction"]) / (
        ax[65]["realized_target_fraction"] - ax[55]["realized_target_fraction"]
    )
    realized_residual = tv[60] - ((1 - blend) * tv[55] + blend * tv[65])
    output = {
        "scope": (
            "Archived source audit on 2026-09-13; no training, checkpoint evaluation, cloud payload reads, "
            "or canonical mutations."
        ),
        "conclusion": (
            "The target p060 slope change exists in saved token loss and remains under a common BPB factor. "
            "No residual metric, plot-mapping, nominal-allocation, or checked configuration bug identified. "
            "Its causal source is unresolved."
        ),
        "population": population,
        "population_canonical_sha256": canonical_sha(populations),
        "conversion_factor": factor,
        "verified_raw_artifacts": len(rows),
        "verified_curve_means": len(curve_checks),
        "max_curve_mean_error": max(curve_checks),
        "reported_schema_counts": dict(Counter(str(r["reported_schema"]) for r in rows)),
        "max_schema2_bpb_reconstruction_error": max(
            abs(r["normalized_bpb"] - r["reported_bpb"]) for r in rows if r["reported_schema"] == 2
        ),
        "current_paper_hashes": paper_hashes,
        "paper_input_matches_corrected_analysis": True,
        "paper_builder_and_allocation_match_receipt": True,
        "source_hash_checks": source_hash_checks,
        "local_curves": local_curves,
        "target_neighborhood": sorted(target_rows, key=lambda r: r["percent"]),
        "per_seed_neighborhood": per_seed,
        "target_training_history": history_rows,
        "configuration_comparisons": config_checks,
        "shared_plan_runtime_versions": plans[0]["runtime_versions"],
        "shared_plan_code_sha256": plans[0]["code_sha256"],
        "allocation_neighborhood": allocations,
        "target_p060_above_chord_using_realized_fractions_bpb": realized_residual,
        "previous_checkpoint_checks": previous_checkpoint_checks,
        "limitations": [
            "One target training seed; no uncertainty interval or repeated endpoint validation near p060.",
            "Two proxy trainer seeds and three matched subsets do not resolve mechanism; "
            "shared data seed/order and finite support can correlate curves.",
            "Checkpoint metadata and archived runtime/configuration receipts were checked; "
            "multi-gigabyte state payloads and bit-identical resume were not replayed.",
            "Three archived checkpoint audits support population normalization but do not freshly "
            "rescore p055/p060/p065/p070.",
            "A positive chord residual measures local shape; linearity is a diagnostic baseline "
            "and is not a predicted true response.",
            "Echo prior-work search returned HTTP 403; no Echo publication attempted. "
            "Canonical data, paper, and ledger were left untouched.",
        ],
    }
    (HERE / "audit.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: output[k]
                for k in (
                    "verified_raw_artifacts",
                    "verified_curve_means",
                    "max_curve_mean_error",
                    "max_schema2_bpb_reconstruction_error",
                    "target_p060_above_chord_using_realized_fractions_bpb",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
