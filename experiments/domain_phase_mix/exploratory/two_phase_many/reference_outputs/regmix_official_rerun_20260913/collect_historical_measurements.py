# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Collect unchanged RegMix path measurements and frozen predictions offline."""

import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "delphi_path_midpoints_3e18_20260912"
FREEZE = SOURCE / "prediction_freeze"
FINGERPRINT = "78b225d045b6d42ed175dd91d5b06cebecca7e7b04ad07102597fd447809337c"
PAPER = Path("/Users/calvinxu/Library/CloudStorage/GoogleDrive-pinlinxu@stanford.edu/My Drive/Research/Marin/data_mixing_paper_one_phase")
ENDPOINTS = PAPER / "revision_notes/20260912_full_outline_revision/baseline_calibration"
RESULTS = SOURCE / "results_20260913"


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    inputs = [Path(__file__), ENDPOINTS / "selected_measured_runs.csv", ENDPOINTS / "data/path_predictions.csv",
              SOURCE / "exact_midpoints.csv", RESULTS / "midpoint_scored_results.csv",
              RESULTS / "midpoint_scoring_receipt.json", RESULTS / "uncheatable_metric_bridge_audit.json",
              FREEZE / "midpoint_predictions.csv", FREEZE / "summary.json",
              HERE.parent / "delphi_comparator_proposals_3e18_20260909/solutions.csv"]
    endpoint_runs = read_csv(inputs[1])
    path_predictions = read_csv(inputs[2])
    midpoint_weights = [row for row in read_csv(inputs[3]) if row["baseline"] == "lgbm"]
    midpoints = [row for row in read_csv(inputs[4]) if row["comparator"] == "lgbm"]
    scoring = json.loads(inputs[5].read_text())
    assert scoring["pass"] and sha256(inputs[4]) == scoring["scored_csv_sha256"]
    freeze = json.loads(inputs[8].read_text())
    assert sha256(inputs[7]) == freeze["midpoint_predictions_sha256"]
    frozen_predictions = read_csv(inputs[7])
    proposal_weights = read_csv(inputs[9])
    rows = []
    all_weights = []
    for target in ("uncheatable", "table9"):
        selected_weights = [row for row in midpoint_weights if row["target"] == target]
        assert len(selected_weights) == 39 and len({row["domain"] for row in selected_weights}) == 39
        fit_path = FREEZE / FINGERPRINT / target / f"{target}_mariner.json"
        inputs.append(fit_path)
        fit = json.loads(fit_path.read_text())
        buckets = fit["buckets"]
        assert set(buckets) == {row["domain"] for row in selected_weights}
        policy_weights = {}
        for policy, field in (("mariner", "mariner_weight"), ("regmix", "baseline_weight"), ("midpoint", "runtime_weight")):
            weights = {row["domain"]: float(row[field]) for row in selected_weights}
            assert math.isclose(sum(weights.values()), 1, rel_tol=0, abs_tol=1e-12)
            assert all(math.isclose(v * 2048, round(v * 2048), abs_tol=1e-10) for v in weights.values())
            policy_weights[policy] = weights
            all_weights.extend({"target": target, "policy": policy, "bucket": bucket, "weight": weights[bucket]}
                               for bucket in buckets)
        for run in endpoint_runs:
            if run["target"] != target or run["key"] not in ("mariner", "lgbm"):
                continue
            policy = "mariner" if run["key"] == "mariner" else "regmix"
            position = 0.0 if policy == "mariner" else 1.0
            original_path = ENDPOINTS / "data" / run["source_file"]
            inputs.append(original_path)
            originals = [r for r in read_csv(original_path) if r["candidate_id"] == run["candidate_id"]
                         and r["target"] == target and int(r.get("trainer_seed") or 0) == int(run["trainer_seed"])]
            # The plotting CSV passed through pandas' default float parser; its
            # final decimal can differ from the raw CSV by one or two ulps.
            assert len(originals) == 1 and math.isclose(float(originals[0][run["plotted_metric"]]), float(run["plotted_loss"]), rel_tol=0, abs_tol=1e-14)
            predictions = {r["predictor"]: float(r["prediction"]) for r in path_predictions
                           if r["target"] == target and r["comparator"] == "lgbm" and float(r["position"]) == position}
            assert set(predictions) == {"mariner", "lgbm"}
            if policy == "regmix":
                archived = {r["bucket"]: float(r["runtime"]) for r in proposal_weights if r["candidate_id"] == run["candidate_id"]}
                assert archived == policy_weights[policy]
            rows.append(dict(target=target, policy=policy, candidate_id=run["candidate_id"], position=position,
                             trainer_seed=int(run["trainer_seed"]), data_seed=666200 if target == "uncheatable" else 662009,
                             measured_bpb=float(run["plotted_loss"]), mariner_prediction=predictions["mariner"],
                             regmix_prediction=predictions["lgbm"],
                             metric="legacy_frozen_weighted_uncheatable_bpb" if target == "uncheatable" else "native_51_component_macro_bpb",
                             source_uri=run["eval_metrics_uri"], source_file=str(original_path),
                             weights_json=json.dumps(policy_weights[policy], sort_keys=True, separators=(",", ":"))))
        midpoint = [row for row in midpoints if row["target"] == target]
        assert len(midpoint) == 1
        midpoint = midpoint[0]
        for predictor, field in (("mariner", "mariner_prediction"), ("lgbm", "baseline_prediction")):
            archived = [row for row in frozen_predictions if row["candidate_id"] == midpoint["candidate_id"]
                        and row["coordinate"] == "runtime" and row["predictor"] == predictor]
            assert len(archived) == 1 and float(archived[0]["prediction_bpb"]) == float(midpoint[field])
        rows.append(dict(target=target, policy="midpoint", candidate_id=midpoint["candidate_id"], position=0.5,
                         trainer_seed=int(midpoint["trainer_seed"]), data_seed=int(midpoint["data_seed"]),
                         measured_bpb=float(midpoint["measured_bpb"]), mariner_prediction=float(midpoint["mariner_prediction"]),
                         regmix_prediction=float(midpoint["baseline_prediction"]), metric=midpoint["metric"],
                         source_uri=midpoint["source_inline_uri"] if target == "uncheatable" else midpoint["source_native_eval_uri"],
                         source_file=str(inputs[4]), weights_json=json.dumps(policy_weights["midpoint"], sort_keys=True, separators=(",", ":"))))
    assert len(rows) == 10 and len(all_weights) == 234
    write_csv(HERE / "historical_measurements.csv", rows)
    write_csv(HERE / "historical_weights.csv", all_weights)
    receipt = {
        "source_sha256": {str(path): sha256(path) for path in sorted(set(inputs))},
        "output_sha256": {name: sha256(HERE / name) for name in ("historical_measurements.csv", "historical_weights.csv")},
        "rows": len(rows), "distinct_policies": 6, "unchanged_fits_and_measurements": True,
        "endpoint_csv_roundtrip_tolerance_bpb": 1e-14,
        "midpoint_weights": "Actual runtime-rounded weights on 1/2048 grid; not the unrounded midpoint.",
        "uncheatable_metric": "Historical endpoints retain legacy frozen objective BPB; midpoint uses frozen seven-task weights applied to schema-2 pooled component BPB, without empirical offset.",
        "uncheatable_max_observed_estimator_discrepancy_bpb": scoring["max_observed_legacy_estimator_delta_bpb"],
        "uncheatable_exact_legacy_metric": False,
        "midpoint_prediction_fingerprint": FINGERPRINT,
        "mariner_loader": "Add sibling mixture-selection repository to sys.path; import mixture_selection as ms; ms.ObjectiveFit.from_json(json.loads(frozen_json.read_text())).predict(rows reordered to frozen_json['buckets']). No fitting required.",
    }
    (HERE / "historical_measurements_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for row in rows:
        print(row["target"], row["policy"], row["trainer_seed"], row["measured_bpb"], row["mariner_prediction"], row["regmix_prediction"])


if __name__ == "__main__":
    main()
