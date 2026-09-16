# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9"]
# ///
"""Audit the saved final metrics without changing paper inputs or run artifacts."""

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
TAG = "paloma/dolma_100_programing_languages-tpp10"
BPB = f"eval/{TAG}/bpb"
LOSS = f"eval/{TAG}/loss"


def main():
    records = json.loads((ROOT / "all_102_final_metric_records.json").read_text())
    counts_path = ROOT.parent.parent / "domain_sweeps/repairs_20260911/population_counts.json"
    counts = json.loads(counts_path.read_text())
    counts_hash = hashlib.sha256(json.dumps(counts, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    population = counts[TAG]
    factor = population["tokens"] / (population["bytes"] * math.log(2))
    assertions = []
    for path in sorted(ROOT.glob("*_previous_checkpoint_audit.json")):
        audit = json.loads(path.read_text())
        assert audit["verified"] and audit["counts_sha256"] == counts_hash
        metrics = audit["metrics"]
        legacy_key = "legacy_survey" if "legacy_survey" in metrics else "legacy_alone"
        difference = abs(metrics[legacy_key][LOSS] * factor - metrics["corrected_expanded"][BPB])
        assert difference < 5e-5
        assertions.append({"run_name": audit["request"]["run_name"], "reconstruction_difference": difference})
    rows = []
    for record in records:
        source = Path(record["source"])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == record["source_sha256"]
        final = record["final"]
        value = final[LOSS] * factor
        assert math.isfinite(value)
        schema = final.get("eval/bpb_schema_version") or 1
        if schema == 2:
            assert abs(value - final[BPB]) < 5e-5
        rows.append({"run_name": record["run_name"], "arm": record["arm"], "percent": record["percent"], "trainer_seed": record["trainer_seed"], "subset_seed": record["subset_seed"], "schema": schema, "reported_bpb": final[BPB], "token_loss": final[LOSS], "consistent_bpb": value})
    assert len(rows) == 102
    grid = sorted({r["percent"] for r in rows})
    curves = []
    for arm, subset in [("target", None), ("unmatched", None)] + [("matched", s) for s in (20260912, 20260913, 20260914)]:
        points = []
        for percent in grid:
            members = [r for r in rows if r["percent"] == percent and ((arm == "matched" and percent == 0 and r["arm"] == "unmatched") or (r["arm"] == arm and r["subset_seed"] == subset))]
            assert len(members) == (1 if arm == "target" else 2)
            points.append({"percent": percent, "reported_bpb": mean(r["reported_bpb"] for r in members), "consistent_bpb": mean(r["consistent_bpb"] for r in members)})
        curves.append({"arm": arm, "subset_seed": subset, "points": points, "selected_percent": min(points, key=lambda p: (p["consistent_bpb"], p["percent"]))["percent"]})
    target = {p["percent"]: p["consistent_bpb"] for p in curves[0]["points"]}
    floor = min(target.values())
    unmatched = curves[1]["selected_percent"]
    comparisons = []
    for curve in curves[2:]:
        matched = curve["selected_percent"]
        comparisons.append({"subset_seed": curve["subset_seed"], "selected_percent": matched, "target_regret": target[matched] - floor, "regret_reduction_percent": 100 * (target[unmatched] - target[matched]) / (target[unmatched] - floor), "matched_percent_over_target_minimum": 100 * (target[matched] - floor) / floor})
    result = {"counts": population, "counts_sha256": counts_hash, "conversion_factor": factor, "schema_counts": dict(Counter(r["schema"] for r in rows)), "schema2_run_names": [r["run_name"] for r in rows if r["schema"] == 2], "prior_checkpoint_verifications": assertions, "curves": curves, "target_minimum_percent": curves[0]["selected_percent"], "target_minimum_bpb": floor, "unmatched_selected_percent": unmatched, "unmatched_target_regret": target[unmatched] - floor, "unmatched_percent_over_target_minimum": 100 * (target[unmatched] - floor) / floor, "matched_comparisons": comparisons, "maximum_schema2_reconstruction_error": max(abs(r["consistent_bpb"] - r["reported_bpb"]) for r in rows if r["schema"] == 2), "scope": "Audit only. Neither original metrics nor paper inputs are changed."}
    (ROOT / "consistent_bpb_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    with (ROOT / "consistent_bpb_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    points = [p for p in curves[0]["points"] if p["percent"] >= 30]
    with plt.rc_context({"font.size": 10, "text.usetex": False}):
        fig, axis = plt.subplots(figsize=(7.5, 3.5), layout="constrained")
        x = [p["percent"] / 100 for p in points]
        axis.plot(x, [p["reported_bpb"] for p in points], "o--", color="#777777", label="Current figure: mixed BPB definitions", markersize=4)
        axis.plot(x, [p["consistent_bpb"] for p in points], "o-", color="#0072B2", label="Consistent BPB from saved token losses", markersize=4)
        best = curves[0]["selected_percent"]
        axis.plot(best / 100, target[best], "*", color="#0072B2", markersize=13)
        axis.annotate(f"Measured minimum: {best}%", xy=(best / 100, target[best]), xytext=(0.62, 0.746), arrowprops={"arrowstyle": "-", "color": "#0072B2"}, color="#0072B2")
        axis.set(xlabel="StarCoder mixture fraction, p", ylabel="Target programming-languages BPB", ylim=(0.74, 0.835), title="The apparent jump comes from mixing two BPB calculations")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, loc="upper center", fontsize=9)
        fig.savefig(ROOT / "target_metric_schema_audit.png", dpi=180)
        plt.close(fig)
    print(json.dumps({k: v for k, v in result.items() if k != "curves"}, indent=2))


if __name__ == "__main__":
    main()
