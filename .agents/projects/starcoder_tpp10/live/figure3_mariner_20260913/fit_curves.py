# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pandas", "matplotlib"]
# ///
"""Fit the complete, corrected TPP10 curves with the existing MARINER procedure.

Run from the repository with PYTHONPATH=. uv run python path/to/fit_curves.py.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import fit_starcoder_tpp10_mariner_20260911 as fitting

DIRECTORY = Path(__file__).resolve().parent
REPO = next(p for p in DIRECTORY.parents if (p / "pyproject.toml").exists())
ANALYSIS = DIRECTORY.parent / "consistent_bpb_20260912/analysis.json"
DESIGN = REPO / "experiments/domain_phase_mix/starcoder_tpp10_assets/design.json"
ALLOCATION = DIRECTORY / "sources/allocation_audit.json"


def main():
    source = json.loads(ANALYSIS.read_text())
    design = json.loads(DESIGN.read_text())
    allocation = json.loads(ALLOCATION.read_text())
    assert source["metric_definition"]["id"] == "scored_byte_bpb_from_token_loss_v1"
    assert source["metric_definition"]["total_records"] == 102
    assert source["refinement_complete"] == source["refinement_planned"] == 45
    assert not source["missing_run_names"] and not source["verified_but_unplotted"]
    assert allocation["status"] == "passed"
    design_payload = {k: v for k, v in design.items() if k != "design_sha256"}
    design_digest = hashlib.sha256(
        json.dumps(design_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert allocation["design_sha256"] == design["design_sha256"] == design_digest
    coordinates = {row["percent"]: row for row in allocation["coordinates"]}
    sources = [
        ANALYSIS,
        DESIGN,
        ALLOCATION,
        Path(__file__),
        Path(fitting.__file__),
        Path(fitting.models.__file__),
        Path(fitting.registry.__file__),
    ]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    curves = list(source["curves"])
    matched = [c for c in curves if c["arm"] == "matched"]
    pooled = {
        "arm": "matched",
        "subset_seed": "pooled",
        "points": [
            {"percent": point["percent"], "value": float(np.mean([c["points"][i]["value"] for c in matched]))}
            for i, point in enumerate(matched[0]["points"])
        ],
    }
    curves.append(pooled)
    output = DIRECTORY / "fits"
    output.mkdir(exist_ok=True)
    results = []
    for curve in curves:
        arm = curve["arm"]
        name = arm if arm != "matched" else f"matched_{curve['subset_seed']}"
        points = curve["points"]
        requested = np.array([p["percent"] / 100 for p in points])
        assert len(requested) == 12 and requested[0] == 0 and requested[-1] == 1
        allocations = [
            coordinates[p["percent"]]["target_allocation" if arm == "target" else "proxy_allocation"] for p in points
        ]
        actual = np.array([a["starcoder"] / sum(a.values()) for a in allocations])
        mapping_percent = np.array(sorted(coordinates))
        mapping_allocations = [
            coordinates[p]["target_allocation" if arm == "target" else "proxy_allocation"] for p in mapping_percent
        ]
        mapping_actual = np.array([a["starcoder"] / sum(a.values()) for a in mapping_allocations])
        assert np.all(np.diff(actual) > 0)
        identity = {"hashes": hashes, "curve": curve}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = output / f"{name}.json"
        if path.exists():
            previous = json.loads(path.read_text())
            if previous["input_sha256"] == digest:
                results.append(previous)
                continue
        role = "target_single_seed" if arm == "target" else "two_seed_mean"
        if curve["subset_seed"] == "pooled":
            role = "secondary_pooled_subset_mean"
        data = fitting.Curve(name, arm, np.array([p["value"] for p in points]), role)
        result = fitting.fit_curve(data, actual, design, source["metric_definition"]["metric"])
        result.update(
            {
                "input_sha256": digest,
                "source_sha256": hashes,
                "subset_seed": curve["subset_seed"],
                "requested_share": requested.tolist(),
                "actual_share": actual.tolist(),
                "dense_requested_share": (
                    np.interp(result["dense_share"], mapping_actual, mapping_percent / 100).tolist()
                ),
                "minimum_requested_share": float(
                    np.interp(result["predicted_minimum_share"], mapping_actual, mapping_percent / 100)
                ),
                "minimum_epochs": result["predicted_minimum_share"] * result["nominal_epoch_scales"][1],
            }
        )
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        results.append(result)
        print(
            json.dumps({k: result[k] for k in ["curve", "minimum_requested_share", "minimum_epochs", "fit_rmse_bpb"]}),
            flush=True,
        )
    summary = {
        "model_id": fitting.MODEL_ID,
        "curves": results,
        "source_sha256": hashes,
        "scope": (
            "Descriptive full-curve fits; leave-one-mixture-out tuning. No measured selection or regret is changed."
        ),
        "coordinates": "Fit realized StarCoder fractions; map back to requested fractions for plotting.",
        "floor": "Fold-specific median anchor and zero external noise margin, as in the existing TPP10 fitter.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
