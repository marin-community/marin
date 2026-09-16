# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pandas", "matplotlib"]
# ///
"""Fit six descriptive MARINER curves using the existing TPP10 implementation.

Run from the repository: PYTHONPATH=. uv run python path/to/fit_mariner.py.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from plot_alternatives import DESIGN, DIRECTORY, MATH, NATIVE, load_curves

from experiments.domain_phase_mix.exploratory.two_phase_many import fit_starcoder_tpp10_mariner_20260911 as fitting

FITS = DIRECTORY / "mariner_fits"
PREFLIGHT = DIRECTORY.parent / "native_closeout_20260912/plots/preflight.json"
SELECTED = [
    (domain, arm, benchmark)
    for domain, benchmark in [
        ("wikipedia", "wikipedia_english"),
        ("wikipedia", "math500"),
        ("finemath_3plus", "math500"),
    ]
    for arm in ["matched", "target"]
]


def main():
    design = json.loads(DESIGN.read_text())
    preflight = json.loads(PREFLIGHT.read_text())
    curves = load_curves()
    FITS.mkdir(exist_ok=True)
    sources = [
        NATIVE,
        MATH,
        DESIGN,
        PREFLIGHT,
        Path(__file__),
        Path(fitting.__file__),
        Path(fitting.models.__file__),
        Path(fitting.registry.__file__),
    ]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    outputs = []
    for domain, arm, benchmark in SELECTED:
        name = f"{domain}_{arm}_{benchmark}"
        output = FITS / f"{name}.json"
        points = curves[domain, arm, benchmark]
        pool_key = f"{domain}/{'matched' if arm == 'matched' else 'parent'}"
        pool_tokens = preflight["data"]["caches"][pool_key]["tokens"]
        sequence_count = design["matched_sequences" if arm == "matched" else "parent_sequences"]
        assert pool_tokens == sequence_count * fitting.SEQUENCE_LENGTH
        horizon = design["models"]["unmatched" if arm == "matched" else "target"]["tokens"]
        scale = horizon / pool_tokens
        x = np.array([p["epochs"] for p in points])
        share = x / scale
        assert share[0] == 0 and np.isclose(share[-1], 1, atol=1e-14)
        share[-1] = 1.0
        identity = {"sources": hashes, "key": [domain, arm, benchmark], "points": points}
        input_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if output.exists():
            existing = json.loads(output.read_text())
            if existing["input_sha256"] == input_hash:
                outputs.append(existing)
                print(f"Reusing {name}", flush=True)
                continue
        curve = fitting.Curve(name, arm, np.array([p["bpb"] for p in points]), "descriptive_single_seed")
        result = fitting.fit_curve(curve, share, design, f"eval/{benchmark}/bpb")
        result.update(
            {
                "domain": domain,
                "benchmark": benchmark,
                "input_sha256": input_hash,
                "observed_epochs": x.tolist(),
                "realized_share": share.tolist(),
                "predicted_minimum_epochs": result["predicted_minimum_share"] * scale,
                "observed_minimum_epochs": points[int(np.argmin(curve.response))]["epochs"],
                "dense_epochs": (np.array(result["dense_share"]) * scale).tolist(),
                "feasible_epoch_range": [0, scale],
                "source_sha256": hashes,
            }
        )
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        outputs.append(result)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in [
                        "curve",
                        "observed_minimum_epochs",
                        "predicted_minimum_epochs",
                        "fit_rmse_bpb",
                        "shape",
                        "ridge",
                    ]
                }
            ),
            flush=True,
        )
    summary = {
        "model_id": fitting.MODEL_ID,
        "curves": outputs,
        "protocol": (
            "Existing two-pool TPP10 MARINER fitter; actual fractions inferred from materialized epochs; "
            "all eight points per curve; leave-one-mixture-out shape/ridge/floor tuning, full refit; "
            "dense scan and basin refinement including both feasible boundaries. "
            "Fold-specific median floor anchor, zero external noise margin."
        ),
        "source_sha256": hashes,
        "scope": (
            "Descriptive within-curve fits. Not independent optimum validation. "
            "No curves forced to have interior minima."
        ),
    }
    (FITS / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
