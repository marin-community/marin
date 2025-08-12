"""Fit hyperparameter scaling rules and emit predicted baseline configs.

This script reads canonical optimizer-sweep results from
`experiments/optimizer_sweep/Analysis/Results`, fits a simple power-law
parameterization for each scalar hyperparameter using the recorded best
configs, and writes both human-readable fit summaries and machine-usable
predicted configs.

Outputs:
- Per-optimizer markdown reports `hyperparameters_fit_{optimizer}.md` that
  include the relative fit error on a held-out largest-data point and the
  predicted values for a 1.2B model at {1, 2, 4, 8}x Chinchilla.
- JSON configs at
  `experiments/optimizer_sweep/Analysis/predicted_baseline_config/{optimizer}/{model_size}/{chinchilla}/config.json`
  containing predicted baseline hyperparameters (produced here for `1.2b`).

Scope:
- Fits use model sizes `130m`, `300m`, `520m` and optimizers
  `adamw`, `nadamw`, `muon`, `soape`. Predictions are emitted for `1.2b` at
  Chinchilla ratios 1, 2, 4, and 8.

Method:
- For each hyperparameter key present in a run's `best_config`, we fit the
  form
      h(model_size, chinchilla) ≈ A * model_size**B * chinchilla**C + D
  via non-linear least squares (`scipy.optimize.curve_fit`), where numeric
  model-size proxies come from `marin.optimizer_sweep.utils_simp.expected_params`.
  Non-scalar values and explicitly skipped keys are ignored.

Inputs and assumptions:
- Results are expected at
  `experiments/optimizer_sweep/Analysis/Results/{optimizer}/{model_size}/{chinchilla}/result.json`
  and must contain a `best_config` mapping of hyperparameter name → value.
- Missing or malformed result files are skipped.
- The base input directory can be changed via `RESULTS_DIR_DEFAULT`.

Usage:
    python experiments/optimizer_sweep/Analysis/hyper_scaling.py
"""

import os
import json

import numpy as np
from scipy.optimize import curve_fit
from marin.optimizer_sweep.utils_simp import expected_params

RESULTS_DIR_DEFAULT = "experiments/optimizer_sweep/Analysis/Results"


model_sizes = ["130m", "300m", "520m"]

optimizers = ["adamw", "nadamw", "muon", "soape"]


def _load_results_payloads(results_dir: str = RESULTS_DIR_DEFAULT) -> dict:
    records = {}
    if not os.path.isdir(results_dir):
        return records
    for optimizer in os.listdir(results_dir):
        opt_dir = os.path.join(results_dir, optimizer)
        if not os.path.isdir(opt_dir):
            continue
        for model_size in os.listdir(opt_dir):
            model_dir = os.path.join(opt_dir, model_size)
            if not os.path.isdir(model_dir):
                continue
            for chin_ratio in os.listdir(model_dir):
                chin_dir = os.path.join(model_dir, chin_ratio)
                if not os.path.isdir(chin_dir):
                    continue
                result_path = os.path.join(chin_dir, "result.json")
                if not os.path.exists(result_path):
                    continue
                try:
                    with open(result_path, "r") as f:
                        payload = json.load(f)
                except Exception:
                    continue
                try:
                    chin_val = int(chin_ratio)
                except Exception:
                    continue
                records[(optimizer.lower(), model_size.lower(), chin_val)] = payload
    return records


actual_list = _load_results_payloads(RESULTS_DIR_DEFAULT)


predicted_configs = {}
# optimal hyperparameters for AdamW
for optimizer_name in optimizers:
    hyperparameters_dict = {}
    for model_size in model_sizes:
        for chinchilla in [1, 2, 4, 8]:
            key_tuple = (optimizer_name, model_size, chinchilla)
            if key_tuple in actual_list and "best_config" in actual_list[key_tuple]:
                hyperparameters_dict[(model_size, chinchilla)] = actual_list[key_tuple]["best_config"]
    if (model_sizes[0], 1) not in hyperparameters_dict:
        print(f"Missing results for {optimizer_name} {model_sizes[0]} chinchilla=1; skipping.")
        continue
    keys = list(hyperparameters_dict[(model_sizes[0], 1)].keys())

    for key in keys:
        # fit a power law that is A * model_size^B * chinchilla^C + D
        x = [(expected_params[model_size], chinchilla) for model_size in model_sizes for chinchilla in [1, 2, 4, 8]]
        y = [
            hyperparameters_dict[(model_size, chinchilla)][key]
            for model_size in model_sizes
            for chinchilla in [1, 2, 4, 8]
        ]
        # fit a power law and print error
        if isinstance(y[-1], float | int):
            baseline = np.mean(y[:-1])
            popt, _ = curve_fit(
                lambda t, A, B, C, D: A * t[:, 0] ** B * t[:, 1] ** C + D,
                x[1:-1],
                y[1:-1],
                p0=[0.0, -0.5, -0.5, baseline],
                maxfev=200000,
            )
            # print error on the last point
            predicted_loss = popt[0] * x[-1][0] ** popt[1] * x[-1][1] ** popt[2] + popt[3]
            error = np.sqrt(np.mean((predicted_loss - y[-1]) ** 2))
            parameter = expected_params["1.2b"]
            for chinchilla in [1, 2, 4, 8]:
                if (optimizer_name, "1.2b", chinchilla) not in predicted_configs:
                    predicted_configs[(optimizer_name, "1.2b", chinchilla)] = {}
                predicted_configs[(optimizer_name, "1.2b", chinchilla)][key] = float(
                    popt[0] * parameter ** popt[1] * chinchilla ** popt[2] + popt[3]
                )

OUTPUT_DIR = "experiments/optimizer_sweep/Analysis/predicted_baseline_config"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Also write per-optimizer/model_size/chinchilla JSONs for convenience
for (opt_name, model_size, chin), config in predicted_configs.items():
    out_dir = os.path.join(OUTPUT_DIR, opt_name, model_size, str(chin))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f_cfg:
        json.dump(config, f_cfg, indent=2)
