# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0", "scikit-learn==1.8.0", "lightgbm==4.7.0", "matplotlib"]
# ///
"""Replay the official RegMix notebook cells on the archived Qwen swarm.

Run from the Marin checkout with its installed environment:
uv run --offline --no-sync --with lightgbm==4.7.0 python PATH/rerun_official.py
"""

import ast
import contextlib
import hashlib
import importlib.metadata
import json
import os
import pickle
import sys
from pathlib import Path

# The repo imports two OpenMP runtimes; one active thread avoids their conflict.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_THREAD_LIMIT"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[5]
sys.path.insert(0, str(ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    plot_comparator_proposal_diagnostics_20260909 as paths,
)

MIDPOINT = OUT.parent / "delphi_path_midpoints_3e18_20260912"
TARGETS = ("uncheatable", "table9")
VARIANTS = ("official_objective", "official_components", "prior_only")
BLOCK = 2048


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def frozen_json(path: Path, payload: dict) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        assert path.read_text() == content, f"Frozen inputs changed: {path}"
    else:
        path.write_text(content)


def rounded(weights: np.ndarray) -> np.ndarray:
    return paths.proposals.ms.constrained_counts(weights, np.full(len(weights), BLOCK)) / BLOCK


def mean_top128(samples: np.ndarray, values: np.ndarray, source: str, label: str, buckets: tuple[str, ...]) -> np.ndarray:
    namespace = {"np": np, "samples": samples, "simulation": values}
    with (OUT / f"proposal_{label}.log").open("w") as stream, contextlib.redirect_stdout(stream):
        exec(compile(source, "official_notebook_cell_18", "exec"), namespace)
    assert namespace["k"] == 128
    order = np.argsort(values)[:128]
    result = namespace["optimal_data_mixture"]
    assert np.array_equal(result, samples[order].mean(axis=0))
    pd.DataFrame(samples[order], columns=buckets).assign(
        candidate_index=order, prediction=values[order]
    ).to_csv(OUT / f"top128_{label}.csv", index=False)
    return result


def main() -> None:
    notebook = json.loads((OUT / "official_regression.ipynb").read_text())
    fit_source = "".join(notebook["cells"][13]["source"])
    sample_source = "".join(notebook["cells"][16]["source"])
    average_source = "".join(notebook["cells"][18]["source"])
    panel = paths.bench.load_panel(paths.proposals.PANEL)
    BUCKETS = tuple(panel.buckets)
    assert panel.rows == 280 and len(BUCKETS) == 39
    assert list(BUCKETS) == sorted(BUCKETS), "Rounding uses alphabetical bucket order"
    x = panel.features.weights
    assert len(np.unique(x, axis=0)) == len(x)
    anchor = paths.bench.calibration_rows(panel)
    assert anchor.tolist() == [0]
    permutation = np.random.RandomState(42).permutation(np.arange(1, len(x)))
    validation = np.sort(permutation[:93])
    train = np.sort(np.r_[anchor, permutation[93:]])
    assert len(train) == 187 and not set(train).intersection(validation)
    groups = {t: panel.group(t) for t in TARGETS}
    aggregate = np.column_stack([groups[t].outcomes @ groups[t].aggregation_weights for t in TARGETS])
    components = np.column_stack([groups[t].outcomes for t in TARGETS])
    y = np.column_stack([aggregate, components])
    names = [f"objective:{t}" for t in TARGETS]
    names += [f"{t}:{c}" for t in TARGETS for c in groups[t].components]
    bucket_table = pd.read_csv(paths.proposals.ms.DATA / "buckets.csv").set_index("bucket").loc[list(BUCKETS)]
    prior = bucket_table.pool_tokens.to_numpy(float)
    prior /= prior.sum()
    check_prior = 1 / panel.features.inventory
    check_prior /= check_prior.sum()
    assert np.allclose(prior, check_prior, atol=1e-15, rtol=0)
    assert np.isfinite(x).all() and np.isfinite(y).all() and (prior > 0).all()
    frozen_dependencies = {}
    for target in TARGETS:
        receipt = json.loads((MIDPOINT / f"prediction_freeze_{target}.json").read_text())
        directory = Path(receipt["directory"])
        frozen_dependencies[target] = receipt
        for name, sha in receipt["checkpoint_sha256"].items():
            if "_lgbm_" in name or name == f"{target}_mariner.json":
                assert digest(directory / name) == sha, f"Frozen model changed: {name}"
    manifest = {
        "notebook_commit": "dd9d1c3b2d7c1756b1a90f0ad7603068e9856cc6",
        "notebook_sha256": digest(OUT / "official_regression.ipynb"),
        "protocol_sha256": digest(OUT / "PROTOCOL.md"),
        "script_sha256": digest(Path(__file__)),
        "panel_input_hashes": panel.input_hashes,
        "data_hashes": {"weights": array_digest(x), "responses": array_digest(y), "prior": array_digest(prior)},
        "buckets": BUCKETS, "response_names": names,
        "component_weights": {t: groups[t].aggregation_weights.tolist() for t in TARGETS},
        "runs": list(panel.runs), "train_indices": train.tolist(), "validation_indices": validation.tolist(),
        "packages": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm")},
        "python": sys.version,
        "frozen_predictor_fingerprints": {t: frozen_dependencies[t]["fingerprint"] for t in TARGETS},
        "source_adaptations": ["280-mixture dataset and frozen random 2:1 split", "39-bucket corpus token prior", "two scalar objectives and secondary component heads", "downstream 1/2048 runtime rounding"],
    }
    frozen_json(OUT / "input_manifest.json", manifest)
    pd.DataFrame(x, columns=BUCKETS).assign(run=panel.runs, split=np.where(np.isin(np.arange(len(x)), train), "train", "early_stop")).to_csv(OUT / "training_inputs.csv", index=False)
    pd.DataFrame(y, columns=names).assign(run=panel.runs).to_csv(OUT / "training_responses.csv", index=False)
    bucket_table.assign(candidate_concentration=prior).to_csv(OUT / "bucket_prior.csv")
    for index, source in ((13, fit_source), (16, sample_source), (18, average_source)):
        (OUT / f"official_cell_{index}.py").write_text(source)
    model_file = OUT / "official_models.pkl"
    if model_file.exists():
        with model_file.open("rb") as stream:
            predictors = pickle.load(stream)
        print("Reused official fits for the identical frozen inputs", flush=True)
    else:
        namespace = {"np": np, "lgb": lgb, "spearmanr": spearmanr, "X_train": x[train], "X_test": x[validation], "y_train": y[train], "y_test": y[validation], "KEY_METRICS": names}
        print(f"Executing original fitting cell: {len(names)} heads, 187 train / 93 early-stop", flush=True)
        with (OUT / "official_fit.log").open("w") as stream, contextlib.redirect_stdout(stream):
            exec(compile(fit_source, "official_notebook_cell_13", "exec"), namespace)
        predictors = namespace["predictor"]
        temporary = model_file.with_suffix(".tmp")
        temporary.write_bytes(pickle.dumps(predictors, protocol=pickle.HIGHEST_PROTOCOL))
        temporary.replace(model_file)
    assert len(predictors) == 60
    fit_records = []
    for name, reg, response in zip(names, predictors, y.T, strict=True):
        pred = reg.predict(x[validation])
        fit_records.append({"response": name, "best_iteration": reg.best_iteration_, "tree_count": reg.booster_.num_trees(), "early_stop_rmse": float(np.sqrt(np.mean((pred-response[validation])**2))), "early_stop_spearman": float(spearmanr(pred,response[validation]).statistic), "parameters": reg.get_params(), "best_score": reg.best_score_})
    (OUT / "fit_records.json").write_text(json.dumps(fit_records, indent=2) + "\n")
    # Replace only the dataset-specific prior literal in the original candidate cell.
    sample_tree = ast.parse(sample_source)
    for node in sample_tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "prior_dist":
            node.value = ast.List(elts=[ast.Constant(float(v)) for v in prior], ctx=ast.Load())
    ast.fix_missing_locations(sample_tree)
    (OUT / "adapted_candidate_cell.py").write_text(ast.unparse(sample_tree) + "\n")
    namespace = {"np": np}
    exec(compile(sample_tree, "official_notebook_cell_16_prior_substitution", "exec"), namespace)
    samples = namespace["samples"]
    assert samples.shape == (100000, 39)
    assert np.array_equal(samples, np.random.RandomState(42).dirichlet(prior, 100000))
    frozen_json(OUT / "candidate_receipt.json", {"sha256": array_digest(samples), "shape": list(samples.shape), "legacy_seed": 42, "prior_sum": float(prior.sum()), "top_k": 128})
    endpoints = paths.proposal_weights(BUCKETS)
    old_midpoints = pd.read_csv(MIDPOINT / "candidate_weights.csv")
    all_policies, all_predictions, summaries, path_records = [], [], [], []
    offset = 2
    for objective_index, target in enumerate(TARGETS):
        count = len(groups[target].components)
        component_models = predictors[offset:offset+count]
        offset += count
        weights = groups[target].aggregation_weights
        receipt = frozen_dependencies[target]
        directory = Path(receipt["directory"])
        old_fits = [pickle.loads((directory / f"{target}_lgbm_{i:02}.pkl").read_bytes()) for i in range(count)]
        old = paths.proposals.ObservatorySurrogate(panel, "lightgbm_regmix", target, old_fits)
        reference = paths.proposals.ms.ObjectiveFit.from_json(json.loads((directory / f"{target}_mariner.json").read_text()))
        ref_order = [BUCKETS.index(b) for b in reference.buckets]
        def component_predict(query):
            return np.column_stack([m.predict(query) for m in component_models]) @ weights
        functions = {"official_objective": predictors[objective_index].predict, "official_components": component_predict, "prior_only": old.predict, "mariner": lambda query: reference.predict(np.asarray(query)[:, ref_order])}
        old_end = endpoints[target]["lgbm"]
        mariner = endpoints[target]["mariner"]
        mp_rows = old_midpoints.loc[(old_midpoints.target == target) & (old_midpoints.surrogate == "lgbm")].set_index("domain")
        old_mid = mp_rows.loc[list(BUCKETS), "weight"].to_numpy()
        assert np.array_equal(old_mid, rounded((mariner + old_end)/2))
        policies = {"mariner": mariner, "old_endpoint": old_end, "old_midpoint": old_mid}
        for variant in VARIANTS:
            print(f"Scoring {target} / {variant}: {len(samples)} official candidates", flush=True)
            values = functions[variant](samples)
            assert np.isfinite(values).all()
            continuous = mean_top128(samples, values, average_source, f"{target}_{variant}", BUCKETS)
            runtime = rounded(continuous)
            midpoint = rounded((mariner + runtime)/2)
            policies[f"{variant}_continuous"] = continuous
            policies[f"{variant}_endpoint"] = runtime
            policies[f"{variant}_midpoint"] = midpoint
            summaries.append({"target": target, "variant": variant, "endpoint_tv_from_old": float(abs(runtime-old_end).sum()/2), "midpoint_tv_from_old": float(abs(midpoint-old_mid).sum()/2), "endpoint_tv_to_mariner": float(abs(runtime-mariner).sum()/2), "rounding_tv": float(abs(runtime-continuous).sum()/2), "endpoint_same_runtime": bool(np.array_equal(runtime, old_end)), "midpoint_same_runtime": bool(np.array_equal(midpoint,old_mid)), "active_buckets": int(np.count_nonzero(runtime)), "max_epochs": float(np.max(runtime*panel.features.inventory)), "best_candidate_prediction": float(values.min()), "mean_top128_prediction": float(np.sort(values)[:128].mean()), "continuous_prediction": float(functions[variant](continuous[None])[0]), "runtime_prediction": float(functions[variant](runtime[None])[0]), "midpoint_prediction": float(functions[variant](midpoint[None])[0]), "mariner_mixture_prediction": float(functions[variant](mariner[None])[0])})
        for name, mixture in policies.items():
            all_policies.extend({"target": target, "policy": name, "bucket": b, "weight": float(v), "epochs": float(v*e)} for b,v,e in zip(BUCKETS,mixture,panel.features.inventory,strict=True))
        matrix = np.stack(list(policies.values()))
        for model, predict in functions.items():
            for policy, value in zip(policies, predict(matrix), strict=True):
                all_predictions.append({"target": target, "policy": policy, "predictor": model, "prediction": float(value)})
        for end_name in ("old_endpoint", "official_objective_endpoint", "official_components_endpoint"):
            t = np.linspace(0,1,201)
            query = (1-t[:,None])*mariner + t[:,None]*policies[end_name]
            for model,predict in functions.items():
                path_records.extend({"target": target, "path": end_name, "predictor": model, "fraction": float(position), "prediction":float(value)} for position,value in zip(t,predict(query),strict=True))
        pd.DataFrame(summaries).to_csv(OUT / "proposal_comparison.csv", index=False)
        pd.DataFrame(all_policies).to_csv(OUT / "policy_weights.csv", index=False)
        pd.DataFrame(all_predictions).to_csv(OUT / "cross_predictions.csv", index=False)
        pd.DataFrame(path_records).to_csv(OUT / "path_predictions.csv", index=False)
    frozen_json(OUT / "completion.json", {"models_sha256": digest(model_file), "input_manifest_sha256": digest(OUT / "input_manifest.json"), "outputs": {name:digest(OUT/name) for name in ("proposal_comparison.csv","policy_weights.csv","cross_predictions.csv","path_predictions.csv")}, "verified_unmodified_fit_cell": True, "verified_prior_only_candidate_cell_change": True, "verified_unmodified_top128_cell": True})
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
