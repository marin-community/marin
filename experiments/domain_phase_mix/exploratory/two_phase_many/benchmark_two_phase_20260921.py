# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Prepare the asymmetric-only benchmark, replay baselines, and score frozen predictions.

This entry point never submits training or reads remote storage. Preparation
freezes inspected local sources and outcome-independent nested folds. Baseline
fits retain the native response law, head and search grid; only the eligible
population and split contract change. Promotion remains closed without a
separately frozen, populated policy and its required evidence.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4].parent / "mixture-selection"))

import mixture_selection as mariner
import retained_power_law_model_20260728 as rpl
import swarm39_harness_20260725 as legacy
import swarm39_models_20260725 as legacy_models
import two_phase_benchmark_data_20260921 as data
import two_phase_benchmark_scoring_20260921 as scoring

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[3]
PROTOCOL_DIR = REPO / ".agents/projects/two_phase_benchmark_20260921"
DEFAULT_OUTPUT = SCRIPT_DIR / "reference_outputs/two_phase_benchmark_20260921"
BASELINES = (
    "aggregate_mariner_train_median",
    "aggregate_dsp",
    "effective_exposure_dsp",
    "compact_retained_state",
    "hierarchical_phase_replay",
    "retained_power_law",
)
QUICK_TASKS = ("programming_languages_bpb", "github_cpp", "github_python", "c4_en", "falcon-refinedweb")
TARGET_ALIASES = {"programming_languages_bpb": "dolma_100_programing_languages"}
BASELINE_SOURCES = (Path(legacy.__file__), Path(legacy_models.__file__), Path(rpl.__file__))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def code_identity() -> dict[str, str]:
    paths = (
        Path(__file__),
        Path(data.__file__),
        Path(scoring.__file__),
        Path(data.starcoder.__file__),
        Path(data.atomic.__file__),
        Path(mariner.__file__),
        REPO / "uv.lock",
        REPO / "pyproject.toml",
        *BASELINE_SOURCES,
    )
    return {str(path.resolve()): sha256(path) for path in paths}


def environment_identity() -> dict[str, str]:
    return {
        "python": sys.version,
        **{
            package: importlib.metadata.version(package)
            for package in ("numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl")
        },
    }


def verify_hashes(hashes: dict[str, str], root: Path | None = None) -> None:
    for name, expected in hashes.items():
        path = Path(name) if root is None else root / name
        if sha256(path) != expected:
            raise ValueError(f"Frozen artifact changed: {path}")


def receipt(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def valid_receipts(receipts: list[dict[str, str]]) -> bool:
    return bool(receipts) and all(
        Path(item["path"]).is_file() and sha256(Path(item["path"])) == item["sha256"] for item in receipts
    )


def verified_policy(path: Path, evidence_path: Path | None, candidate_model: str) -> dict[str, Any]:
    policy = json.loads(path.read_text())
    receipts = json.loads(evidence_path.read_text()) if evidence_path else policy.get("evidence_receipts", {})
    verified = {}
    for name in policy["required_evidence"]:
        items = receipts.get(name, [])
        verified[name] = False
        if not valid_receipts(items):
            continue
        records = [json.loads(Path(item["path"]).read_text()) for item in items]
        verified[name] = all(
            record.get("evidence") == name
            and record.get("protocol_id") == policy["benchmark_protocol_id"]
            and record.get("candidate_model") == candidate_model
            and record.get("data_contract_sha256") == policy["data_contract_sha256"]
            and record.get("status") == "pass"
            for record in records
        )
    policy["evidence"] = verified
    policy["evidence_receipts"] = receipts
    provenance = policy.get("tolerance_provenance", {})
    if not valid_receipts(provenance.get("artifacts", [])):
        policy["tolerances_frozen"] = False
    if policy.get("data_contract_sha256") != sha256(PROTOCOL_DIR / "data_contract.json") or policy.get(
        "procedure_contract_sha256"
    ) != sha256(PROTOCOL_DIR / "protocol.json"):
        policy["tolerances_frozen"] = False
    return policy


def promotion_inputs(candidate_path: Path, reference_path: Path, policy: dict[str, Any], reference: str) -> pd.DataFrame:
    """Pair one candidate with the authentic frozen reference on required menus."""
    expected = policy["reference_predictions"]
    if receipt(reference_path) != expected or expected not in policy["tolerance_provenance"]["artifacts"]:
        raise ValueError("Reference predictions differ from the frozen baseline receipt")
    candidate = pd.read_csv(candidate_path, keep_default_na=False, float_precision="round_trip")
    if candidate.model.nunique() != 1 or reference in set(candidate.model):
        raise ValueError("Candidate input must contain exactly one model distinct from the reference")
    baseline = pd.read_csv(reference_path, keep_default_na=False, float_precision="round_trip")
    baseline = baseline[baseline.model.eq(reference)]
    columns = ["panel", "target", "repeat"]
    required = pd.DataFrame(policy["required_cells"])[columns].drop_duplicates()
    candidate = candidate.merge(required, on=columns, how="inner", validate="many_to_one")
    baseline = baseline.merge(required, on=columns, how="inner", validate="many_to_one")
    candidate = candidate[candidate.role.isin(["test", "tied_test"])]
    baseline = baseline[baseline.role.isin(["test", "tied_test"])]
    return pd.concat([candidate, baseline], ignore_index=True)


def freeze_screen_policies(prepared: Path, baseline_roots: list[Path], output: Path) -> None:
    """Freeze empirical quick-screen margins from five baseline-only repeats."""
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if sha256(PROTOCOL_DIR / "data_contract.json") != manifest["identity"]["data_contract_sha256"]:
        raise ValueError("Data contract changed after preparation")
    verify_hashes(manifest["sources"])
    verify_hashes(manifest["outputs"], prepared)
    panel_name = "starcoder_wsd80_dense"
    audit_path = prepared / panel_name / "split_audit.json"
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "sufficient" or any(cell["status"] != "sufficient" for cell in audit["cells"]):
        raise ValueError("Quick-screen panel has insufficient frozen menus")
    targets = selected_targets(read_panel(prepared, panel_name), list(QUICK_TASKS), "screen")
    summaries, artifacts = [], [receipt(manifest_path), receipt(audit_path)]
    reference_receipts = {}
    for root in baseline_roots:
        run_path = root / "run.json"
        run = json.loads(run_path.read_text())
        if run["prepared_manifest"] != sha256(manifest_path) or run["claim"] != "baseline_execution_only":
            raise ValueError("Screen calibration needs baseline-only runs on the same prepared manifest")
        verify_hashes(run["outputs"], root)
        predictions = pd.read_csv(root / "predictions.csv", keep_default_na=False, float_precision="round_trip")
        if not set(predictions.model).issubset(BASELINES):
            raise ValueError("New candidate outcomes cannot set screen margins")
        for model in predictions.model.unique():
            if model in reference_receipts:
                raise ValueError(f"Duplicate baseline reference source: {model}")
            reference_receipts[model] = receipt(root / "predictions.csv")
        for _, partition in predictions.groupby(["model", "panel", "target", "repeat"]):
            if sorted(partition.outer_fold.unique()) != list(range(5)):
                raise ValueError("Screen calibration needs all five outer menus in every repeat")
        for _, cell in predictions[predictions.role.eq("test")].groupby(
            ["model", "panel", "target", "repeat", "outer_fold"]
        ):
            if len(cell) < 10 or cell.group_id.nunique() < 2:
                raise ValueError("Quick-screen target has insufficient finite evaluation rows")
        summaries.append(scoring.summarize_predictions(predictions))
        artifacts.extend([receipt(run_path), receipt(root / "predictions.csv")])
    metrics = pd.concat(summaries, ignore_index=True)
    protocol = json.loads((PROTOCOL_DIR / "protocol.json").read_text())
    references = ("effective_exposure_dsp", "aggregate_mariner_train_median")
    for reference in references:
        cells, gates, calibration = [], [], []
        for target in targets:
            roles = ("test", "joint_test") if reference == "effective_exposure_dsp" else ("joint_test",)
            for role in roles:
                selected = metrics[
                    metrics.model.eq(reference)
                    & metrics.panel.eq(panel_name)
                    & metrics.target.eq(target)
                    & metrics.role.eq(role)
                ].sort_values("repeat")
                if selected.repeat.tolist() != list(range(5)):
                    raise ValueError(f"Need exactly five baseline partition repeats: {reference}/{target}/{role}")
                metrics_to_gate = (
                    ["rmse", "frontier_rmse", "frontier_optimism", "selected_optimism", "regret_at3"]
                    if reference == "effective_exposure_dsp"
                    else ["regret_at3", "selected_optimism"]
                )
                if not np.isfinite(selected[["rmse", *metrics_to_gate]].to_numpy(float)).all():
                    raise ValueError("Undefined baseline calibration metric")
                base = selected.iloc[0]
                rules = []
                for metric in metrics_to_gate:
                    spread = float(selected[metric].std(ddof=1))
                    scale = float(base[metric] if metric in ("rmse", "frontier_rmse") else base.rmse)
                    allowance = max(0.05 * scale, spread)
                    if metric in ("selected_optimism", "frontier_optimism"):
                        rules.append({"metric": metric, "absolute_limit": max(0.0, float(base[metric])) + allowance})
                    else:
                        rules.append({"metric": metric, "allowed_increase": allowance})
                    calibration.append(
                        {
                            "target": target,
                            "role": role,
                            "metric": metric,
                            "reference_repeat0": float(base[metric]),
                            "partition_sd": spread,
                            "allowance": allowance,
                        }
                    )
                cell = {"panel": panel_name, "target": target, "role": role, "repeat": 0, "outer_fold": "macro"}
                cells.append(cell)
                gates.append({"panel": panel_name, "target": target, "role": role, "metrics": rules})
        policy = {
            "protocol_id": protocol["protocol_id"] + "/quick_screen/" + reference,
            "benchmark_protocol_id": protocol["protocol_id"],
            "data_contract_sha256": sha256(PROTOCOL_DIR / "data_contract.json"),
            "procedure_contract_sha256": sha256(PROTOCOL_DIR / "protocol.json"),
            "reference_predictions": reference_receipts[reference],
            "reference": reference,
            "aggregation": "fold_macro",
            "tolerances_frozen": True,
            "frozen_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "tolerance_provenance": {
                "kind": "reference_calibration",
                "source": str(output.resolve()),
                "frozen_before_candidate_evaluation": True,
                "artifacts": artifacts,
            },
            "required_cells": cells,
            "gates": gates,
            "required_evidence": [
                "candidate_registration",
                "asymmetric_only_fitting",
                "nested_split_integrity",
                "tied_class_mapping",
            ],
            "evidence_receipts": {},
            "calibration": calibration,
            "fold_sizes": audit["cells"],
            "resolution": "partition sensitivity only; no training-seed certificate",
            "advancement": "Both quick-screen policies required; finite-support and optimizer gates remain separate",
        }
        destination = output / f"{reference}.json"
        if destination.exists():
            existing = json.loads(destination.read_text())
            if {k: v for k, v in existing.items() if k != "frozen_at"} != {
                k: v for k, v in policy.items() if k != "frozen_at"
            }:
                raise ValueError("Frozen screen policy changed; use a new version")
        else:
            write_json(destination, policy)


def target_matches(target: str, selector: str) -> bool:
    if target == selector:
        return True
    parts = target.split("/")
    return (
        len(parts) >= 2
        and parts[-1] == "bpb"
        and parts[-2].removesuffix("-llama3") == TARGET_ALIASES.get(selector, selector)
    )


def selected_targets(panel: data.BenchPanel, selectors: list[str] | None, tier: str) -> list[str]:
    requested = selectors
    if requested is None and tier in ("smoke", "screen"):
        requested = list(QUICK_TASKS) if panel.name.startswith("starcoder") else ["uncheatable_bpb"]
    if requested is None:
        return sorted(panel.targets)
    matches = sorted(target for target in panel.targets if any(target_matches(target, value) for value in requested))
    missing = [value for value in requested if not any(target_matches(target, value) for target in panel.targets)]
    if missing:
        raise ValueError(f"Unavailable requested targets in {panel.name}: {missing}")
    return matches


def prepare(output: Path, names: tuple[str, ...] | None, repeats: int) -> dict[str, Any]:
    protocol_path = PROTOCOL_DIR / "data_contract.json"
    protocol = json.loads(protocol_path.read_text())
    panels = data.load_panels(names)
    identity = {
        "code": code_identity(),
        "environment": environment_identity(),
        "data_contract_sha256": sha256(protocol_path),
        "panels": [p.name for p in panels],
        "repeats": repeats,
    }
    root = output / "prepared"
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["identity"] != identity:
            raise ValueError("Preparation configuration/code changed; use a new output directory")
        verify_hashes(previous["sources"])
        verify_hashes(previous["outputs"], root)
        return previous
    root.mkdir(parents=True, exist_ok=True)
    sources: dict[str, str] = {}
    inventory = []
    audits = []
    for panel in panels:
        sources.update(panel.input_hashes)
        folder = root / panel.name
        folder.mkdir(exist_ok=True)
        panel.rows.to_csv(folder / "rows.csv", index=False)
        pd.DataFrame(panel.targets).to_csv(folder / "outcomes.csv", index=False)
        np.savez_compressed(
            folder / "policies.npz",
            phase0=panel.phase0,
            phase1=panel.phase1,
            c0=panel.c0,
            c1=panel.c1,
            phase0_epochs=panel.phase0_epochs,
            phase1_epochs=panel.phase1_epochs,
        )
        folds = pd.concat(
            [
                data.frozen_folds(panel, protocol["outer_folds"], protocol["inner_folds"], protocol["seed"], repeat)
                for repeat in range(repeats)
            ],
            ignore_index=True,
        )
        folds.to_csv(folder / "folds.csv", index=False)
        audit = data.split_coverage_audit(
            panel,
            folds,
            outer_folds=protocol["outer_folds"],
            inner_folds=protocol["inner_folds"],
            minimum_asymmetric_test_rows=10,
            minimum_asymmetric_test_groups=2,
            expected_group_count=protocol["expected_group_counts"][panel.name],
        )
        if audit["structural_errors"]:
            raise ValueError(f"Invalid frozen splits: {panel.name}: {audit['structural_errors']}")
        write_json(folder / "split_audit.json", audit)
        audits.append(audit)
        counts = {target: int(np.isfinite(values).sum()) for target, values in panel.targets.items()}
        metadata = {
            "name": panel.name,
            "alpha": panel.alpha,
            "buckets": list(panel.buckets),
            "metadata": panel.metadata,
            "targets": counts,
            "source_hashes": panel.input_hashes,
        }
        # Metadata can contain exact exposure arrays retained separately in policies.npz.
        metadata["metadata"] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in panel.metadata.items()}
        write_json(folder / "metadata.json", metadata)
        inventory.append(
            {
                "panel": panel.name,
                "rows": len(panel.rows),
                "asymmetric": int((~panel.rows.tied).sum()),
                "tied": int(panel.rows.tied.sum()),
                "target_count": len(counts),
                "availability": counts,
            }
        )
    write_json(root / "split_audit.json", audits)
    if all(name in identity["panels"] for name in data.panel_names()[5:]):
        write_json(root / "support_transfer_plan.json", data.support_transfer_plan(panels))
    manifest = {
        "identity": identity,
        "sources": sources,
        "inventory": inventory,
        "status": "prepared_development_data",
        "outputs": {
            str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file() and p != manifest_path
        },
    }
    verify_hashes(sources)
    write_json(manifest_path, manifest)
    return manifest


def read_panel(root: Path, name: str) -> data.BenchPanel:
    folder = root / name
    metadata = json.loads((folder / "metadata.json").read_text())
    rows = pd.read_csv(
        folder / "rows.csv",
        keep_default_na=False,
        dtype={"row_id": str, "policy_id": str, "group_id": str},
        float_precision="round_trip",
    )
    rows["tied"] = rows.tied.astype(str).str.lower().eq("true")
    outcomes = pd.read_csv(folder / "outcomes.csv", float_precision="round_trip")
    with np.load(folder / "policies.npz") as arrays:
        extra = dict(metadata["metadata"])
        extra["realized_phase0_epochs"] = arrays["phase0_epochs"]
        extra["realized_phase1_epochs"] = arrays["phase1_epochs"]
        return data.BenchPanel(
            name=name,
            buckets=tuple(metadata["buckets"]),
            phase0=arrays["phase0"],
            phase1=arrays["phase1"],
            alpha=metadata["alpha"],
            c0=arrays["c0"],
            c1=arrays["c1"],
            targets={k: outcomes[k].to_numpy(float) for k in outcomes},
            rows=rows,
            input_hashes=metadata["source_hashes"],
            metadata=extra,
        )


def native_panel(panel: data.BenchPanel, indices: np.ndarray) -> legacy.Panel:
    """Expose exact realized row exposures to legacy policy-only design builders."""
    p0, p1 = panel.phase0[indices], panel.phase1[indices]
    c0 = np.divide(panel.phase0_epochs[indices], p0, out=np.broadcast_to(panel.c0, p0.shape).copy(), where=p0 > 0)
    c1 = np.divide(panel.phase1_epochs[indices], p1, out=np.broadcast_to(panel.c1, p1.shape).copy(), where=p1 > 0)
    rows = panel.rows.iloc[indices]
    families = np.asarray(panel.metadata["family_index"], dtype=int)
    names = tuple(panel.metadata["family_names"])
    return legacy.Panel(
        scale=panel.name,
        split="asymmetric_benchmark",
        alpha=panel.alpha,
        buckets=panel.buckets,
        c0=c0,
        c1=c1,
        family_index=families,
        family_names=names,
        phase0=p0,
        phase1=p1,
        targets={k: v[indices] for k, v in panel.targets.items()},
        series=rows.source.to_numpy(),
        policy_class=np.where(rows.tied, "tied", "asymmetric"),
        group=rows.group_id.to_numpy(),
        row_id=rows.row_id.to_numpy(),
    )


def baseline_model(panel: legacy.Panel, name: str) -> legacy.Model:
    models = {model.name: model for model in legacy_models.observatory_baselines(panel)}
    if name == "aggregate_dsp":
        return dataclasses.replace(
            models["effective_exposure_dsp"],
            name=name,
            shapes=lambda: ({"power": p, "late_multiplier": 1.0} for p in (0.4, 0.7, 1.0)),
        )
    return models[name]


def fit_predict_baseline(
    panel: data.BenchPanel, target: str, name: str, train: np.ndarray, query: np.ndarray, inner: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    if panel.rows.iloc[train].tied.any():
        raise ValueError("Tied outcomes reached a baseline fit")
    observed = panel.targets[target][train]
    if not np.isfinite(observed).all() or len(set(inner)) < 2 or np.any(inner < 0):
        raise ValueError("A fit needs finite asymmetric labels and at least two inner folds")
    splits = tuple((np.flatnonzero(inner != fold), np.flatnonzero(inner == fold)) for fold in sorted(set(inner)))
    if name == "aggregate_mariner_train_median":
        # The standalone's explicit NaN-anchor branch recomputes a median from
        # each actual training partition. No tied reference or noise SD is used.
        anchor = mariner.Anchor(float("nan"), 0.0)
        provisional = mariner.FloorSpec(anchor.proportional, anchor.repeat_sd, mariner.KAPPA_PROVISIONAL)
        shape, ridge, _ = mariner.select_shape_and_ridge(panel.epochs[train], observed, provisional, splits)
        matrix = mariner.design_matrix(panel.epochs[train], shape)
        kappa, flat, score = mariner.search_kappa(matrix, observed, ridge, anchor, splits)
        fitted = mariner.fit_head(matrix, observed, ridge, mariner.FloorSpec(float("nan"), 0.0, kappa))
        return fitted.predict(mariner.design_matrix(panel.epochs[query], shape)), {
            "shape": shape,
            "ridge": ridge,
            "kappa": kappa,
            "flat_profile": flat,
            "inner_rmse": score,
            "floor": fitted.floor,
            "intercept": fitted.intercept,
            "coefficients": fitted.coefficients.tolist(),
            "head": "mariner_train_median_no_repeat_noise",
            "calibration_contract": "asymmetric_only_reference; not the validated proportional-anchor estimator",
        }
    if name == "retained_power_law":
        if not (
            np.allclose(panel.phase0_epochs, panel.phase0 * panel.c0, rtol=1e-9, atol=1e-12)
            and np.allclose(panel.phase1_epochs, panel.phase1 * panel.c1, rtol=1e-9, atol=1e-12)
        ):
            raise ValueError("Native RPL requires constant exposure multipliers; this panel uses discrete row exposures")
        geometry = rpl.Geometry(
            panel.c0,
            panel.c1,
            panel.alpha,
            np.asarray(panel.metadata["family_index"]),
        )
        weights = np.stack([panel.phase0, panel.phase1], axis=1)
        fitted = rpl.fit(weights[train], observed, geometry, splits)
        predicted = fitted.predict(weights[query])
        return predicted, {
            "shape": dataclasses.asdict(fitted.shape),
            "ridge": fitted.ridge,
            "intercept": fitted.intercept,
            "coefficients": fitted.coefficients.tolist(),
            "head": "native_robust_rpl",
        }
    training = native_panel(panel, train)
    testing = native_panel(panel, query)
    model = baseline_model(training, name)
    best: tuple[float, dict[str, Any], float] | None = None
    for shape in model.shapes():
        matrix = model.build(training, shape).matrix
        for ridge in model.l2_grid:
            errors = []
            for fit_rows, held_rows in splits:
                intercept, coef = legacy.fit_head(matrix[fit_rows], observed[fit_rows], ridge)
                errors.extend((intercept + matrix[held_rows] @ coef - observed[held_rows]).tolist())
            score = float(np.mean(np.square(errors)))
            if best is None or score < best[0]:
                best = score, shape, ridge
    if best is None:
        raise ValueError("Empty native baseline search")
    score, shape, ridge = best
    matrix = model.build(training, shape).matrix
    intercept, coef = legacy.fit_head(matrix, observed, ridge)
    predicted = intercept + model.build(testing, shape).matrix @ coef
    return predicted, {
        "shape": shape,
        "ridge": ridge,
        "inner_rmse": float(np.sqrt(score)),
        "intercept": intercept,
        "coefficients": coef.tolist(),
        "head": "native_scaled_nnls",
    }


def run_baselines(args: argparse.Namespace) -> None:
    prepared = args.prepared
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    verify_hashes(manifest["sources"])
    verify_hashes(manifest["outputs"], prepared)
    if manifest["identity"]["code"] != code_identity():
        raise ValueError("Benchmark code changed after preparation; prepare a fresh version")
    if manifest["identity"]["environment"] != environment_identity():
        raise ValueError("Numerical environment changed after preparation; prepare a fresh version")
    protocol = json.loads((PROTOCOL_DIR / "protocol.json").read_text())
    if sha256(PROTOCOL_DIR / "data_contract.json") != manifest["identity"]["data_contract_sha256"]:
        raise ValueError("Data or split contract changed after preparation")
    tier = protocol["tiers"][args.tier]
    panels = args.panels or (
        ["starcoder_wsd80_dense"] if args.tier in ("smoke", "screen") else manifest["identity"]["panels"]
    )
    frames = []
    for name in panels:
        panel = read_panel(prepared, name)
        folds = pd.read_csv(prepared / name / "folds.csv", dtype={"row_id": str})
        index = pd.Series(np.arange(len(panel.rows)), index=panel.rows.row_id)
        for target in selected_targets(panel, args.targets, args.tier):
            for repeat in range(tier["repeats"]):
                for fold in tier["outer_folds_scored"]:
                    table = folds[(folds.repeat == repeat) & (folds.outer_fold == fold)]
                    if table.empty:
                        raise ValueError(f"Missing prepared partition: {name}/{repeat}/{fold}")
                    positions = index.loc[table.row_id].to_numpy(int)
                    table = table.loc[np.isfinite(panel.targets[target][positions])].copy()
                    train_table = table[table.role.eq("train")]
                    query_table = table[table.role.ne("train")]
                    train = index.loc[train_table.row_id].to_numpy(int)
                    query = index.loc[query_table.row_id].to_numpy(int)
                    if len(query) == 0:
                        raise ValueError(f"No finite evaluation rows: {name}/{target}/{fold}")
                    for model in args.models:
                        key = hashlib.sha256(json.dumps([name, target, repeat, fold, model]).encode()).hexdigest()[:20]
                        folder = args.output / "shards" / key
                        signature = {
                            "manifest": sha256(manifest_path),
                            "model": model,
                            "panel": name,
                            "target": target,
                            "repeat": repeat,
                            "outer_fold": fold,
                        }
                        completion = folder / "complete.json"
                        if completion.exists():
                            cached = json.loads(completion.read_text())
                            if cached["identity"] != signature:
                                raise ValueError(f"Stale shard: {folder}")
                            verify_hashes(cached["outputs"], folder)
                            frames.append(
                                pd.read_csv(
                                    folder / "predictions.csv", keep_default_na=False, float_precision="round_trip"
                                )
                            )
                            continue
                        started = time.monotonic()
                        predicted, fit = fit_predict_baseline(
                            panel, target, model, train, query, train_table.inner_fold.to_numpy(int)
                        )
                        if not np.isfinite(predicted).all():
                            raise ValueError(f"Nonfinite baseline predictions: {signature}")
                        frame = panel.rows.iloc[query][["row_id", "policy_id", "group_id", "source", "region"]].copy()
                        frame["model"], frame["panel"], frame["target"] = model, name, target
                        frame["repeat"], frame["outer_fold"] = repeat, fold
                        frame["role"] = query_table.role.to_numpy()
                        frame["observed"], frame["predicted"] = panel.targets[target][query], predicted
                        folder.mkdir(parents=True, exist_ok=True)
                        frame.to_csv(folder / "predictions.csv", index=False)
                        fit["training_row_ids"] = train_table.row_id.tolist()
                        fit["available_training_rows"] = len(train_table)
                        fit["available_scored_rows"] = len(query_table)
                        fit["inner_folds"] = train_table.inner_fold.tolist()
                        fit["runtime_seconds"] = time.monotonic() - started
                        write_json(folder / "fit.json", fit)
                        write_json(
                            completion,
                            {
                                "identity": signature,
                                "outputs": {
                                    p.name: sha256(p) for p in (folder / "predictions.csv", folder / "fit.json")
                                },
                            },
                        )
                        frames.append(frame)
                        print(
                            f"Completed {model} {name} {target} fold={fold}, {fit['runtime_seconds']:.1f} seconds",
                            flush=True,
                        )
    predictions = pd.concat(frames, ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    scoring.score_predictions(predictions).to_csv(args.output / "metrics.csv", index=False)
    scoring.summarize_predictions(predictions).to_csv(args.output / "summary.csv", index=False)
    write_json(
        args.output / "run.json",
        {
            "tier": args.tier,
            "prepared_manifest": sha256(manifest_path),
            "models": args.models,
            "claim": "baseline_execution_only",
            "new_candidates": 0,
            "lm_jobs": 0,
            "outputs": {
                p.name: sha256(p)
                for p in (args.output / "predictions.csv", args.output / "metrics.csv", args.output / "summary.csv")
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prep.add_argument("--panels", nargs="+")
    prep.add_argument("--repeats", type=int, choices=range(1, 6), default=1)
    run = sub.add_parser("baseline")
    run.add_argument("--prepared", type=Path, default=DEFAULT_OUTPUT / "prepared")
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "baseline_smoke")
    run.add_argument("--tier", choices=("smoke", "screen", "certify", "finalist"), default="smoke")
    run.add_argument("--models", nargs="+", choices=BASELINES, default=["aggregate_dsp", "effective_exposure_dsp"])
    run.add_argument("--panels", nargs="+")
    run.add_argument("--targets", nargs="+")
    score = sub.add_parser("score")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    promote = sub.add_parser("promote")
    promote.add_argument("--predictions", type=Path, required=True)
    promote.add_argument("--policy", type=Path, required=True)
    promote.add_argument("--reference", required=True)
    promote.add_argument("--reference-predictions", type=Path, required=True)
    promote.add_argument("--output", type=Path, required=True)
    promote.add_argument("--evidence", type=Path, help="JSON mapping of required evidence names to file/hash receipts")
    freeze = sub.add_parser("freeze-screen")
    freeze.add_argument("--prepared", type=Path, required=True)
    freeze.add_argument("--baselines", nargs="+", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare(args.output, tuple(args.panels) if args.panels else None, args.repeats)
        print(
            json.dumps(
                [{k: v for k, v in item.items() if k != "availability"} for item in manifest["inventory"]], indent=2
            )
        )
    elif args.command == "baseline":
        run_baselines(args)
    elif args.command == "freeze-screen":
        freeze_screen_policies(args.prepared, args.baselines, args.output)
    elif args.command == "score":
        frame = pd.read_csv(args.predictions, keep_default_na=False, float_precision="round_trip")
        args.output.mkdir(parents=True, exist_ok=True)
        scoring.score_predictions(frame).to_csv(args.output / "metrics.csv", index=False)
        scoring.summarize_predictions(frame).to_csv(args.output / "summary.csv", index=False)
    else:
        candidate = pd.read_csv(args.predictions, keep_default_na=False, float_precision="round_trip")
        if candidate.model.nunique() != 1:
            raise ValueError("Candidate predictions must contain exactly one model")
        policy = verified_policy(args.policy, args.evidence, str(candidate.model.iloc[0]))
        frame = promotion_inputs(args.predictions, args.reference_predictions, policy, args.reference)
        policy["candidates"] = [str(candidate.model.iloc[0])]
        verdict = scoring.compare_models(frame, args.reference, policy)
        verdict["receipts"] = {
            "policy": receipt(args.policy),
            "predictions": receipt(args.predictions),
            "reference_predictions": receipt(args.reference_predictions),
            "evidence": policy.get("evidence_receipts", {}),
        }
        write_json(args.output, verdict)


if __name__ == "__main__":
    main()
