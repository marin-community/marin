# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["gcsfs", "numpy", "scipy"]
# ///
"""Compare final, frozen accuracy reports with same-harness and native BPB."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import fsspec
import numpy as np
from scipy.stats import pearsonr, spearmanr

PLAN_SHA = "2ec4492fd7cd0363284dc2ba24eb1dbca48aa25af34de6d6b1c1ebb730c99a7c"
ROOT = f"gs://marin-us-east5/experiments/mariner_ladder_accuracy_20260914/{PLAN_SHA}"
ROWS = {"proportional": "proportional_1e21-2f1a48", "unimax8": "unimax8_1e21-d685cd"}
FAMILIES = [
    "mmlu",
    "arc_easy",
    "arc_challenge",
    "csqa",
    "hellaswag",
    "winogrande",
    "socialiqa",
    "piqa",
    "sciq",
    "lambada",
    "medmcqa",
]
NORMALIZED = {"arc_easy", "arc_challenge", "hellaswag", "piqa"}


def read_report(fs, name, output):
    root = f"{ROOT}/{ROWS[name]}"
    marker_bytes = fs.cat(root + "/SUCCESS.json")
    marker = json.loads(marker_bytes)
    assert marker["plan_sha256"] == PLAN_SHA
    assert marker["row"]["name"] == ROWS[name]
    reports = {}
    for filename in ("summary_metrics.json", "provenance.json"):
        payload = fs.cat(root + "/" + filename)
        identity = {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        assert identity == marker["artifacts"][filename]
        (output / f"{name}_{filename}").write_bytes(payload)
        reports[filename] = json.loads(payload)
    provenance = reports["provenance.json"]
    assert provenance["plan_sha256"] == PLAN_SHA and provenance["row"] == marker["row"]
    assert provenance["topology"] == {"backend": "tpu", "devices": 4, "processes": 1}
    summary = reports["summary_metrics.json"]
    counts = provenance["expected_task_samples"]
    assert len(counts) == 67 and sum(counts.values()) == 44248
    for task, count in counts.items():
        assert summary["n-samples"][task] == {"original": count, "effective": count}
    (output / f"{name}_SUCCESS.json").write_bytes(marker_bytes)
    return summary, provenance


def correlation(rows, bpb_key):
    selected = [r for r in rows if r[bpb_key] is not None]
    # Positive x and y both mean UniMax improves; task deltas, not absolute task difficulties.
    x = np.array([-r[bpb_key] for r in selected])
    y = np.array([r["delta_accuracy_pp"] for r in selected])
    return {
        "n": len(selected),
        "direction_agreement": int(np.sum(x * y > 0)),
        "ties": int(np.sum(x * y == 0)),
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
    }


def comparison_row(task, metric, summaries, native, counts):
    p, u = [summaries[name]["results"][task] for name in ROWS]
    family = task.removesuffix("_5shot").removesuffix("_0shot")
    if family == "mmlu":
        subjects = [name for name in counts if name.startswith("mmlu_")]
        native_values = []
        for name in ROWS:
            native_values.append(
                sum(
                    native[name]["summary"]["olmo_base_eval/easy_bpb/" + subject.removesuffix("_5shot") + "_rc/bpb"]
                    * counts[subject]
                    for subject in subjects
                )
                / sum(counts[s] for s in subjects)
            )
    else:
        native_task = family + "_rc" if family.startswith("mmlu_") else family
        key = f"olmo_base_eval/easy_bpb/{native_task}/bpb"
        native_values = [native[name]["summary"][key] for name in ROWS]
    bpbs = [r.get("bpb,none") for r in (p, u)]
    return {
        "task": task,
        "accuracy_metric": metric,
        "proportional_accuracy_pct": p[metric] * 100,
        "unimax8_accuracy_pct": u[metric] * 100,
        "delta_accuracy_pp": (u[metric] - p[metric]) * 100,
        "proportional_same_harness_bpb": bpbs[0],
        "unimax8_same_harness_bpb": bpbs[1],
        "delta_same_harness_bpb": bpbs[1] - bpbs[0] if None not in bpbs else None,
        "proportional_native_bpb": native_values[0],
        "unimax8_native_bpb": native_values[1],
        "delta_native_bpb": native_values[1] - native_values[0],
    }


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fs = fsspec.filesystem("gcs")
    reports = {name: read_report(fs, name, args.output) for name in ROWS}
    summaries = {name: value[0] for name, value in reports.items()}
    provenance = {name: value[1] for name, value in reports.items()}
    assert provenance["proportional"]["plan"] == provenance["unimax8"]["plan"]
    for key in ("expected_task_samples", "runtime_versions"):
        assert provenance["proportional"][key] == provenance["unimax8"][key]
    for key in ("configs", "versions", "n-shot"):
        assert summaries["proportional"][key] == summaries["unimax8"][key]
    counts = provenance["proportional"]["expected_task_samples"]
    native_bytes = args.native_summary.read_bytes()
    native = json.loads(native_bytes)
    (args.output / "native_wandb_summaries.json").write_bytes(native_bytes)
    for name in ROWS:
        assert native[name]["config"]["checkpoint_path"] == provenance[name]["row"]["checkpoint_uri"]
    family_rows = []
    for family in FAMILIES:
        task = family + ("_0shot" if family == "lambada" else "_5shot")
        metric = "acc_norm,none" if family in NORMALIZED else "acc,none"
        family_rows.append(comparison_row(task, metric, summaries, native, counts))
    leaf_rows = []
    for task in counts:
        metric = "acc_norm,none" if task.removesuffix("_5shot") in NORMALIZED else "acc,none"
        leaf_rows.append(comparison_row(task, metric, summaries, native, counts))
    write_csv(args.output / "family_comparison.csv", family_rows)
    write_csv(args.output / "leaf_comparison.csv", leaf_rows)

    prefix = "olmo_base_easy/table9/"
    components = [k for k in native["proportional"]["summary"] if k.startswith(prefix)]
    assert len(components) == 51
    component_rows = []
    for key in components:
        p, u = [native[name]["summary"][key] for name in ROWS]
        task = key[len(prefix) : -4]
        group = "qa_and_other"
        if task.startswith("minerva_math_"):
            group = "math"
        elif task.startswith("mt_mbpp_") or task in {"codex_humaneval", "mbpp"}:
            group = "code"
        elif task.startswith("basic_skills_"):
            group = "basic_skills"
        component_rows.append(
            {"task": task, "group": group, "proportional_bpb": p, "unimax8_bpb": u, "delta_bpb": u - p}
        )
    write_csv(args.output / "native_51_components.csv", component_rows)
    decomposition = {}
    for group in ("math", "code", "basic_skills", "qa_and_other"):
        rows = [r for r in component_rows if r["group"] == group]
        decomposition[group] = {
            "n": len(rows),
            "unimax_wins": sum(r["delta_bpb"] < 0 for r in rows),
            "proportional_mean": float(np.mean([r["proportional_bpb"] for r in rows])),
            "unimax8_mean": float(np.mean([r["unimax8_bpb"] for r in rows])),
            "contribution_to_macro_delta": sum(r["delta_bpb"] for r in rows) / 51,
        }
    macro_key = "olmo_base_easy/table9_51_component_macro_bpb"
    for name in ROWS:
        np.testing.assert_allclose(
            np.mean([native[name]["summary"][key] for key in components]), native[name]["summary"][macro_key], rtol=1e-12
        )
    results = {
        "plan_sha256": PLAN_SHA,
        "accuracy_macro_definition": (
            "Descriptive equal mean over 11 families; MMLU has one weight. "
            "acc_norm for ARC Easy/Challenge, HellaSwag, PIQA; acc otherwise. Not a preregistered headline."
        ),
        "proportional_family_macro_pct": float(np.mean([r["proportional_accuracy_pct"] for r in family_rows])),
        "unimax8_family_macro_pct": float(np.mean([r["unimax8_accuracy_pct"] for r in family_rows])),
        "unimax8_family_wins": sum(r["delta_accuracy_pp"] > 0 for r in family_rows),
        "same_harness_family_correlation": correlation(family_rows, "delta_same_harness_bpb"),
        "native_family_correlation": correlation(family_rows, "delta_native_bpb"),
        "same_harness_leaf_correlation": correlation(leaf_rows, "delta_same_harness_bpb"),
        "native_leaf_correlation": correlation(leaf_rows, "delta_native_bpb"),
        "native_macro": {name: native[name]["summary"][macro_key] for name in ROWS},
        "native_decomposition": decomposition,
        "limitations": [
            "One training seed per mixture; seeds differ (660704 vs 660705).",
            "Correlations are descriptive task-level deltas for two models, not population-level model correlations.",
            "Native BPB uses reading-comprehension prompts, 8192 context and FP32; "
            "fresh accuracy uses standard lm-eval, 4096 and BF16.",
            "Only seven accuracy families report same-harness BPB; leaf correlations are dominated by 57 MMLU subjects.",
            "Full per-document artifacts remain in GCS; only bounded summary/provenance files read in this analysis. "
            "Parent completion entails full artifact verification.",
        ],
    }
    (args.output / "analysis.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
