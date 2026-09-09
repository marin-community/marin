# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replot audited historical cadence studies without claiming a fixed-age experiment."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ACCEPTED = {"complete", "end_turn", "eos", "stop"}


def values(audit, label):
    rows = audit["vectors"][label]["100"]
    data = {r[1]: int(r[2] == 1 and r[3] in ACCEPTED) for r in rows}
    assert len(data) == len(rows)
    return data


def quantile(groups, q):
    total = sum(int(r["groups"]) for r in groups)
    rank = max(1, math.ceil(q * total))
    seen = 0
    for row in sorted(groups, key=lambda r: r["age"]):
        seen += int(row["groups"])
        if seen >= rank:
            return row["age"]
    raise AssertionError("Empty native age histogram")


def paired(ref, candidate, *, alpha, repetitions=50000):
    assert ref.keys() == candidate.keys()
    ids = sorted(ref)
    diff = np.array([candidate[x] - ref[x] for x in ids], dtype=float)
    rng = np.random.default_rng(20260909)
    draws = np.empty(repetitions)
    for start in range(0, repetitions, 250):
        end = min(start + 250, repetitions)
        draws[start:end] = diff[rng.integers(0, len(ids), (end - start, len(ids)))].mean(1) * 100
    return {
        "delta_pp": float(diff.mean() * 100),
        "interval_pp": np.quantile(draws, [alpha / 2, 1 - alpha / 2]).tolist(),
        "questions": len(ids),
        "candidate_only": int((diff == 1).sum()),
        "reference_only": int((diff == -1).sum()),
    }


def build_report(args: argparse.Namespace) -> None:
    inputs = [
        args.qwen_report,
        args.qwen_audit,
        args.qwen_native,
        args.snowball_report,
        args.snowball_audit,
        args.snowball_native,
    ]
    qreport, qaudit, qnative, sreport, saudit, snative = [json.loads(p.read_text()) for p in inputs]
    source_map = {}
    for entry in args.source_map:
        name, separator, replacement = entry.partition("=")
        if not separator or not name or not replacement:
            raise ValueError("--source-map requires ORIGINAL=LOCAL_PATH")
        source_map[name] = Path(replacement)
    for report, report_path in [(qreport, args.qwen_report), (sreport, args.snowball_report)]:
        assert report["status"] == "PASS"
        for name, digest in report["inputs_sha256"].items():
            source = source_map.get(name, Path(name))
            if not source.is_absolute():
                source = report_path.parent / source
            assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, str(source)
    for audit in [qaudit, saudit]:
        assert audit["clean_end_to_end"] and not audit["errors"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, report, audit, native in [("Qwen", qreport, qaudit, qnative), ("Snowball", sreport, saudit, snative)]:
        for arm in report["arms"]:
            label = arm.get("arm", arm.get("label"))
            key = "qwen-factorial16-" + label if model == "Qwen" else label
            vals = values(audit, key)
            n = native["runs"][label]
            ages = n["ages"] if model == "Qwen" else n["age_groups"]
            completed = arm["contract_completed"] if model == "Qwen" else arm["completed"]
            assert sum(vals.values()) == completed and len(vals) == arm["questions"]
            assert abs(n["core_seconds"] - arm["core_seconds"]) < 1e-8
            assert n.get("consumed_response_tokens", n.get("tokens")) == arm["tokens"]
            c = int(label.split("-")[0][1:]) if model == "Qwen" else int(label.split("-")[2][1:])
            objective = "REG+TIS" if "regular-tis" in label else "REG" if "regular-no-tis" in label else "BC"
            row = {
                "model": model,
                "arm": label,
                "cadence": c,
                "seed": arm["seed"],
                "objective": objective,
                "updates": 100,
                "completed": completed,
                "questions": len(vals),
                "completed_percent": 100 * completed / len(vals),
                "core_seconds": arm["core_seconds"],
                "task_gpu_hours": arm["task_gpu_hours"],
                "consumed_tokens": arm["tokens"],
                "age_p50": quantile(ages, 0.5),
                "age_p95": quantile(ages, 0.95),
                "weight_sync_seconds": n["weight_sync_seconds"],
                "weight_sync_percent_core": 100 * n["weight_sync_seconds"] / n["core_seconds"],
            }
            assert 0 <= row["weight_sync_percent_core"] <= 100
            rows.append(row)

    contrasts = []
    for objective, suffix in [("BC", "behavior"), ("REG", "regular-no-tis"), ("REG+TIS", "regular-tis")]:
        result = paired(
            values(qaudit, "qwen-factorial16-c1-a1-" + suffix),
            values(qaudit, "qwen-factorial16-c4-a3-" + suffix),
            alpha=0.05 / 3,
        )
        contrasts.append(
            {
                "model": "Qwen",
                "objective": objective,
                "cadence": 4,
                "reference_cadence": 1,
                "seed": 17,
                "scope": (
                    "98.333333% paired-question interval; nominal 95% Bonferroni family of three; "
                    "conditional on one seed"
                ),
                **result,
            }
        )
    for seed in [17, 29, 43]:
        result = paired(
            values(saudit, f"snowball-confirm100-c2-a1-s{seed}"),
            values(saudit, f"snowball-confirm100-c4-a3-s{seed}"),
            alpha=0.05,
        )
        contrasts.append(
            {
                "model": "Snowball",
                "objective": "BC",
                "cadence": 4,
                "reference_cadence": 2,
                "seed": seed,
                "scope": "Pointwise 95% paired-question interval conditional on this seed",
                **result,
            }
        )

    metadata = {
        "status": "PASS",
        "scope": (
            "Historical partial cadence study: cadence and permitted age changed together; "
            "no new training; not the fixed-A=4 study"
        ),
        "age_quantile": "Nearest rank of observed native prompt-group counts; not per-token version age",
        "inputs_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "rows": rows,
        "contrasts": contrasts,
        "snowball_mean_delta_pp": sreport["mean_delta_pp"],
        "snowball_seed_t95_pp": sreport["training_seed_t95_interval_pp"],
        "snowball_conditional_question_ci95_pp": sreport["conditional_question_bootstrap95_interval_pp"],
    }
    (args.output_dir / "cadence-partial.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (args.output_dir / "cadence-partial.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"}
    )
    COLORS = ["#176c9a", "#bd5a2c", "#38845a"]
    for model in ["Qwen", "Snowball"]:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
        axs = axes.ravel()
        selected = [r for r in rows if r["model"] == model]
        groups = ["BC", "REG", "REG+TIS"] if model == "Qwen" else [17, 29, 43]
        key = "objective" if model == "Qwen" else "seed"
        metrics = [
            ("completed_percent", "Completed correctness (%)", 1),
            ("core_seconds", "Core time (seconds)", 1),
            ("task_gpu_hours", "Task H100-hours", 1),
            ("consumed_tokens", "Consumed response tokens (millions)", 1e6),
        ]
        for group, color in zip(groups, COLORS, strict=True):
            subset = sorted([r for r in selected if r[key] == group], key=lambda r: r["cadence"])
            x = [r["cadence"] for r in subset]
            label = str(group) if model == "Qwen" else f"Seed {group}"
            for ax, (metric, title, scale) in zip(axs[:4], metrics, strict=True):
                ax.plot(x, [r[metric] / scale for r in subset], "-o", color=color, label=label)
                ax.set_title(title)
            axs[4].plot(x, [r["age_p50"] for r in subset], "-o", color=color, label=label + " p50")
            axs[4].plot(x, [r["age_p95"] for r in subset], "--s", color=color, label=label + " p95")
            axs[5].plot(x, [r["weight_sync_percent_core"] for r in subset], "-o", color=color, label=label)
            contrast = next(c for c in contrasts if c["model"] == model and c[key] == group)
            offset = (groups.index(group) - 1) * 0.065
            d = contrast["delta_pp"]
            lo, hi = contrast["interval_pp"]
            axs[6].errorbar(
                contrast["cadence"] + offset, d, yerr=[[d - lo], [hi - d]], fmt="o", capsize=4, color=color, label=label
            )
        axs[4].set_title("Realised group age (updates)")
        axs[5].set_title("Weight sync / core time (%)")
        axs[6].set_title("Paired completed difference (pp)")
        axs[6].axhline(0, color="#777", linewidth=0.8)
        for ax in axs[:7]:
            ax.set_xlabel("Weight-sync cadence C")
            ax.set_xticks([1, 4] if model == "Qwen" else [2, 4])
            ax.grid(alpha=0.15)
        axs[6].set_xlim((0.6, 4.4) if model == "Qwen" else (1.6, 4.4))
        axs[0].set_ylim(0, 100)
        axs[0].legend(loc="best")
        axs[4].legend(fontsize=8, ncol=2)
        axs[7].axis("off")
        if model == "Qwen":
            note = (
                "One training seed; three objectives.\nPaired intervals: 98.33% each, nominal\n"
                "95% across the three cadence contrasts.\nReference: C1/A1; candidate: C4/A3.\n"
                "128 development questions; reused battery."
            )
        else:
            note = (
                "Three training seeds; behavior clipping.\nPaired per-seed intervals: pointwise 95%.\n"
                f'Mean paired difference: {sreport["mean_delta_pp"]:.2f} pp.\n'
                f'Seed t interval: [{sreport["training_seed_t95_interval_pp"][0]:.2f}, '
                f'{sreport["training_seed_t95_interval_pp"][1]:.2f}] pp.\n'
                f'Question-only interval: [{sreport["conditional_question_bootstrap95_interval_pp"][0]:.2f}, '
                f'{sreport["conditional_question_bootstrap95_interval_pp"][1]:.2f}] pp.\n'
                "Reference: C2/A1; candidate: C4/A3.\n1,191 locked historical questions."
            )
        axs[7].text(
            0,
            1,
            note + "\n\nCadence and age limits vary together.\nNo causal age or equal-quality claim.",
            va="top",
            linespacing=1.6,
        )
        fig.suptitle(f"{model}: historical cadence, quality and training cost", fontsize=19, x=0.5, y=0.98)
        fig.text(
            0.5,
            0.01,
            "100 updates per arm • audited native receipts • "
            "paired-difference intervals are distinct from absolute correctness rates",
            ha="center",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0.045, 1, 0.945))
        for ext in ["png", "pdf"]:
            fig.savefig(args.output_dir / f"{model.lower()}-cadence-partial.{ext}", dpi=170)
        plt.close(fig)
    print("CADENCE_PARTIAL_REPORT_PASS: 12 audited arms; 6 paired contrasts; two tables/plots; input hashes verified")
    print(json.dumps(contrasts, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("qwen-report", "qwen-audit", "qwen-native", "snowball-report", "snowball-audit", "snowball-native"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-map",
        action="append",
        default=[],
        metavar="ORIGINAL=LOCAL_PATH",
        help="Resolve a source named in an audited report at another path; its hash must still match.",
    )
    build_report(parser.parse_args())


if __name__ == "__main__":
    main()
