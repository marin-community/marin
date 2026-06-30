"""Plot FP8 ragged-dot speedup (vs tuned bf16) against tokens/expert, one curve per run.

Reads the ``result_json`` / ``summary.json`` payloads emitted by ``orchestrate_fp8_autotune.py``
(each carries ``results[].speedup_vs_bf16_best`` + ``results[].shape_dims``) and overlays one curve
per input, so a set of runs that differ in a single knob (e.g. ``--mosaic-wgrad fp8`` vs ``bf16``)
can be compared as a function of per-expert batch size.

Usage:
    uv run --with matplotlib python lib/levanter/scripts/bench/plot_speedup_vs_tokens.py \
      --series "fp8-wgrad=results/wgrad_tokens_per_expert/wgrad_fp8.json" \
      --series "bf16-wgrad=results/wgrad_tokens_per_expert/wgrad_bf16.json" \
      --out results/wgrad_tokens_per_expert/wgrad_sweep.png

Paths are resolved relative to the current working directory (run from the bench dir, or pass
absolute paths). The data files are the raw orchestrator payloads — commit them alongside the PNG so
the plot is reproducible without rerunning the cluster sweep.
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

_OPERATING_POINT = 1024  # EP4-8 full-run load; the realistic per-expert batch (grug-moe-d2560-real-config)
_POWERS_OF_TWO = [128, 256, 512, 1024, 2048, 4096, 8192]


def _load_curve(path: str) -> list[tuple[int, float, float, float]]:
    """(tokens/expert, speedup median, ci95_low, ci95_high) per shape, sorted by tokens/expert."""
    payload = json.load(open(path))
    results = payload["results"] if isinstance(payload, dict) and "results" in payload else payload
    points = []
    for entry in results:
        dims = entry["shape_dims"]
        speedup = entry.get("speedup_vs_bf16_best")
        if speedup is None:
            continue
        tokens_per_expert = dims["tokens"] // dims["experts"]
        points.append((tokens_per_expert, speedup["median"], speedup["ci95_low"], speedup["ci95_high"]))
    points.sort()
    return points


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--series",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="repeatable; a curve label and the orchestrator result/summary JSON for that run",
    )
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument(
        "--title",
        default="FP8 ragged-dot speedup over tuned bf16 by tokens/expert\n(H100, mixed E4M3xE5M2, best-vs-best tuned)",
    )
    ap.add_argument("--xlabel", default="tokens / expert")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for spec in args.series:
        label, path = spec.split("=", 1)
        points = _load_curve(path)
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        yerr = [[p[1] - p[2] for p in points], [p[3] - p[1] for p in points]]
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=label)

    ax.axhline(1.0, color="gray", ls="--", lw=1, label="bf16 parity")
    ax.axvline(_OPERATING_POINT, color="green", ls=":", lw=1, alpha=0.7, label=f"operating point (~{_OPERATING_POINT})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(_POWERS_OF_TWO)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel("fp8 speedup over tuned bf16 (fwd+bwd, x)")
    ax.set_title(args.title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
