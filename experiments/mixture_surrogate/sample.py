# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI: propose the best data mixtures to sweep next, and export them for the training workstream.

    python experiments/mixture_surrogate/sample.py --budget-tokens 1.037e13 --top-k 50 --out proposals.json

Exports a JSON list of ranked proposals (humaneval bpb), each with its predicted quality (epoch
penalty included), the content/harm split, uncertainty, off-support distance, and the per-phase
weights over the 168 named buckets -- ready to hand to whatever launches the actual training runs.
This scores mixtures over the EXISTING grug buckets; it trains nothing.
"""

import argparse
import json

import numpy as np
from surrogate import MixtureSurrogate


def _weights_to_dict(w: np.ndarray, buckets: list[str], drop_below: float = 1e-4) -> dict:
    """(2, n_buckets) -> {"phase0": {bucket: weight}, "phase1": {...}}, dropping negligible entries."""
    return {
        f"phase{p}": {buckets[b]: round(float(w[p, b]), 6) for b in np.argsort(w[p])[::-1] if w[p, b] >= drop_below}
        for p in range(2)
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--budget-tokens",
        type=float,
        required=True,
        help="REAL token budget the proposed runs will train at; sets the epoch penalty (see README)",
    )
    ap.add_argument("--hidden-dim", type=int, default=512, help="model width the runs will train at")
    ap.add_argument("--n", type=int, default=50000, help="candidate mixtures to sample")
    ap.add_argument("--top-k", type=int, default=50, help="best proposals to export")
    ap.add_argument(
        "--concentration",
        type=float,
        default=200.0,
        help="Dirichlet concentration around the design centre; lower = more exploratory",
    )
    ap.add_argument(
        "--allow-off-support",
        action="store_true",
        help="also propose mixtures beyond the training p95 (higher, poorly-calibrated risk)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="proposals.json")
    args = ap.parse_args()

    m = MixtureSurrogate(target="humaneval_bpb", budget_tokens=args.budget_tokens, hidden_dim=args.hidden_dim)
    prop = m.propose(
        n=args.n,
        concentration=args.concentration,
        top_k=args.top_k,
        in_support_only=not args.allow_off_support,
        seed=args.seed,
    )
    anchor = m.predict(m.anchor())

    out = {
        "target": "humaneval_bpb",
        "units": "bits_per_byte",
        "lower_is_better": True,
        "budget_tokens": args.budget_tokens,
        "hidden_dim": args.hidden_dim,
        "anchor_mean": float(anchor["mean"]),
        "n_sampled": prop["n_sampled"],
        "n_in_support": prop["n_in_support"],
        "buckets": m.buckets,
        "meta": {"gamma": m.gamma, "alpha": m.alpha, "p95_nn_support": m._p95_nn, "issue": m.meta["issue"]},
        "proposals": [
            {
                "rank": i + 1,
                "predicted_mean": round(float(prop["mean"][i]), 6),
                "content_mean": round(float(prop["content_mean"][i]), 6),
                "epoch_harm": round(float(prop["epoch_harm"][i]), 6),
                "predicted_sd": None if prop["sd"] is None else round(float(prop["sd"][i]), 6),
                "nn_distance": round(float(prop["nn_distance"][i]), 6),
                "in_support": bool(prop["in_support"][i]),
                "improvement_vs_anchor": round(float(anchor["mean"] - prop["mean"][i]), 6),
                "weights": _weights_to_dict(prop["weights"][i], m.buckets),
            }
            for i in range(len(prop["mean"]))
        ],
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    best = out["proposals"][0]
    print(
        f"target=humaneval_bpb  budget={args.budget_tokens:.3g}  anchor={out['anchor_mean']:.4f}  "
        f"in-support {prop['n_in_support']}/{prop['n_sampled']}"
    )
    print(
        f"best proposal: {best['predicted_mean']:.4f} "
        f"(content {best['content_mean']:.4f} + harm {best['epoch_harm']:.4f}; "
        f"improvement {best['improvement_vs_anchor']:+.4f}, nn={best['nn_distance']:.3f})"
    )
    print(f"wrote {len(out['proposals'])} proposals -> {args.out}")


if __name__ == "__main__":
    main()
