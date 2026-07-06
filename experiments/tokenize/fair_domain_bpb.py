# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collapse per-domain BPB curves into a domain-fair macro ladder for ``bakeoff_analysis``.

Each arm's BPB at a FLOP point becomes the mean over the shared English+code ``DOMAINS`` only, so a
baseline that eval'd 7 domains and arms that eval'd 11 are compared on identical domains. Reads the
``{arms:{arm:{domain:[[flop,bpb]]}}}`` JSON emitted by ``soak_wandb_ladder.py --domain-curves-out``
and writes ``{arms:{arm:[[flop,macro_bpb]]}}`` for ``bakeoff_analysis.py --bpb``.
"""

import argparse
import json

from experiments.tokenize.soak_wandb_ladder import DOMAINS


def fair_macro_ladder(domain_curves: dict[str, dict[str, list[list[float]]]]) -> dict[str, list[list[float]]]:
    """Mean BPB over ``DOMAINS`` at each FLOP point where every domain logged a value."""
    out: dict[str, list[list[float]]] = {}
    for arm, doms in domain_curves.items():
        by_flop: dict[float, dict[str, float]] = {}
        for domain in DOMAINS:
            for flop, bpb in doms.get(domain, []):
                by_flop.setdefault(flop, {})[domain] = bpb
        curve = [
            [flop, sum(vals[d] for d in DOMAINS) / len(DOMAINS)]
            for flop, vals in sorted(by_flop.items())
            if len(vals) == len(DOMAINS)
        ]
        if curve:
            out[arm] = curve
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--curves", required=True, help="domain-curves JSON from soak_wandb_ladder --domain-curves-out")
    ap.add_argument("--out", required=True, help="fair macro-BPB ladder JSON for bakeoff_analysis --bpb")
    args = ap.parse_args()

    with open(args.curves) as f:
        data = json.load(f)
    ladder = fair_macro_ladder(data.get("arms", data))
    with open(args.out, "w") as f:
        json.dump({"arms": ladder}, f, indent=2)
    for arm, curve in ladder.items():
        print(f"{arm:34s} {len(curve):4d} pts  max {max(p[0] for p in curve):.2e}")


if __name__ == "__main__":
    main()
