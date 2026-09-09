# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit the mixture GP and write the next named phase weights."""

import argparse
from pathlib import Path

from experiments.datakit.mixprior.hf import load_data, load_hf, write_candidates
from experiments.datakit.mixprior.model import fit
from experiments.datakit.mixprior.objective import fit_objective
from experiments.datakit.mixprior.search import search


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Hugging Face dataset commit")
    parser.add_argument("--swarm", required=True)
    parser.add_argument("--data", type=Path, help="Use a local HF snapshot instead of downloading")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=65_536)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--max-epochs", type=float, default=16.0)
    args = parser.parse_args()
    data = load_data(args.data, args.swarm) if args.data else load_hf(args.revision, args.swarm)
    objective = fit_objective(data)
    values, variances = objective(data.outcomes)
    model = fit(data, values, variances)
    weights = search(
        model,
        data.available_tokens,
        data.phase_budgets,
        data.weights,
        pool_size=args.pool_size,
        batch_size=args.batch_size,
        seed=args.seed,
        max_epochs=args.max_epochs,
    )
    write_candidates(args.output, data, weights, args.revision)
    print(f"Wrote {len(weights)} mixtures to {args.output}")


if __name__ == "__main__":
    main()
