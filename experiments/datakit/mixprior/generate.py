# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load a fitted GP artifact and write the next named phase weights."""

import argparse
from pathlib import Path

from experiments.datakit.mixprior.hf import write_candidates
from experiments.datakit.mixprior.search import search
from experiments.datakit.mixprior.train import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Directory containing a fitted GP artifact")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=65_536)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--max-epochs", type=float, default=16.0)
    args = parser.parse_args()
    artifact, model, observed = load_model(args.model)
    weights = search(
        model,
        model.features.available,
        model.features.budgets,
        observed,
        pool_size=args.pool_size,
        batch_size=args.batch_size,
        seed=args.seed,
        max_epochs=args.max_epochs,
    )
    write_candidates(args.output, weights, artifact.swarm_id, artifact.components, artifact.hf_revision)
    print(f"Wrote {len(weights)} mixtures to {args.output}")


if __name__ == "__main__":
    main()
