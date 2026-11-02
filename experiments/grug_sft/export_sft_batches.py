# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export the first optimizer steps' batches of the 10.05 SFT for exact replay (marin-community/marin#10400).

Rebuilds the run's data pipeline (``special_token_full_ab_grouped.py``: mix, store manifest, seed 0, batch
256 at 262,144 tokens) and writes each step's batch with ``replay_data``. Global sequence index
``step * 256 + row`` is the loader's own order. Run it in us-central2, next to the source caches.
"""

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import jax
import numpy as np
from levanter.schedule import BatchSchedule
from rigging.log_setup import configure_logging

from experiments.grug_sft.replay_data import batch_from_examples, example_at, write_step
from experiments.grug_sft.special_token_full_ab_grouped import MIX_PATH, STEPS, STORE_MANIFEST
from experiments.grug_sft.special_token_lr import BATCH, CONTEXT, data_config
from experiments.grug_sft.special_token_train import build_train_dataset

logger = logging.getLogger(__name__)
SEED = 0
WORKERS = 8


def check_round_trip(batch, examples) -> None:
    for row in (0, len(examples) - 1):
        rebuilt, original = example_at(batch, row), examples[row]
        leaves = jax.tree.leaves(rebuilt), jax.tree.leaves(original)
        assert jax.tree.structure(rebuilt) == jax.tree.structure(original)
        for a, b in zip(*leaves, strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, required=True, help="Export optimizer steps [first, first + steps)")
    parser.add_argument("--first", type=int, default=0)
    args = parser.parse_args()

    data, steps = data_config(STEPS, STORE_MANIFEST, mix_path=MIX_PATH)
    assert steps == STEPS
    # Same key derivation as special_token_train._run_grug_local with seed 0 and no data_seed.
    data_key, _ = jax.random.split(jax.random.PRNGKey(SEED), 2)
    dataset = build_train_dataset(
        data, max_seq_len=CONTEXT, batch_schedule=BatchSchedule(BATCH), key=data_key
    ).as_sync_dataset()

    def export(step: int) -> int:
        started = time.perf_counter()
        examples = dataset.get_batch(range(step * BATCH, (step + 1) * BATCH))
        fetched = time.perf_counter()
        batch = batch_from_examples(examples)
        if step == args.first:
            check_round_trip(batch, examples)
        write_step(args.output, step, batch)
        logger.info(
            "step %d: %d loss tokens, %d segments; fetch %.0fs, pack and write %.0fs",
            step,
            int(batch.loss_weight.sum()),
            len(np.unique(batch.segment_ids)),
            fetched - started,
            time.perf_counter() - fetched,
        )
        return step

    with ThreadPoolExecutor(WORKERS) as pool:
        for _ in pool.map(export, range(args.first, args.first + args.steps)):
            pass
    logger.info("exported steps %d..%d to %s", args.first, args.first + args.steps - 1, args.output)


if __name__ == "__main__":
    configure_logging(logging.INFO)
    main()
