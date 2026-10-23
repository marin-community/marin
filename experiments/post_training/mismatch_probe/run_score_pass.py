# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""C2-C5: run the WP4 scoring passes over the dev sequences.

One SkyRL GPU job per subcommand; each boots the Megatron policy worker on one
checkpoint with the production scoring geometry (TP1, PP1, EP8, packing off) and
calls ``probe_score_forward``:

- ``reference``: the checkpoint's top-256 ids and fp32 logprobs per position,
  stored as ``.npz`` for the candidate pass;
- ``candidate``: the checkpoint's logprobs gathered at a stored reference's ids
  (plus the realized token), reduced to ``SequenceKLRow`` parquet;
- ``exact``: both models in one job — full-vocabulary log-softmax rows on the
  seeded 1% position subsample, reduced to ``exact_kl`` and summarized (never
  stored whole);
- ``noise``: a checkpoint scored against its own reference under a different
  batch layout — micro-batch 4 instead of 1, sequence order reversed — giving
  the pre-registered B-vs-B floor.

    python -m experiments.post_training.mismatch_probe.run_score_pass reference \
        --model-path .../models/e-step272 --sequences .../sequences.jsonl \
        --out .../kl-dev/reference-E.npz
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import ray
import torch

from experiments.post_training.mismatch_probe.long_context import scoring_worker
from experiments.post_training.mismatch_probe.scoring_pass import (
    TOP_K,
    build_score_batch,
    candidate_gather_ids,
    exact_rows_from_passes,
    kl_rows_from_passes,
    load_probe_sequences,
    write_rows_parquet,
)

PAD_TOKEN_ID = 0
EXACT_RATE = 0.01
MASTER_SEED = 20261023


def run_score(policy, batch: dict, *, top_k: int | None, micro_batch_size: int = 1) -> tuple[dict, float]:
    """One ``probe_score_forward`` call over the whole batch, CPU tensors back."""
    from skyrl_train.training_batch import TrainingInputBatch  # noqa: PLC0415  # skyrl_train only in a SkyRL runtime

    data = TrainingInputBatch({key: torch.from_numpy(value) for key, value in batch.items() if value is not None})
    data.metadata = {
        "response_length": batch["sequences"].shape[1],
        "probe_micro_batch_size": micro_batch_size,
        "probe_score_top_k": top_k,
    }
    started = time.monotonic()
    output = policy.run_method("mesh", "probe_score_forward", data=data)  # DP shards concatenated in order
    duration = time.monotonic() - started
    return {key: value.numpy() for key, value in output.items()}, duration


def shuffled_batch(batch: dict, order: np.ndarray) -> dict:
    """The batch in ``order``, for the noise floor's layout change."""
    return {key: None if value is None else value[order] for key, value in batch.items()}


def reference_pass(args) -> None:
    sequences = load_probe_sequences(args.sequences)
    batch = build_score_batch(
        sequences,
        pad_token_id=PAD_TOKEN_ID,
        gather_ids=None,
        exact_positions=False,
        exact_rate=0.0,
        master_seed=MASTER_SEED,
    )
    with scoring_worker(args.model_path, num_gpus=args.num_gpus) as policy:
        results, duration = run_score(policy, batch, top_k=TOP_K)
    np.savez(
        args.out,
        topk_ids=results["topk_ids"],
        topk_logprobs=results["topk_logprobs"],
        model_path=args.model_path,
        sequences_path=args.sequences,
        top_k=TOP_K,
        seconds=duration,
    )
    print(f"reference pass for {args.model_path}: {duration:.0f} s -> {args.out}")


def candidate_pass(args) -> None:
    sequences = load_probe_sequences(args.sequences)
    with np.load(args.reference) as stored:
        topk_ids = stored["topk_ids"]
        topk_logprobs = stored["topk_logprobs"]
        reference_id = str(stored["model_path"])
        if str(stored["sequences_path"]) != args.sequences:
            raise ValueError("the reference pass scored different sequences")
    padded_len = topk_ids.shape[1]
    gather = np.stack(
        [
            # The last position predicts no token, so the support covers len - 1 positions.
            candidate_gather_ids(sequence, topk_ids[row, padded_len - len(sequence.token_ids) : -1], padded_len)
            for row, sequence in enumerate(sequences)
        ]
    )
    batch = build_score_batch(
        sequences,
        pad_token_id=PAD_TOKEN_ID,
        gather_ids=gather,
        exact_positions=False,
        exact_rate=0.0,
        master_seed=MASTER_SEED,
    )
    with scoring_worker(args.model_path, num_gpus=args.num_gpus) as policy:
        results, duration = run_score(policy, batch, top_k=None)
    rows = kl_rows_from_passes(
        sequences,
        topk_ids,
        topk_logprobs,
        results["gathered_logprobs"],
        reference_id=reference_id,
        candidate_id=args.model_path,
        candidate_step=args.candidate_step,
    )
    count = write_rows_parquet(rows, args.out)
    kls = np.asarray([row.kl for row in rows])
    sequence_means = {sequence.sample_id: [] for sequence in sequences}
    for row in rows:
        sequence_means[row.sequence_id].append(row.kl)
    print(
        f"candidate pass for {args.model_path}: {count} rows in {duration:.0f} s; "
        f"token-mean KL {kls.mean():.6f} nats/token, "
        f"sequence-mean {np.mean([np.mean(values) for values in sequence_means.values()]):.6f}"
    )


def exact_pass(args) -> None:
    sequences = load_probe_sequences(args.sequences)
    batch = build_score_batch(
        sequences,
        pad_token_id=PAD_TOKEN_ID,
        gather_ids=None,
        exact_positions=True,
        exact_rate=EXACT_RATE,
        master_seed=MASTER_SEED,
    )
    passes = {}
    for label, model_path in (("reference", args.reference_model), ("candidate", args.candidate_model)):
        with scoring_worker(model_path, num_gpus=args.num_gpus) as policy:
            results, duration = run_score(policy, batch, top_k=None)
        passes[label] = (results["exact_logprobs"], results["exact_positions"].tolist())
        print(f"{label} exact pass ({model_path}): {len(results['exact_positions'])} positions in {duration:.0f} s")
    rows = exact_rows_from_passes(sequences, *passes["reference"], *passes["candidate"])
    kls = np.asarray([row[3] for row in rows])
    print(
        f"exact subsample: {len(rows)} positions, mean exact KL {kls.mean():.6f} nats/token, "
        f"p50 {np.percentile(kls, 50):.6f}, p90 {np.percentile(kls, 90):.6f}"
    )
    with open(args.out, "w") as handle:
        json.dump(
            {
                "reference_model": args.reference_model,
                "candidate_model": args.candidate_model,
                "rate": EXACT_RATE,
                "positions": len(rows),
                "mean_exact_kl": float(kls.mean()),
                "p50": float(np.percentile(kls, 50)),
                "p90": float(np.percentile(kls, 90)),
                "rows": rows,
            },
            handle,
            indent=2,
        )


def noise_pass(args) -> None:
    """B against B under a different batch layout: the pre-registered floor."""
    sequences = load_probe_sequences(args.sequences)
    batch = build_score_batch(
        sequences,
        pad_token_id=PAD_TOKEN_ID,
        gather_ids=None,
        exact_positions=False,
        exact_rate=0.0,
        master_seed=MASTER_SEED,
    )
    order = np.arange(len(sequences))[::-1]
    with scoring_worker(args.model_path, num_gpus=args.num_gpus) as policy:
        reference, _duration = run_score(policy, batch, top_k=TOP_K, micro_batch_size=1)
        rescored, duration = run_score(policy, shuffled_batch(batch, order), top_k=TOP_K, micro_batch_size=4)
    # Undo the layout's permutation, then compare the supports position by position.
    rescored = {key: value[order] for key, value in rescored.items()}
    mask = batch["attention_mask"].astype(bool)
    mask[:, 0] = False  # position 0's support belongs to no scored token
    delta = (rescored["topk_logprobs"].astype(np.float64) - reference["topk_logprobs"].astype(np.float64))[mask]
    report = {
        "model_path": args.model_path,
        "layout": "micro_batch 1 vs 4, order reversed",
        "compared_values": int(delta.size),
        "abs_max": float(np.abs(delta).max()),
        "abs_mean": float(np.abs(delta).mean()),
        "rms": float(np.sqrt((delta**2).mean())),
        "seconds": duration,
    }
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    reference = subparsers.add_parser("reference")
    reference.add_argument("--model-path", required=True)
    reference.add_argument("--sequences", required=True)
    reference.add_argument("--out", required=True)
    reference.add_argument("--num-gpus", type=int, default=8)

    candidate = subparsers.add_parser("candidate")
    candidate.add_argument("--model-path", required=True)
    candidate.add_argument("--reference", required=True)
    candidate.add_argument("--sequences", required=True)
    candidate.add_argument("--out", required=True)
    candidate.add_argument("--candidate-step", type=int, default=0)
    candidate.add_argument("--num-gpus", type=int, default=8)

    exact = subparsers.add_parser("exact")
    exact.add_argument("--reference-model", required=True)
    exact.add_argument("--candidate-model", required=True)
    exact.add_argument("--sequences", required=True)
    exact.add_argument("--out", required=True)
    exact.add_argument("--num-gpus", type=int, default=8)

    noise = subparsers.add_parser("noise")
    noise.add_argument("--model-path", required=True)
    noise.add_argument("--sequences", required=True)
    noise.add_argument("--out", required=True)
    noise.add_argument("--num-gpus", type=int, default=8)

    args = parser.parse_args()
    {"reference": reference_pass, "candidate": candidate_pass, "exact": exact_pass, "noise": noise_pass}[args.command](
        args
    )
    ray.shutdown()


if __name__ == "__main__":
    main()
