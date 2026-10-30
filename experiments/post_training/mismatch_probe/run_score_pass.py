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


def run_score(
    policy, batch: dict, *, top_k: int | None, micro_batch_size: int = 1, support_width: int | None = None
) -> tuple[dict, float]:
    """One ``probe_score_forward`` call over the whole batch, CPU tensors back."""
    from skyrl_train.training_batch import TrainingInputBatch  # noqa: PLC0415  # skyrl_train only in a SkyRL runtime

    data = TrainingInputBatch({key: torch.from_numpy(value) for key, value in batch.items() if value is not None})
    data.metadata = {
        "response_length": batch["sequences"].shape[1],
        "probe_micro_batch_size": micro_batch_size,
        "probe_score_top_k": top_k,
        "probe_score_support_width": support_width,
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
        topk_tail_logmass=results["topk_tail_logmass"],
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
        reference_tail_logmass = stored["topk_tail_logmass"]
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
        results, duration = run_score(policy, batch, top_k=None, support_width=TOP_K)
    # Kept beside the rows so the reduction can be rerun without the GPU pass.
    np.savez(
        args.gathered_out,
        gathered_logprobs=results["gathered_logprobs"],
        gathered_tail_logmass=results["gathered_tail_logmass"],
        model_path=args.model_path,
    )
    rows = kl_rows_from_passes(
        sequences,
        topk_ids,
        topk_logprobs,
        reference_tail_logmass,
        results["gathered_logprobs"],
        results["gathered_tail_logmass"],
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
    rows = exact_rows_from_passes(
        sequences, *passes["reference"], *passes["candidate"], padded_len=batch["sequences"].shape[1]
    )
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
    """B against B under changed batch layouts: the pre-registered floor, in KL nats/token.

    B's micro-batch-1 pass in file order supplies the top-K support and tail; B is
    then gathered on that support under each changed layout and reduced exactly as
    the E-vs-B signal is. The layouts separate run-to-run nondeterminism (same
    layout), rank and neighbour placement (order reversed), and micro-batch
    composition (order reversed at micro-batch 4).
    """
    sequences = load_probe_sequences(args.sequences)
    plain = build_score_batch(
        sequences,
        pad_token_id=PAD_TOKEN_ID,
        gather_ids=None,
        exact_positions=False,
        exact_rate=0.0,
        master_seed=MASTER_SEED,
    )
    reversed_order = np.arange(len(sequences))[::-1]
    layouts = {
        "same layout": (np.arange(len(sequences)), 1),
        "order reversed": (reversed_order, 1),
        "order reversed, micro-batch 4": (reversed_order, 4),
    }
    report = {"model_path": args.model_path, "layouts": {}}
    with scoring_worker(args.model_path, num_gpus=args.num_gpus) as policy:
        support, _duration = run_score(policy, plain, top_k=TOP_K, micro_batch_size=1)
        padded_len = support["topk_ids"].shape[1]
        gather = np.stack(
            [
                candidate_gather_ids(
                    sequence, support["topk_ids"][row, padded_len - len(sequence.token_ids) : -1], padded_len
                )
                for row, sequence in enumerate(sequences)
            ]
        )
        gathered_batch = build_score_batch(
            sequences,
            pad_token_id=PAD_TOKEN_ID,
            gather_ids=gather,
            exact_positions=False,
            exact_rate=0.0,
            master_seed=MASTER_SEED,
        )
        for name, (order, micro_batch_size) in layouts.items():
            rescored, duration = run_score(
                policy,
                shuffled_batch(gathered_batch, order),
                top_k=None,
                micro_batch_size=micro_batch_size,
                support_width=TOP_K,
            )
            # order is a permutation that is its own inverse (identity or reversal).
            rescored = {key: value[order] for key, value in rescored.items()}
            rows = kl_rows_from_passes(
                sequences,
                support["topk_ids"],
                support["topk_logprobs"],
                support["topk_tail_logmass"],
                rescored["gathered_logprobs"],
                rescored["gathered_tail_logmass"],
                reference_id=f"{args.model_path} micro-batch 1",
                candidate_id=f"{args.model_path} {name}",
                candidate_step=0,
            )
            kls = np.asarray([row.kl for row in rows])
            per_sequence = {}
            for row in rows:
                per_sequence.setdefault(row.sequence_id, []).append(row.kl)
            report["layouts"][name] = {
                "positions": len(rows),
                "token_mean_kl": float(kls.mean()),
                "p50": float(np.percentile(kls, 50)),
                "p90": float(np.percentile(kls, 90)),
                "max": float(kls.max()),
                "sequence_mean_kl": float(np.mean([np.mean(values) for values in per_sequence.values()])),
                "seconds": duration,
                "per_sequence_mean_kl": {key: float(np.mean(values)) for key, values in per_sequence.items()},
            }
            summary = {key: value for key, value in report["layouts"][name].items() if key != "per_sequence_mean_kl"}
            print(name, json.dumps(summary))
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)


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
    candidate.add_argument("--gathered-out", required=True, help="where to save the gathered logprobs (.npz)")
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
