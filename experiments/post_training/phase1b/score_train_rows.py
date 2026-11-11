# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score teacher KL and chosen-token NLL on fixed Phase 1b SFT rows."""

import argparse

import numpy as np
import torch

from experiments.post_training.mismatch_probe.sequence_kl import binned_kl
from experiments.post_training.phase1b.score_native_responses import (
    TOP_K,
    group_sequences,
    groups,
    load_rows,
    load_support,
    reference_pass,
    response_logits_positions,
    score,
    scoring_worker,
    write_json,
)


def candidate_pass(model_path: str, input_uri: str, support_root: str, output_root: str) -> None:
    rows = load_rows(input_uri)
    with scoring_worker(model_path) as policy:
        for index, indices, real in groups(rows):
            selected = [rows[item] for item in indices[:real]]
            support = load_support(f"{support_root.rstrip('/')}/support-{index:04d}.npz", selected)
            sequences = group_sequences(rows, indices, real)
            padded_len = max(len(sequence.token_ids) for sequence in sequences)
            gather = np.zeros((len(sequences), padded_len, TOP_K + 1), dtype=np.int64)
            offsets = support["offsets"]
            for rank, row in enumerate(selected):
                positions = response_logits_positions(row, padded_len)
                start, stop = offsets[rank : rank + 2]
                gather[rank, positions, :TOP_K] = support["topk_ids"][start:stop]
                gather[rank, positions, TOP_K] = row.response_ids
            results = score(policy, sequences, top_k=None, gather=gather)
            kl_sum = 0.0
            nll_sum = 0.0
            positions_total = 0
            for rank, row in enumerate(selected):
                positions = response_logits_positions(row, padded_len)
                start, stop = offsets[rank : rank + 2]
                kls = binned_kl(
                    torch.from_numpy(support["topk_logprobs"][start:stop].astype(np.float64)),
                    torch.from_numpy(support["topk_tail_logmass"][start:stop].astype(np.float64)),
                    torch.from_numpy(results["gathered_logprobs"][rank, positions, :TOP_K].astype(np.float64)),
                    torch.from_numpy(results["gathered_tail_logmass"][rank, positions].astype(np.float64)),
                ).numpy()
                chosen = results["gathered_logprobs"][rank, positions, TOP_K]
                if not np.all(np.isfinite(kls)) or not np.all(np.isfinite(chosen)):
                    raise ValueError(f"{row.sequence.sample_id}: non-finite KL or chosen-token logprob")
                kl_sum += float(kls.sum())
                nll_sum -= float(chosen.sum())
                positions_total += len(positions)
            write_json(
                f"{output_root.rstrip('/')}/score-{index:04d}.json",
                {
                    "model": model_path,
                    "reference": support_root,
                    "rows": real,
                    "positions": positions_total,
                    "kl_sum": kl_sum,
                    "chosen_nll_sum": nll_sum,
                },
            )
    write_json(
        f"{output_root.rstrip('/')}/candidate.json",
        {"model": model_path, "reference": support_root, "input": input_uri, "rows": len(rows)},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("reference", "candidate"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--support-root")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.mode == "reference":
        reference_pass(args.model_path, args.input, args.out)
        return
    if args.support_root is None:
        parser.error("candidate mode requires --support-root")
    candidate_pass(args.model_path, args.input, args.support_root, args.out)


if __name__ == "__main__":
    main()
