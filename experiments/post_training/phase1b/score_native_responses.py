# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score fixed response tokens after model-specific native chat prompts."""

import argparse
import io
import json
from collections import defaultdict
from dataclasses import dataclass, replace

import numpy as np
import torch
from rigging.filesystem.buckets import filesystem_for

from experiments.post_training.mismatch_probe.score_probe_sets import (
    DP_SIZE,
    MAX_LONG_PER_FORWARD,
    PAD_TOKEN_ID,
    TOP_K,
    forward_groups,
    unique_rows,
)
from experiments.post_training.mismatch_probe.scoring_pass import ProbeSequence, build_score_batch, run_score
from experiments.post_training.mismatch_probe.sequence_kl import binned_kl

PADDING_TOKENS = [1, 1]


@dataclass(frozen=True)
class NativeRow:
    sequence: ProbeSequence
    prompt_tokens: int

    @property
    def response_ids(self) -> list[int]:
        return self.sequence.token_ids[self.prompt_tokens :]


def load_rows(uri: str) -> list[NativeRow]:
    rows = []
    filesystem, path = filesystem_for(uri)
    with filesystem.open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            sequence = ProbeSequence(
                sample_id=record["sample_id"],
                token_ids=record["token_ids"],
                token_type_tags=record["token_type_tags"],
                position_tags=record["position_tags"],
                bucket_group=record["bucket_group"],
            )
            prompt_tokens = record["prompt_tokens"]
            if not 0 < prompt_tokens < len(sequence.token_ids):
                raise ValueError(f"{sequence.sample_id}: empty prompt or response")
            if any(tag != "context" for tag in sequence.token_type_tags[:prompt_tokens]):
                raise ValueError(f"{sequence.sample_id}: prompt tokens were marked for scoring")
            rows.append(NativeRow(sequence, prompt_tokens))
    if not rows or len({row.sequence.sample_id for row in rows}) != len(rows):
        raise ValueError(f"{uri}: empty or duplicate sample IDs")
    return rows


def groups(rows: list[NativeRow]) -> list[tuple[int, list[int], int]]:
    indices = forward_groups([len(row.sequence.token_ids) for row in rows], max_long=MAX_LONG_PER_FORWARD)
    return [(index, group, unique_rows(group)) for index, group in enumerate(indices)]


def group_sequences(rows: list[NativeRow], indices: list[int], real: int) -> list[ProbeSequence]:
    sequences = [rows[index].sequence for index in indices]
    for rank in range(real, len(sequences)):
        sequences[rank] = replace(
            sequences[rank],
            sample_id=f"padding:{rank}",
            token_ids=PADDING_TOKENS,
            token_type_tags=["context"] * len(PADDING_TOKENS),
            position_tags=[0] * len(PADDING_TOKENS),
        )
    return sequences


def score(policy, sequences: list[ProbeSequence], *, top_k: int | None, gather: np.ndarray | None = None) -> dict:
    batch = build_score_batch(
        sequences,
        pad_token_id=PAD_TOKEN_ID,
        gather_ids=gather,
        exact_positions=False,
        exact_rate=0,
        master_seed=0,
    )
    results, _seconds = run_score(policy, batch, top_k=top_k, support_width=TOP_K if gather is not None else None)
    return results


def response_logits_positions(row: NativeRow, padded_len: int) -> np.ndarray:
    first = padded_len - len(row.sequence.token_ids) + row.prompt_tokens - 1
    return np.arange(first, first + len(row.response_ids), dtype=np.int64)


def save_support(path: str, rows: list[NativeRow], results: dict) -> None:
    padded_len = results["topk_ids"].shape[1]
    positions = [response_logits_positions(row, padded_len) for row in rows]
    offsets = np.cumsum([0, *(len(position) for position in positions)])
    payload = io.BytesIO()
    np.savez(
        payload,
        sample_ids=np.asarray([row.sequence.sample_id for row in rows]),
        response_ids=np.concatenate([row.response_ids for row in rows]).astype(np.int64),
        offsets=offsets,
        topk_ids=np.concatenate([results["topk_ids"][index, position] for index, position in enumerate(positions)]),
        topk_logprobs=np.concatenate(
            [results["topk_logprobs"][index, position] for index, position in enumerate(positions)]
        ),
        topk_tail_logmass=np.concatenate(
            [results["topk_tail_logmass"][index, position] for index, position in enumerate(positions)]
        ),
    )
    filesystem, inner = filesystem_for(path)
    with filesystem.open(inner, "wb") as handle:
        handle.write(payload.getvalue())


def load_support(path: str, rows: list[NativeRow]) -> dict[str, np.ndarray]:
    filesystem, inner = filesystem_for(path)
    with filesystem.open(inner, "rb") as handle:
        data = handle.read()
    with np.load(io.BytesIO(data)) as stored:
        support = {key: stored[key] for key in stored.files}
    if list(support["sample_ids"]) != [row.sequence.sample_id for row in rows]:
        raise ValueError(f"{path}: sample IDs differ")
    if support["response_ids"].tolist() != [token for row in rows for token in row.response_ids]:
        raise ValueError(f"{path}: response tokens differ")
    return support


def path_exists(path: str) -> bool:
    filesystem, inner = filesystem_for(path)
    return filesystem.exists(inner)


def write_json(path: str, data: dict) -> None:
    filesystem, inner = filesystem_for(path)
    with filesystem.open(inner, "w") as handle:
        json.dump(data, handle, sort_keys=True)


def scoring_worker(model_path: str):
    from experiments.post_training.mismatch_probe.long_context import scoring_worker as worker  # noqa: PLC0415

    return worker(model_path, num_gpus=DP_SIZE)


def reference_pass(model_path: str, input_uri: str, output_root: str) -> None:
    rows = load_rows(input_uri)
    with scoring_worker(model_path) as policy:
        for index, indices, real in groups(rows):
            path = f"{output_root.rstrip('/')}/support-{index:04d}.npz"
            if path_exists(path):
                load_support(path, [rows[item] for item in indices[:real]])
                continue
            sequences = group_sequences(rows, indices, real)
            results = score(policy, sequences, top_k=TOP_K)
            save_support(path, [rows[item] for item in indices[:real]], results)
    write_json(f"{output_root.rstrip('/')}/reference.json", {"model": model_path, "input": input_uri, "rows": len(rows)})


def aligned_rows(reference: list[NativeRow], candidate: list[NativeRow]) -> None:
    if len(reference) != len(candidate):
        raise ValueError("reference and candidate have different row counts")
    for left, right in zip(reference, candidate, strict=True):
        if left.sequence.sample_id != right.sequence.sample_id or left.response_ids != right.response_ids:
            raise ValueError(f"unaligned response: {left.sequence.sample_id}")


def candidate_group(
    policy, reference: list[NativeRow], candidate: list[NativeRow], indices: list[int], real: int, support: dict
) -> dict:
    selected = [candidate[item] for item in indices[:real]]
    sequences = group_sequences(candidate, indices, real)
    padded_len = max(len(sequence.token_ids) for sequence in sequences)
    gather = np.zeros((len(sequences), padded_len, TOP_K + 1), dtype=np.int64)
    offsets = support["offsets"]
    for row, item in enumerate(selected):
        positions = response_logits_positions(item, padded_len)
        start, stop = offsets[row : row + 2]
        gather[row, positions, :TOP_K] = support["topk_ids"][start:stop]
        gather[row, positions, TOP_K] = item.response_ids
    results = score(policy, sequences, top_k=None, gather=gather)
    totals: dict[str, list[float]] = defaultdict(list)
    for row, item in enumerate(selected):
        start, stop = offsets[row : row + 2]
        positions = response_logits_positions(item, padded_len)
        kls = binned_kl(
            torch.from_numpy(support["topk_logprobs"][start:stop].astype(np.float64)),
            torch.from_numpy(support["topk_tail_logmass"][start:stop].astype(np.float64)),
            torch.from_numpy(results["gathered_logprobs"][row, positions, :TOP_K].astype(np.float64)),
            torch.from_numpy(results["gathered_tail_logmass"][row, positions].astype(np.float64)),
        ).numpy()
        if not np.all(np.isfinite(kls)):
            raise ValueError(f"{item.sequence.sample_id}: non-finite KL")
        tags = reference[indices[row]].sequence.token_type_tags[reference[indices[row]].prompt_tokens :]
        turns = reference[indices[row]].sequence.position_tags[reference[indices[row]].prompt_tokens :]
        for kl, tag, turn in zip(kls, tags, turns, strict=True):
            totals["all"].append(float(kl))
            totals[f"bucket:{tag}"].append(float(kl))
            totals[f"turn:{turn}"].append(float(kl))
    return {name: {"positions": len(values), "kl_sum": sum(values)} for name, values in totals.items()}


def candidate_pass(model_path: str, reference_uri: str, candidate_uri: str, support_root: str, output_root: str) -> None:
    reference = load_rows(reference_uri)
    candidate = load_rows(candidate_uri)
    aligned_rows(reference, candidate)
    with scoring_worker(model_path) as policy:
        for index, indices, real in groups(reference):
            path = f"{output_root.rstrip('/')}/kl-{index:04d}.json"
            if path_exists(path):
                continue
            selected = [reference[item] for item in indices[:real]]
            support = load_support(f"{support_root.rstrip('/')}/support-{index:04d}.npz", selected)
            summary = candidate_group(policy, reference, candidate, indices, real, support)
            write_json(path, {"model": model_path, "reference": support_root, "groups": summary})
    write_json(
        f"{output_root.rstrip('/')}/candidate.json",
        {
            "model": model_path,
            "reference_input": reference_uri,
            "candidate_input": candidate_uri,
            "rows": len(candidate),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("reference", "candidate"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--reference-set", required=True)
    parser.add_argument("--candidate-set")
    parser.add_argument("--support-root")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.mode == "reference":
        reference_pass(args.model_path, args.reference_set, args.out)
    else:
        if not args.candidate_set or not args.support_root:
            parser.error("candidate mode requires --candidate-set and --support-root")
        candidate_pass(args.model_path, args.reference_set, args.candidate_set, args.support_root, args.out)


if __name__ == "__main__":
    main()
