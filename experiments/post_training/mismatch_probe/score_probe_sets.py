# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score checkpoints on the real probe sets with the WP4 passes, resumably, on one node.

The dev driver (``run_score_pass``) sends one padded batch per pass. Probe sets hold
thousands of sequences up to 32k tokens (131k for P_agent), so this driver splits each
set into forward groups and writes one shard per group to object storage; a rerun skips
every shard already written, so a preempted job resumes where it stopped.

Grouping (``forward_groups``): sequences sorted by length, at most ``rows_per_group``
rows and ``token_budget`` padded tokens per group, and every group a multiple of the
data-parallel size (the last row repeats to fill; its copies are dropped). A sequence
longer than ``LONG_SEQUENCE_TOKENS`` goes in a group of exactly one row per rank, at most
``MAX_LONG_PER_FORWARD`` of them, so no forward scores more than four (C1: one node fits
131k tokens with up to four long rows per forward). Every pass runs at micro-batch 1, the
layout the B-vs-B floor was measured at (C5).

Passes, each writing under ``<out>/<set>/``:

- ``reference``: one model's top-256 support per logits position (``reference-*.npz``);
- ``candidate``: another model gathered on that support, reduced to ``SequenceKLRow``
  parquet (``kl-<label>-*.parquet``);
- ``exact``: the seeded 1% position subsample scored with full-vocabulary KL by both
  models (C4), summarized against the binned rows at the same positions;
- ``noise``: the reference model gathered on its own support with every group's rows
  reversed, so each sequence lands on another rank beside other neighbours (C5);
- ``flips``: the candidate's own top-1 per scored position against the reference's,
  reduced to ``SequenceFlipRow`` parquet (``flip-<label>-*.parquet``).

    python -m experiments.post_training.mismatch_probe.score_probe_sets reference \\
        --model-path /tmp/model/e-x --set P_X=s3://.../continuations/snowball-E_X/P_X.jsonl \\
        --out s3://.../scores/ref-E_X --limit 64
"""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import fsspec
import numpy as np

from experiments.post_training.mismatch_probe.scoring_pass import (
    TOP_K,
    ProbeSequence,
    build_score_batch,
    candidate_gather_ids,
    exact_rows_from_passes,
    kl_rows_from_passes,
    load_probe_sequences,
    read_flip_rows_parquet,
    read_rows_parquet,
    write_flip_rows_parquet,
    write_rows_parquet,
)
from experiments.post_training.mismatch_probe.sequence_kl import ProbeBucket, SequenceFlipRow

DP_SIZE = 8
ROWS_PER_GROUP = 64
TOKEN_BUDGET = 64 * 8_192
LONG_SEQUENCE_TOKENS = 65_536
MAX_LONG_PER_FORWARD = 4
PAD_TOKEN_ID = 0
EXACT_RATE = 0.01
MASTER_SEED = 20261023
MICRO_BATCH_SIZE = 1


# --- grouping ---


def forward_groups(
    lengths: list[int],
    *,
    dp: int = DP_SIZE,
    rows_per_group: int = ROWS_PER_GROUP,
    token_budget: int = TOKEN_BUDGET,
    long_threshold: int = LONG_SEQUENCE_TOKENS,
    max_long: int = MAX_LONG_PER_FORWARD,
) -> list[list[int]]:
    """Sequence indices per forward group, each group a multiple of ``dp`` rows.

    Long sequences fill groups of exactly ``dp`` rows (one per rank, so one forward),
    at most ``max_long`` of them, topped up with the shortest remaining sequences.
    The rest go longest first, a group closing at ``rows_per_group`` rows or when its
    padded size (rows rounded up to ``dp``, times the longest) would pass ``token_budget``. A group's tail is filled by
    repeating its last index; ``unique_rows`` recovers the real rows.
    """
    if rows_per_group % dp:
        raise ValueError(f"rows_per_group {rows_per_group} is not a multiple of the DP size {dp}")
    order = sorted(range(len(lengths)), key=lambda index: (-lengths[index], index))
    long = [index for index in order if lengths[index] > long_threshold]
    rest = [index for index in order if lengths[index] <= long_threshold]
    groups: list[list[int]] = []
    while long:
        group, long = long[:max_long], long[max_long:]
        filler = min(dp - len(group), len(rest))
        group += rest[len(rest) - filler :] if filler else []
        rest = rest[: len(rest) - filler]
        groups.append(group)
    group: list[int] = []
    for index in rest:
        longest = lengths[group[0]] if group else lengths[index]
        padded_rows_after = -(-(len(group) + 1) // dp) * dp  # the fill rounds the group up to the DP size
        if group and (len(group) == rows_per_group or padded_rows_after * longest > token_budget):
            groups.append(group)
            group = []
        group.append(index)
    if group:
        groups.append(group)
    return [group + [group[-1]] * (-len(group) % dp) for group in groups]


def unique_rows(group: list[int]) -> int:
    """How many leading rows of a padded group are real (the fill repeats the last index)."""
    count = len(group)
    while count > 1 and group[count - 1] == group[count - 2]:
        count -= 1
    return count


# --- compact per-sequence storage ---


def compact_rows(padded: np.ndarray, widths: list[int]) -> list[np.ndarray]:
    """Each row's logits positions that predict a token: ``[width - 1, ...]`` of a left-padded array."""
    padded_len = padded.shape[1]
    return [padded[row, padded_len - width : padded_len - 1] for row, width in enumerate(widths)]


def padded_rows(rows: list[np.ndarray], widths: list[int], padded_len: int) -> np.ndarray:
    """The inverse of ``compact_rows``: rows back in a left-padded ``[n, padded_len, ...]`` array."""
    out = np.zeros((len(rows), padded_len, *rows[0].shape[1:]), dtype=rows[0].dtype)
    for row, (values, width) in enumerate(zip(rows, widths, strict=True)):
        out[row, padded_len - width : padded_len - 1] = values
    return out


def save_npz(path: str, **arrays: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    with fsspec.open(path, "wb") as handle:
        handle.write(buffer.getvalue())


def load_npz(path: str) -> dict[str, np.ndarray]:
    with fsspec.open(path, "rb") as handle:
        with np.load(io.BytesIO(handle.read())) as stored:
            return {key: stored[key] for key in stored.files}


def save_reference_shard(path: str, sequences: list[ProbeSequence], results: dict[str, np.ndarray]) -> None:
    widths = [len(sequence.token_ids) for sequence in sequences]
    pieces = {key: compact_rows(results[key], widths) for key in ("topk_ids", "topk_logprobs", "topk_tail_logmass")}
    save_npz(
        path,
        sample_ids=np.asarray([sequence.sample_id for sequence in sequences]),
        offsets=np.cumsum([0, *[width - 1 for width in widths]]),
        topk_ids=np.concatenate(pieces["topk_ids"]).astype(np.int32),
        topk_logprobs=np.concatenate(pieces["topk_logprobs"]).astype(np.float32),
        topk_tail_logmass=np.concatenate(pieces["topk_tail_logmass"]).astype(np.float32),
    )


def load_reference_shard(path: str, sequences: list[ProbeSequence]) -> dict[str, np.ndarray]:
    """A reference shard back in the group's padded layout; raises if it holds other sequences."""
    stored = load_npz(path)
    if list(stored["sample_ids"]) != [sequence.sample_id for sequence in sequences]:
        raise ValueError(f"{path} holds a different group of sequences")
    widths = [len(sequence.token_ids) for sequence in sequences]
    padded_len = max(widths)
    offsets = stored["offsets"]
    result = {}
    for key in ("topk_ids", "topk_logprobs", "topk_tail_logmass"):
        rows = [stored[key][offsets[row] : offsets[row + 1]] for row in range(len(sequences))]
        result[key] = padded_rows(rows, widths, padded_len)
    result["topk_ids"] = result["topk_ids"].astype(np.int64)
    return result


# --- sets and shards ---


def stage_set(uri: str, limit: int | None) -> list[ProbeSequence]:
    """A continuation set's rows as probe sequences, optionally its first ``limit`` rows."""
    with fsspec.open(uri, "rt") as handle:
        lines = [line for line in handle if line.strip()]
    if limit is not None:
        lines = lines[:limit]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as local:
        local.writelines(lines)
    return load_probe_sequences(local.name)


def group_sequences(sequences: list[ProbeSequence]) -> list[tuple[int, list[ProbeSequence], int]]:
    """(group index, the group's padded rows, its count of real rows)."""
    groups = forward_groups([len(sequence.token_ids) for sequence in sequences])
    return [(index, [sequences[row] for row in group], unique_rows(group)) for index, group in enumerate(groups)]


def exists(path: str) -> bool:
    filesystem, inner = fsspec.core.url_to_fs(path)
    return filesystem.exists(inner)


def shard_path(out: str, set_name: str, kind: str, index: int, suffix: str) -> str:
    return f"{out.rstrip('/')}/{set_name}/{kind}-{index:04d}.{suffix}"


def parse_sets(specs: list[str]) -> dict[str, str]:
    return dict(spec.split("=", 1) for spec in specs)


def write_json(path: str, value: Any) -> None:
    with fsspec.open(path, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


# --- passes (need a SkyRL runtime) ---


def _score(policy, batch: dict, *, top_k: int | None, support_width: int | None = None) -> dict[str, np.ndarray]:
    # skyrl_train exists only inside a SkyRL runtime.
    from experiments.post_training.mismatch_probe.run_score_pass import run_score  # noqa: PLC0415

    results, _seconds = run_score(
        policy, batch, top_k=top_k, micro_batch_size=MICRO_BATCH_SIZE, support_width=support_width
    )
    return results


def _worker(model_path: str):
    from experiments.post_training.mismatch_probe.long_context import scoring_worker  # noqa: PLC0415

    return scoring_worker(model_path, num_gpus=DP_SIZE)


def _batch(sequences: list[ProbeSequence], *, gather: np.ndarray | None = None, exact: bool = False) -> dict:
    return build_score_batch(
        sequences,
        pad_token_id=PAD_TOKEN_ID,
        gather_ids=gather,
        exact_positions=exact,
        exact_rate=EXACT_RATE if exact else 0.0,
        master_seed=MASTER_SEED,
    )


def _gather_ids(sequences: list[ProbeSequence], reference: dict[str, np.ndarray]) -> np.ndarray:
    padded_len = reference["topk_ids"].shape[1]
    return np.stack(
        [
            candidate_gather_ids(
                sequence, reference["topk_ids"][row, padded_len - len(sequence.token_ids) : -1], padded_len
            )
            for row, sequence in enumerate(sequences)
        ]
    )


def reference_pass(model_path: str, sets: dict[str, str], out: str, limit: int | None) -> None:
    with _worker(model_path) as policy:
        for set_name, uri in sets.items():
            for index, group, _real in group_sequences(stage_set(uri, limit)):
                path = shard_path(out, set_name, "reference", index, "npz")
                if exists(path):
                    continue
                save_reference_shard(path, group, _score(policy, _batch(group), top_k=TOP_K))
            write_json(
                f"{out.rstrip('/')}/{set_name}/reference.json",
                {"model": model_path, "sequences": uri, "limit": limit, "top_k": TOP_K},
            )


def candidate_rows(
    policy,
    group: list[ProbeSequence],
    reference: dict[str, np.ndarray],
    *,
    real: int,
    reference_id: str,
    candidate_id: str,
    candidate_step: int,
    reverse: bool = False,
):
    """KL rows of one group: the candidate gathered on the reference's support (rows reversed for the floor)."""
    order = np.arange(len(group))[::-1] if reverse else np.arange(len(group))
    gather = _gather_ids(group, reference)
    batch = _batch([group[row] for row in order], gather=gather[order])
    results = _score(policy, batch, top_k=None, support_width=TOP_K)
    restored = np.empty_like(order)
    restored[order] = np.arange(len(order))
    rows = kl_rows_from_passes(
        group[:real],
        reference["topk_ids"][:real],
        reference["topk_logprobs"][:real],
        reference["topk_tail_logmass"][:real],
        results["gathered_logprobs"][restored][:real],
        results["gathered_tail_logmass"][restored][:real],
        reference_id=reference_id,
        candidate_id=candidate_id,
        candidate_step=candidate_step,
    )
    return rows


def candidate_pass(
    model_path: str,
    label: str,
    step: int,
    sets: dict[str, str],
    reference_roots: dict[str, str],
    out: str,
    limit: int | None,
) -> None:
    with _worker(model_path) as policy:
        for set_name, uri in sets.items():
            reference_out = reference_roots[set_name]
            with fsspec.open(f"{reference_out.rstrip('/')}/{set_name}/reference.json", "rt") as handle:
                reference_meta = json.load(handle)
            for index, group, real in group_sequences(stage_set(uri, limit)):
                path = shard_path(out, set_name, f"kl-{label}", index, "parquet")
                if exists(path):
                    continue
                reference = load_reference_shard(shard_path(reference_out, set_name, "reference", index, "npz"), group)
                rows = candidate_rows(
                    policy,
                    group,
                    reference,
                    real=real,
                    reference_id=reference_meta["model"],
                    candidate_id=model_path,
                    candidate_step=step,
                )
                write_rows_shard(path, rows)
            write_json(f"{out.rstrip('/')}/{set_name}/kl-{label}.json", summarize_kl(out, set_name, f"kl-{label}"))


def noise_pass(
    model_path: str, sets: dict[str, str], reference_roots: dict[str, str], out: str, limit: int | None
) -> None:
    """The reference model gathered on its own support with each group's rows reversed (C5)."""
    with _worker(model_path) as policy:
        for set_name, uri in sets.items():
            reference_out = reference_roots[set_name]
            for index, group, real in group_sequences(stage_set(uri, limit)):
                path = shard_path(out, set_name, "kl-noise", index, "parquet")
                if exists(path):
                    continue
                reference = load_reference_shard(shard_path(reference_out, set_name, "reference", index, "npz"), group)
                rows = candidate_rows(
                    policy,
                    group,
                    reference,
                    real=real,
                    reference_id=f"{model_path} micro-batch 1",
                    candidate_id=f"{model_path} rows reversed",
                    candidate_step=0,
                    reverse=True,
                )
                write_rows_shard(path, rows)
            write_json(f"{out.rstrip('/')}/{set_name}/kl-noise.json", summarize_kl(out, set_name, "kl-noise"))


def exact_pass(
    reference_model: str, candidate_model: str, label: str, sets: dict[str, str], out: str, limit: int | None
) -> None:
    """Full-vocabulary KL on the seeded 1% subsample (C4), against the binned rows at the same positions."""
    work = tempfile.mkdtemp()
    staged = {set_name: stage_set(uri, limit) for set_name, uri in sets.items()}
    for role, model in (("reference", reference_model), ("candidate", candidate_model)):
        with _worker(model) as policy:
            for set_name, sequences in staged.items():
                for index, group, _real in group_sequences(sequences):
                    local = f"{work}/{set_name}-{role}-{index:04d}.npz"
                    if Path(local).exists():
                        continue
                    results = _score(policy, _batch(group, exact=True), top_k=None)
                    np.savez(local, rows=results["exact_logprobs"], positions=results["exact_positions"])
    for set_name, sequences in staged.items():
        exact_rows = []
        for index, group, real in group_sequences(sequences):
            reference = np.load(f"{work}/{set_name}-reference-{index:04d}.npz")
            candidate = np.load(f"{work}/{set_name}-candidate-{index:04d}.npz")
            rows = exact_rows_from_passes(
                group,
                reference["rows"],
                reference["positions"].tolist(),
                candidate["rows"],
                candidate["positions"].tolist(),
                padded_len=max(len(sequence.token_ids) for sequence in group),
            )
            real_ids = {sequence.sample_id for sequence in group[:real]}
            exact_rows += [row for row in rows if row[0] in real_ids]
        binned = {
            (row.sequence_id, row.position): row
            for path in _shards(out, set_name, f"kl-{label}")
            for row in read_rows_parquet(_local_copy(path))
        }
        paired = [(binned[(sample_id, position)], kl) for sample_id, position, _hash, kl in exact_rows]
        write_json(f"{out.rstrip('/')}/{set_name}/exact-{label}.json", exact_summary(paired))


def exact_summary(paired: list[tuple[Any, float]]) -> dict[str, Any]:
    """Binned against exact KL on the same positions, overall and per bucket."""

    def summary(pairs):
        binned = np.asarray([row.kl for row, _ in pairs])
        exact = np.asarray([kl for _, kl in pairs])
        return {
            "positions": len(pairs),
            "mean_binned_kl": float(binned.mean()),
            "mean_exact_kl": float(exact.mean()),
            "relative_error_of_mean": float((binned.mean() - exact.mean()) / exact.mean()) if exact.mean() else None,
            "binned_at_most_exact_share": float(np.mean(binned <= exact + 1e-9)),
            "gap_p99": float(np.percentile(exact - binned, 99)),
        }

    by_bucket: dict[str, list] = {}
    for row, kl in paired:
        by_bucket.setdefault(row.token_type_tag.value, []).append((row, kl))
    return {"all": summary(paired), "buckets": {bucket: summary(pairs) for bucket, pairs in sorted(by_bucket.items())}}


# --- summaries ---


def write_rows_shard(path: str, rows) -> None:
    local = Path(tempfile.mkdtemp()) / Path(path).name
    write_rows_parquet(rows, local)
    with fsspec.open(path, "wb") as handle:
        handle.write(local.read_bytes())


def write_flip_rows_shard(path: str, rows) -> None:
    local = Path(tempfile.mkdtemp()) / Path(path).name
    write_flip_rows_parquet(rows, local)
    with fsspec.open(path, "wb") as handle:
        handle.write(local.read_bytes())


def _local_copy(path: str) -> Path:
    local = Path(tempfile.mkdtemp()) / Path(path).name
    with fsspec.open(path, "rb") as handle:
        local.write_bytes(handle.read())
    return local


def _shards(out: str, set_name: str, kind: str) -> Iterator[str]:
    filesystem, inner = fsspec.core.url_to_fs(f"{out.rstrip('/')}/{set_name}")
    protocol = out.split("://", 1)[0] + "://" if "://" in out else ""
    for name in sorted(filesystem.ls(inner, detail=False)):
        if Path(name).name.startswith(f"{kind}-") and name.endswith(".parquet"):
            yield f"{protocol}{name}"


def summarize_kl(out: str, set_name: str, kind: str) -> dict[str, Any]:
    """Token-mean and sequence-mean KL of a set's shards, overall and per bucket."""
    kls: list[float] = []
    per_sequence: dict[str, list[float]] = {}
    per_bucket: dict[str, list[float]] = {}
    for path in _shards(out, set_name, kind):
        for row in read_rows_parquet(_local_copy(path)):
            kls.append(row.kl)
            per_sequence.setdefault(row.sequence_id, []).append(row.kl)
            per_bucket.setdefault(row.token_type_tag.value, []).append(row.kl)
    return {
        "positions": len(kls),
        "sequences": len(per_sequence),
        "token_mean_kl": float(np.mean(kls)),
        "sequence_mean_kl": float(np.mean([np.mean(values) for values in per_sequence.values()])),
        "buckets": {
            bucket: {"positions": len(values), "token_mean_kl": float(np.mean(values))}
            for bucket, values in sorted(per_bucket.items())
        },
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def flips_pass(
    model_path: str,
    label: str,
    step: int,
    sets: dict[str, str],
    reference_roots: dict[str, str],
    out: str,
    limit: int | None,
) -> None:
    """Greedy-token flips per scored position: the candidate's top-1 against the reference's."""
    with _worker(model_path) as policy:
        for set_name, uri in sets.items():
            reference_out = reference_roots[set_name]
            with fsspec.open(f"{reference_out.rstrip('/')}/{set_name}/reference.json", "rt") as handle:
                reference_meta = json.load(handle)
            for index, group, real in group_sequences(stage_set(uri, limit)):
                path = shard_path(out, set_name, f"flip-{label}", index, "parquet")
                if exists(path):
                    continue
                reference = load_reference_shard(shard_path(reference_out, set_name, "reference", index, "npz"), group)
                results = _score(policy, _batch(list(group)), top_k=1)
                rows = flip_rows(
                    group,
                    reference,
                    results,
                    real=real,
                    reference_id=reference_meta["model"],
                    candidate_id=model_path,
                    candidate_step=step,
                )
                write_flip_rows_shard(path, rows)
            write_json(
                f"{out.rstrip('/')}/{set_name}/flip-{label}.json", summarize_flips(out, set_name, f"flip-{label}")
            )


def flip_rows(
    sequences: list[ProbeSequence],
    reference: dict[str, np.ndarray],
    results: dict[str, np.ndarray],
    *,
    real: int,
    reference_id: str,
    candidate_id: str,
    candidate_step: int,
) -> list[SequenceFlipRow]:
    """Greedy-token flips of one group: the candidate's own top-1 against the reference's.

    ``results`` is the candidate's top-1 pass over the same batch layout as the
    stored reference shard; both are aligned with the model's logits positions,
    where position t's distribution predicts the token at t+1.
    """
    rows: list[SequenceFlipRow] = []
    padded_len = reference["topk_ids"].shape[1]
    for row, sequence in enumerate(sequences[:real]):
        positions = np.asarray(sequence.scored_positions(), dtype=np.int64)
        if not len(positions):
            continue
        logits_positions = padded_len - len(sequence.token_ids) + positions - 1
        reference_top1 = reference["topk_ids"][row, logits_positions, 0]
        candidate_top1 = results["topk_ids"][row, logits_positions, 0]
        probe_hash = sequence.probe_hash  # a sha256 over the whole sequence: once per sequence
        for index, position in enumerate(positions.tolist()):
            rows.append(
                SequenceFlipRow(
                    sequence_id=sequence.sample_id,
                    position=position,
                    reference_top1=int(reference_top1[index]),
                    candidate_top1=int(candidate_top1[index]),
                    flipped=bool(reference_top1[index] != candidate_top1[index]),
                    reference_top1_logprob=float(reference["topk_logprobs"][row, logits_positions[index], 0]),
                    candidate_top1_logprob=float(results["topk_logprobs"][row, logits_positions[index], 0]),
                    candidate_step=candidate_step,
                    reference_id=reference_id,
                    candidate_id=candidate_id,
                    probe_hash=probe_hash,
                    token_type_tag=ProbeBucket(sequence.token_type_tags[position]),
                    position_tag=sequence.position_tags[position],
                )
            )
    return rows


def summarize_flips(out: str, set_name: str, kind: str) -> dict[str, Any]:
    """Flip rate of a set's shards, overall, per bucket, and per reference-margin quartile."""
    flips: list[bool] = []
    margins: list[float] = []
    per_sequence: dict[str, list[bool]] = {}
    per_bucket: dict[str, list[bool]] = {}
    for path in _shards(out, set_name, kind):
        for row in read_flip_rows_parquet(_local_copy(path)):
            flips.append(row.flipped)
            margins.append(row.reference_top1_logprob)
            per_sequence.setdefault(row.sequence_id, []).append(row.flipped)
            per_bucket.setdefault(row.token_type_tag.value, []).append(row.flipped)
    quartile_edges = np.quantile(margins, [0.25, 0.5, 0.75]) if flips else np.zeros(3)
    per_quartile: dict[str, list[bool]] = {}
    for flip, margin in zip(flips, margins, strict=True):
        quartile = int(np.searchsorted(quartile_edges, margin, side="right")) + 1
        per_quartile.setdefault(f"q{quartile}", []).append(flip)

    def rates(values: list[bool]) -> dict[str, Any]:
        return {"positions": len(values), "flip_rate": float(np.mean(values))}

    return {
        "positions": len(flips),
        "sequences": len(per_sequence),
        "flip_rate": float(np.mean(flips)) if flips else 0.0,
        "sequence_flip_rate": float(np.mean([np.mean(values) for values in per_sequence.values()])),
        "buckets": {bucket: rates(values) for bucket, values in sorted(per_bucket.items())},
        "reference_margin_quartiles": {quartile: rates(values) for quartile, values in sorted(per_quartile.items())},
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def resolve_reference_roots(sets: dict[str, str], default: str, overrides: list[str] | None) -> dict[str, str]:
    """One reference root per set: the ``--reference-out`` default, with ``SET=DIR`` overrides."""
    roots = {set_name: default for set_name in sets}
    for spec in overrides or []:
        set_name, separator, directory = spec.partition("=")
        if not separator or set_name not in sets:
            raise ValueError(f"--reference-root {spec!r} must be SET=DIR for a scored set ({', '.join(sorted(sets))})")
        roots[set_name] = directory
    return roots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("reference", "candidate", "noise", "exact", "flips"):
        command = commands.add_parser(name)
        command.add_argument("--set", dest="sets", action="append", required=True, help="SET=continuation jsonl uri")
        command.add_argument("--out", required=True)
        command.add_argument("--limit", type=int, default=None, help="score only each set's first N sequences")
        if name in ("reference", "candidate", "noise", "flips"):
            command.add_argument("--model-path", required=True)
        if name in ("candidate", "noise", "flips"):
            command.add_argument("--reference-out", required=True)
            command.add_argument(
                "--reference-root",
                dest="reference_roots",
                action="append",
                default=None,
                help="SET=DIR: where SET's stored reference lives, when sets share one --reference-out",
            )
        if name in ("candidate", "exact", "flips"):
            command.add_argument("--label", required=True, help="names this candidate's shards")
        if name in ("candidate", "flips"):
            command.add_argument("--step", type=int, required=True)
        if name == "exact":
            command.add_argument("--reference-model", required=True)
            command.add_argument("--candidate-model", required=True)
    args = parser.parse_args()
    sets = parse_sets(args.sets)
    if args.command == "reference":
        reference_pass(args.model_path, sets, args.out, args.limit)
    elif args.command == "candidate":
        roots = resolve_reference_roots(sets, args.reference_out, args.reference_roots)
        candidate_pass(args.model_path, args.label, args.step, sets, roots, args.out, args.limit)
    elif args.command == "noise":
        roots = resolve_reference_roots(sets, args.reference_out, args.reference_roots)
        noise_pass(args.model_path, sets, roots, args.out, args.limit)
    elif args.command == "flips":
        roots = resolve_reference_roots(sets, args.reference_out, args.reference_roots)
        flips_pass(args.model_path, args.label, args.step, sets, roots, args.out, args.limit)
    else:
        exact_pass(args.reference_model, args.candidate_model, args.label, sets, args.out, args.limit)


if __name__ == "__main__":
    main()
