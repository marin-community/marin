# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
VLM benchmark evaluation using log-likelihood comparison.

Supports two task types:
- **Multiple-choice** (ai2d, mmmu): Uses Modified MCQ format (TASK_OVERVIEW.md
  Format 3) where each completion is "letter. choice_text" (e.g., "B. London")
  instead of just the letter. Reports both accuracy (argmax over options) and
  loss (mean length-normalized log P of correct answer). The Modified MCQ format
  gives R²=0.61 for loss predictability vs R²=0.28 for letter-only.
- **Open-ended** (textvqa, chartqa): compute length-normalized
  log P(answer | prompt + image) → mean normalized log-likelihood.

Both task types reuse the same Levanter loglikelihood infrastructure:
PromptCompletion → greedy_pack → make_array_from_callback → named_jit(eval_loglikelihood).

All benchmarks are merged into a SINGLE continuous batch loop (via
evaluate_all_vlm_benchmarks) to avoid TPU crashes at benchmark transitions.
Process 0 prepares eval payloads and broadcasts them to worker hosts in lockstep,
matching the text-mc coordinator/worker communication pattern.
Results are split by benchmark afterward.

Usage (standalone):
    uv run experiments/unified/vlm_mc_eval.py \
        --checkpoint_path gs://marin-eu-west4/checkpoints/unified-qwen3-1.7b-1-1-1-w0.5-3e4-demo5-87244a/hf/step-6000/ \
        --benchmarks ai2d mmmu \
        --eval_batch_size 4 --checkpoint_is_hf

    # MC only
    uv run experiments/unified/vlm_mc_eval.py \
        --checkpoint_path path/to/hf/checkpoint \
        --checkpoint_is_hf \
        --benchmarks ai2d mmmu
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import tempfile
import uuid
import zlib
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq
from jax.sharding import PartitionSpec
from tqdm_loggable.auto import tqdm

import haliax as hax

from levanter.data.loader import stack_batches
from levanter.data.packing import PromptCompletion, greedy_pack_prompt_completions, per_segment_loss
from levanter.eval_harness import get_padding_count_from_batch, get_segment_ids_from_batch
from levanter.models.loss import next_token_loss
from levanter.models.lm_model import LmExample
from levanter.utils.jax_utils import broadcast_shard, multihost_broadcast_sync

from experiments.unified.vlm_tokenize_captions import (
    VISION_END_ID,
    VISION_START_ID,
    VISUAL_TOKEN_OFFSET,
)
from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)


def _hlo_stage_fingerprints(
    dump_dir: str,
    stage_marker: str,
    module_marker: str = "vlm_eval_loglikelihood",
    max_items: int = 24,
) -> list[str]:
    """Return stable fingerprints for dumped HLO files for one stage."""
    if not os.path.isdir(dump_dir):
        return []

    candidates: list[str] = []
    for root, _dirs, files in os.walk(dump_dir):
        for name in files:
            lower = name.lower()
            if stage_marker not in lower:
                continue
            if not lower.endswith((".txt", ".hlo", ".mlir", ".pb")):
                continue
            candidates.append(os.path.join(root, name))

    if not candidates:
        return []

    tagged = [p for p in candidates if module_marker in p.lower()]
    chosen = tagged if tagged else candidates
    chosen = sorted(chosen)[:max_items]

    out: list[str] = []
    for path in chosen:
        try:
            with open(path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()[:16]
            out.append(f"{os.path.basename(path)}:{digest}")
        except Exception as e:
            out.append(f"{os.path.basename(path)}:error={type(e).__name__}")
    return out


def _log_vlm_hlo_fingerprints(eval_run_uuid: str, dump_dir: str = "/tmp/xla_dumps_vlm_once") -> None:
    """Log host-local HLO dump fingerprints for before/after optimization stages."""
    host = jax.process_index()
    for stage, marker in (("before", "before_optimizations"), ("after", "after_optimizations")):
        digests = _hlo_stage_fingerprints(dump_dir=dump_dir, stage_marker=marker)
        logger.warning(
            "VLM_HLO_FINGERPRINT eval_run_uuid=%s host=%d stage=%s count=%d dump_dir=%s digests=%s",
            eval_run_uuid,
            host,
            stage,
            len(digests),
            dump_dir,
            digests,
        )


def _to_host_array_for_debug(x):
    """Best-effort conversion of an array-like object for diagnostics only.

    This helper must never be used for metric computation; it exists to keep
    logging/inspection from crashing when arrays are globally sharded.
    """
    try:
        return np.asarray(x)
    except Exception:
        pass

    shards = getattr(x, "addressable_shards", None)
    if shards is not None:
        local_arrays = [np.asarray(s.data) for s in shards]
        if local_arrays:
            try:
                return np.concatenate(local_arrays, axis=0)
            except Exception:
                return local_arrays[0]

    raise RuntimeError(f"Could not convert value to host array for debug: {type(x)}")


def _compute_batch_fingerprint(batch_orig, batch_idx: int) -> dict[str, object]:
    """Compute a deterministic host-side fingerprint for a pre-shard batch."""
    tokens_host = np.asarray(_to_host_array_for_debug(batch_orig.tokens.array))
    loss_weight_host = np.asarray(_to_host_array_for_debug(batch_orig.loss_weight.array))
    seg = batch_orig.attn_mask.segment_ids
    if isinstance(seg, tuple):
        seg = seg[0]
    seg_arr = getattr(seg, "array", seg)
    seg_host = np.asarray(_to_host_array_for_debug(seg_arr))

    return {
        "batch_idx": batch_idx,
        "tokens_shape": tuple(tokens_host.shape),
        "tokens_dtype": str(tokens_host.dtype),
        "tokens_crc32": int(zlib.crc32(tokens_host.tobytes())),
        "loss_weight_shape": tuple(loss_weight_host.shape),
        "loss_weight_dtype": str(loss_weight_host.dtype),
        "loss_weight_crc32": int(zlib.crc32(loss_weight_host.tobytes())),
        "segment_shape": tuple(seg_host.shape),
        "segment_dtype": str(seg_host.dtype),
        "segment_crc32": int(zlib.crc32(seg_host.tobytes())),
    }


def _canonicalize_fingerprint(fp: dict[str, object]) -> str:
    """Return a stable JSON string for cross-host fingerprint comparison."""
    return json.dumps(fp, sort_keys=True, separators=(",", ":"))


def _make_vlm_eval_jit(axis_resources, mp, EvalPos, EvalBatch, max_packed_segments):
    """Create a JIT loglikelihood function for VLM evaluation.

    Structurally identical to eval_harness.py's _eval_loglikelihood:
    - model(tokens, attn_mask) → logits → next_token_loss(reduction=None)
    - per_segment_loss inside JIT via hax.vmap → small aggregated output
    - out_axis_resources={} forces fully replicated outputs

    Returns (segment_ids, neg_losses) flattened across batch and segments.
    """
    from levanter.utils.tree_utils import inference_mode

    EvalPos_name = EvalPos.name

    def _eval_step(model, packed_example):
        model = inference_mode(model, True)
        if mp is not None:
            model = mp.cast_to_compute(model)

        logits = model(packed_example.tokens, attn_mask=packed_example.attn_mask)
        logits = logits.astype(jnp.float32)
        Pos = logits.resolve_axis(EvalPos_name)

        loss = next_token_loss(
            Pos=Pos,
            Vocab=model.Vocab,
            logits=logits,
            true_ids=packed_example.tokens,
            loss_weight=packed_example.loss_weight,
            reduction=None,
        )

        # +1 because -1 is used as padding value for segments
        max_Segments = hax.Axis("Segments", size=max_packed_segments + 1)

        batched_segment_ids, batched_per_segment_losses = hax.vmap(per_segment_loss, EvalBatch)(
            packed_example, loss, max_Segments
        )

        segments = hax.flatten(batched_segment_ids, "segment")
        losses = hax.flatten(batched_per_segment_losses, "segment")

        return segments, -losses

    _eval_step.__name__ = "vlm_eval_loglikelihood_step"
    return hax.named_jit(_eval_step, axis_resources=axis_resources, out_axis_resources={})


def _make_dummy_eval_batch(EvalBatch, EvalPos):
    """Build a fixed-shape dummy batch used by non-source hosts in broadcast_shard."""
    dummy_batch = hax.vmap(LmExample.causal, EvalBatch)(
        hax.zeros(EvalPos, dtype=jnp.int32),
        loss_weight=hax.zeros(EvalPos, dtype=jnp.float32),
        segment_ids=hax.zeros(EvalPos, dtype=jnp.int32),
    )
    # Keep dummy payload construction aligned with text-mc eval_harness worker
    # setup so worker-side payload specs/sharding shape stay deterministic.
    return hax.shard(dummy_batch, {})


class _BatchLoopMessage:
    """Control messages used to keep source/worker batch execution in lockstep."""

    STOP = 0
    RUN_BATCH = 1


def _send_batch_loop_message(message: int) -> int:
    """Source host sends a control message to all hosts (text-mc pattern)."""
    if jax.process_index() != 0:
        raise RuntimeError("Only source host can send batch-loop messages.")
    out = broadcast_shard(jnp.array(message, dtype=jnp.int32), PartitionSpec())
    return int(out.item())


def _receive_batch_loop_message() -> int:
    """Worker hosts block until a source control message is broadcast."""
    if jax.process_index() == 0:
        raise RuntimeError("Source host must not call _receive_batch_loop_message.")
    msg = broadcast_shard(jnp.array(_BatchLoopMessage.STOP, dtype=jnp.int32), PartitionSpec())
    return int(msg.item())


def _send_batch_payload(payload):
    """Source host sends one batch payload using text-mc's spec inference pattern."""
    assert jax.process_index() == 0
    payload_specs = hax.partitioning.infer_resource_partitions(payload)
    return broadcast_shard(payload, payload_specs)


def _receive_batch_payload(dummy_payload):
    """Worker hosts receive one batch payload using dummy-payload-derived specs."""
    payload_specs = hax.partitioning.infer_resource_partitions(dummy_payload)
    return broadcast_shard(dummy_payload, payload_specs)


def _run_batch_loop(
    jit_fn,
    model,
    all_completions,
    tokenizer,
    EvalPos,
    EvalBatch,
    max_packed_segments,
    axis_resources,
    vocab_size: int,
    eval_run_uuid: str = "n/a",
):
    """Pack completions, batch, sync across hosts, and run one continuous loop.

    Returns (result_probs, covered_points) numpy arrays indexed by completion
    list position.  This is the shared "hot loop" — keeping it as a single
    uninterrupted iteration prevents the TPU-crash that occurs when separate
    benchmark loops are run back-to-back.

    Per-segment aggregation is done inside JIT (via per_segment_loss + hax.vmap),
    matching eval_harness.py's pattern exactly.
    """
    pad_token = tokenizer.pad_token_id
    if pad_token is None:
        pad_token = tokenizer.eos_token_id

    is_source = jax.process_index() == 0
    if is_source:
        packed = greedy_pack_prompt_completions(
            EvalPos,
            all_completions,
            pad_token=pad_token,
            max_segments_per_example=max_packed_segments,
        )
        all_batches = list(stack_batches(iter(packed), EvalPos, EvalBatch))
        local_num_points = len(all_completions)
        local_total_tokens = len(packed) * EvalPos.size
    else:
        all_batches = []
        local_num_points = 0
        local_total_tokens = 0

    has_work = multihost_broadcast_sync(local_num_points > 0)
    if not has_work:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=bool)

    total_points = int(multihost_broadcast_sync(local_num_points))
    total_tokens = int(multihost_broadcast_sync(local_total_tokens))
    local_num_batches = len(all_batches)
    logger.info(
        "[%s][host=%d] Prepared packed batches: local=%d",
        eval_run_uuid,
        jax.process_index(),
        local_num_batches,
    )
    num_batches = multihost_broadcast_sync(local_num_batches)
    logger.info(
        "[%s][host=%d] Broadcast batch count: local=%d, leader=%d",
        eval_run_uuid,
        jax.process_index(),
        local_num_batches,
        num_batches,
    )
    if is_source and len(all_batches) != num_batches:
        raise RuntimeError(
            f"[{eval_run_uuid}] Source host has inconsistent batch count: local={len(all_batches)} leader={num_batches}"
        )
    padded_or_truncated_count = 0
    logger.warning(
        "VLM_EVAL_BATCH_SYNC eval_run_uuid=%s host=%d local_num_batches=%d leader_num_batches=%d "
        "padded_or_truncated_count=%d",
        eval_run_uuid,
        jax.process_index(),
        local_num_batches,
        num_batches,
        padded_or_truncated_count,
    )

    result_probs = np.zeros(total_points)
    covered_points = np.zeros(total_points, dtype=bool)

    pbar = tqdm(total=total_tokens, desc="vlm_eval", unit="tok") if is_source else None
    dummy_batch = _make_dummy_eval_batch(EvalBatch, EvalPos)
    out_specs = hax.partitioning.infer_resource_partitions(dummy_batch)
    logger.warning(
        "VLM_EVAL_OUT_SPECS eval_run_uuid=%s host=%d out_specs_crc32=%d mesh=%s",
        eval_run_uuid,
        jax.process_index(),
        int(zlib.crc32(str(out_specs).encode())),
        hax.partitioning._get_mesh(),
    )
    source_batch_idx = 0
    worker_batch_idx = 0
    while True:
        if is_source:
            if source_batch_idx >= num_batches:
                _send_batch_loop_message(_BatchLoopMessage.STOP)
                break
            batch_idx = source_batch_idx
            _send_batch_loop_message(_BatchLoopMessage.RUN_BATCH)
            batch_orig = all_batches[batch_idx]
            source_batch_idx += 1
        else:
            message = _receive_batch_loop_message()
            if message == _BatchLoopMessage.STOP:
                break
            if message != _BatchLoopMessage.RUN_BATCH:
                raise RuntimeError(f"[{eval_run_uuid}] unknown batch-loop message={message}")
            batch_idx = worker_batch_idx
            worker_batch_idx += 1
            batch_orig = dummy_batch

        logger.warning(
            "VLM_EVAL_BATCH_ENTER eval_run_uuid=%s host=%d batch_idx=%d leader_num_batches=%d",
            eval_run_uuid,
            jax.process_index(),
            batch_idx,
            num_batches,
        )

        if is_source:
            # Hard host-side guardrail: catch out-of-vocab ids before launching
            # TPU kernels so failures surface as actionable Python exceptions.
            tokens_host = np.asarray(_to_host_array_for_debug(batch_orig.tokens.array))
            if tokens_host.size > 0:
                tok_min = int(tokens_host.min())
                tok_max = int(tokens_host.max())
                if tok_min < 0 or tok_max >= vocab_size:
                    raise RuntimeError(
                        f"[{eval_run_uuid}][host={jax.process_index()}] token id out of range at batch={batch_idx}: "
                        f"min={tok_min}, max={tok_max}, vocab_size={vocab_size}, shape={tokens_host.shape}"
                    )

        if batch_idx < 2:
            tokens_array = batch_orig.tokens.array
            segment_ids = batch_orig.attn_mask.segment_ids
            if isinstance(segment_ids, tuple):
                segment_ids = segment_ids[0]
            logger.info(
                "[%s][host=%d] Batch %d pre-shard: tokens_type=%s shape=%s dtype=%s sharding=%s "
                "segment_ids_type=%s segment_ids_array_type=%s",
                eval_run_uuid,
                jax.process_index(),
                batch_idx,
                type(tokens_array).__name__,
                getattr(tokens_array, "shape", None),
                getattr(tokens_array, "dtype", None),
                getattr(tokens_array, "sharding", None),
                type(batch_orig.attn_mask.segment_ids).__name__,
                type(getattr(segment_ids, "array", segment_ids)).__name__,
            )
            if batch_idx == 0:
                logger.warning(
                    "VLM_EVAL_BATCH0 eval_run_uuid=%s host=%d tokens_shape=%s tokens_dtype=%s "
                    "pre_shard_tokens_sharding=%s segment_ids_type=%s",
                    eval_run_uuid,
                    jax.process_index(),
                    getattr(tokens_array, "shape", None),
                    getattr(tokens_array, "dtype", None),
                    getattr(tokens_array, "sharding", None),
                    type(batch_orig.attn_mask.segment_ids).__name__,
                )

        if is_source:
            _padding_count, batch_tokens = get_padding_count_from_batch(batch_orig, pad_token)
            segments_this_batch = get_segment_ids_from_batch(batch_orig, max_packed_segments * EvalBatch.size)
        else:
            batch_tokens = 0
            segments_this_batch = []

        if batch_idx == num_batches - 1:
            logger.warning(
                "VLM_EVAL_LAST_BATCH_PRE_BROADCAST eval_run_uuid=%s host=%d batch_idx=%d",
                eval_run_uuid,
                jax.process_index(),
                batch_idx,
            )

        if batch_idx == 0:
            if is_source:
                payload_specs = hax.partitioning.infer_resource_partitions(batch_orig)
                role = "source"
            else:
                payload_specs = hax.partitioning.infer_resource_partitions(dummy_batch)
                role = "worker"
            logger.warning(
                "VLM_EVAL_BATCH0_PAYLOAD_SPECS eval_run_uuid=%s host=%d role=%s payload_specs_crc32=%d payload_specs=%s",
                eval_run_uuid,
                jax.process_index(),
                role,
                int(zlib.crc32(str(payload_specs).encode())),
                payload_specs,
            )

        try:
            if is_source:
                batch = _send_batch_payload(batch_orig)
            else:
                batch = _receive_batch_payload(dummy_batch)
        except Exception:
            tokens_array = batch_orig.tokens.array
            logger.exception(
                "[%s][host=%d] batch sharding failed at batch=%d axis_resources=%s tokens_type=%s shape=%s dtype=%s "
                "tokens_sharding=%s",
                eval_run_uuid,
                jax.process_index(),
                batch_idx,
                axis_resources,
                type(tokens_array).__name__,
                getattr(tokens_array, "shape", None),
                getattr(tokens_array, "dtype", None),
                getattr(tokens_array, "sharding", None),
            )
            raise

        out_ids, out_lls = jit_fn(model, batch)
        # Ensure all hosts complete this step's device work before either
        # source-side host parsing or the next control-message broadcast.
        jax.block_until_ready(out_ids.array)
        jax.block_until_ready(out_lls.array)
        if batch_idx == num_batches - 1:
            logger.warning(
                "VLM_EVAL_LAST_BATCH_POST_JIT eval_run_uuid=%s host=%d batch_idx=%d",
                eval_run_uuid,
                jax.process_index(),
                batch_idx,
            )

        valid_points = 0
        if is_source:
            # out_axis_resources={} means outputs are fully replicated on every
            # host. We only materialize host values on source to avoid extra
            # host/device synchronization paths on worker hosts.
            out_ids = _to_host_array_for_debug(out_ids.array)
            out_lls = _to_host_array_for_debug(out_lls.array)
            valid = out_ids != -1
            out_ids_this_batch = out_ids[valid].astype(int).tolist()
            missing_ids = set(segments_this_batch) - set(out_ids_this_batch)
            extra_ids = set(out_ids_this_batch) - set(segments_this_batch)
            if missing_ids or extra_ids:
                raise RuntimeError(
                    f"[{eval_run_uuid}][host={jax.process_index()}] Segment mismatch at batch={batch_idx}: "
                    f"segments={len(segments_this_batch)}, out_ids={len(out_ids_this_batch)}, "
                    f"missing={sorted(list(missing_ids))[:10]}, extra={sorted(list(extra_ids))[:10]}, "
                    f"segments_sample={segments_this_batch[:10]}, out_ids_sample={out_ids_this_batch[:10]}"
                )

            result_probs[out_ids[valid].astype(int)] = out_lls[valid]
            covered_points[out_ids[valid].astype(int)] = True
            valid_points = int(np.sum(valid))

            if pbar is not None:
                pbar.update(batch_tokens)

        logger.warning(
            "VLM_EVAL_BATCH_EXIT eval_run_uuid=%s host=%d batch_idx=%d valid_points=%d batch_tokens=%d",
            eval_run_uuid,
            jax.process_index(),
            batch_idx,
            valid_points,
            batch_tokens,
        )

    if pbar is not None:
        pbar.close()
    if is_source and source_batch_idx != num_batches:
        raise RuntimeError(
            f"[{eval_run_uuid}] Source loop consumed {source_batch_idx} batches but expected {num_batches} batches."
        )
    if not is_source and worker_batch_idx != num_batches:
        raise RuntimeError(
            f"[{eval_run_uuid}] Worker loop consumed {worker_batch_idx} batches but expected {num_batches} batches."
        )

    logger.info("[%s][host=%d] vlm_eval batch loop complete", eval_run_uuid, jax.process_index())

    if is_source:
        missing = np.where(~covered_points)[0]
        if len(missing) > 0:
            logger.warning("Missing %d segments: %s", len(missing), missing[:10])

    return result_probs, covered_points


def _compute_mc_metrics(result_probs, covered_points, segment_metadata):
    """Compute MC accuracy and loss from per-segment log-likelihoods.

    segment_metadata maps local segment IDs (0-based) to
    (question_idx, option_idx, is_correct, num_completion_tokens).
    result_probs and covered_points are sliced to match.
    """
    num_questions = len({meta[0] for meta in segment_metadata.values()})

    question_options: dict[int, list[tuple[int, bool, float]]] = defaultdict(list)
    for seg_id, (q_idx, opt_idx, is_correct, _n_tokens) in segment_metadata.items():
        if covered_points[seg_id]:
            ll = float(result_probs[seg_id])
            question_options[q_idx].append((opt_idx, is_correct, ll))

    num_correct = 0
    per_question = []
    for q_idx in range(num_questions):
        options = question_options.get(q_idx, [])
        if not options:
            per_question.append({"question_idx": q_idx, "predicted": -1, "correct": False})
            continue

        best = max(options, key=lambda x: x[2])
        correct = best[1]
        if correct:
            num_correct += 1
        per_question.append(
            {
                "question_idx": q_idx,
                "predicted": best[0],
                "correct": correct,
                "log_likelihoods": {opt_idx: ll for opt_idx, _, ll in options},
            }
        )

    accuracy = num_correct / max(num_questions, 1)

    total_correct_norm_ll = 0.0
    num_with_loss = 0
    for seg_id, (_q_idx, _opt_idx, is_correct, n_tokens) in segment_metadata.items():
        if is_correct and covered_points[seg_id]:
            ll = float(result_probs[seg_id])
            norm_ll = ll / max(n_tokens, 1)
            total_correct_norm_ll += norm_ll
            num_with_loss += 1

    loss = total_correct_norm_ll / max(num_with_loss, 1)
    logger.info(
        "MC accuracy: %d/%d = %.4f, loss (mean norm LL): %.4f",
        num_correct,
        num_questions,
        accuracy,
        loss,
    )

    return {
        "accuracy": accuracy,
        "loss": loss,
        "num_questions": num_questions,
        "num_correct": num_correct,
        "per_question": per_question,
    }


def _compute_open_ended_metrics(result_probs, covered_points, segment_metadata):
    """Compute open-ended mean normalized log-likelihood from per-segment results.

    segment_metadata maps local segment IDs (0-based) to
    (question_idx, answer_str, num_answer_tokens).
    """
    num_questions = len(segment_metadata)
    per_question = []
    total_norm_ll = 0.0
    num_covered = 0

    for seg_id, (q_idx, answer, n_tokens) in segment_metadata.items():
        if not covered_points[seg_id]:
            per_question.append({"question_idx": q_idx, "answer": answer, "norm_ll": None})
            continue

        ll = float(result_probs[seg_id])
        norm_ll = ll / max(n_tokens, 1)
        total_norm_ll += norm_ll
        num_covered += 1

        per_question.append(
            {
                "question_idx": q_idx,
                "answer": answer,
                "log_likelihood": ll,
                "norm_ll": norm_ll,
                "num_answer_tokens": n_tokens,
            }
        )

    mean_ll = total_norm_ll / max(num_covered, 1)
    logger.info("Open-ended mean normalized LL: %.4f (%d questions)", mean_ll, num_covered)

    return {
        "mean_ll": mean_ll,
        "num_questions": num_questions,
        "num_covered": num_covered,
        "per_question": per_question,
    }


def evaluate_all_vlm_benchmarks(
    jit_fn,
    model,
    prepared,
    tokenizer,
    EvalPos,
    EvalBatch,
    max_packed_segments,
    axis_resources,
    vocab_size: int,
    eval_run_uuid: str = "n/a",
):
    """Evaluate all VLM benchmarks in a single continuous batch loop.

    Merges all completions from all benchmarks into one flat list, runs one
    packing/batching/eval loop, then splits results by benchmark for
    per-benchmark metric computation.  Running everything in one loop avoids
    the benchmark-transition TPU crash ("unexpected peer with different launch
    id") that occurs when separate loops are run back-to-back.
    """
    all_completions = []
    bench_slices = {}
    if jax.process_index() == 0:
        for bench_name, (task_type, completions, metadata) in prepared.items():
            if not completions:
                continue
            bench_start = len(all_completions)

            # Explicitly map segment IDs by completion list index (0..N-1) for
            # each benchmark, then map to global ids by adding bench_start.
            # This avoids relying on implicit "offset + original segment_id"
            # behavior that can silently break when prepare_* changes.
            original_seg_ids = []
            remapped_completions = []
            for local_idx, pc in enumerate(completions):
                original_seg_ids.append(pc.segment_id if pc.segment_id is not None else local_idx)
                remapped_completions.append(
                    PromptCompletion(
                        ids=pc.ids,
                        prompt_length=pc.prompt_length,
                        segment_id=bench_start + local_idx,
                    )
                )

            if len(set(original_seg_ids)) != len(original_seg_ids):
                raise RuntimeError(
                    f"[{eval_run_uuid}] Duplicate segment ids in benchmark={bench_name}: "
                    f"sample={original_seg_ids[:10]}"
                )

            original_to_local = {seg_id: idx for idx, seg_id in enumerate(original_seg_ids)}
            remapped_metadata = {}
            for old_seg_id, meta in metadata.items():
                if old_seg_id not in original_to_local:
                    raise RuntimeError(
                        f"[{eval_run_uuid}] Metadata segment id missing from completions in benchmark={bench_name}: "
                        f"seg_id={old_seg_id}"
                    )
                remapped_metadata[original_to_local[old_seg_id]] = meta

            if len(remapped_metadata) != len(remapped_completions):
                raise RuntimeError(
                    f"[{eval_run_uuid}] Benchmark={bench_name} completion/metadata size mismatch: "
                    f"completions={len(remapped_completions)}, metadata={len(remapped_metadata)}"
                )

            all_completions.extend(remapped_completions)
            bench_slices[bench_name] = (bench_start, len(remapped_completions), task_type, remapped_metadata)

    # One sync for the combined completion count
    local_n = len(all_completions)
    expected_n = multihost_broadcast_sync(local_n)
    if jax.process_index() == 0 and local_n != expected_n:
        raise RuntimeError(
            f"Host {jax.process_index()} has {local_n} completions but leader has {expected_n}. "
            "Likely a GCS download inconsistency — all hosts must see identical data."
        )
    if expected_n == 0:
        return {}

    result_probs, covered_points = _run_batch_loop(
        jit_fn,
        model,
        all_completions,
        tokenizer,
        EvalPos,
        EvalBatch,
        max_packed_segments,
        axis_resources,
        vocab_size,
        eval_run_uuid=eval_run_uuid,
    )

    results = {}
    for bench_name, (start, length, task_type, metadata) in bench_slices.items():
        bench_probs = result_probs[start : start + length]
        bench_covered = covered_points[start : start + length]

        if task_type == "multiple_choice":
            results[bench_name] = _compute_mc_metrics(bench_probs, bench_covered, metadata)
        elif task_type == "open_ended":
            results[bench_name] = _compute_open_ended_metrics(bench_probs, bench_covered, metadata)

    return results


MC_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


# Benchmark parquet paths
DEFAULT_EVAL_PARQUET_PATH = "gs://marin-vlm/eval_benchmarks_tokenized"
MC_BENCHMARKS = ["ai2d", "mmmu", "seedbench_image"]
OPEN_ENDED_BENCHMARKS = ["textvqa", "chartqa"]
VLM_EVAL_BENCHMARKS = MC_BENCHMARKS + OPEN_ENDED_BENCHMARKS


# ---------------------------------------------------------------------------
# Build PromptCompletion from VLM parquet rows
# ---------------------------------------------------------------------------


def _build_user_token_ids(
    messages: list[dict],
    image_token_lists: list[list[int]],
    tokenizer,
) -> list[int]:
    """Build the prompt token IDs from user message content (images + text).

    Interleaves visual tokens and text tokens in the order they appear in the
    user message, matching the logic in tokenize_eval_benchmarks.build_understanding_sequence().
    """
    user_content = []
    for msg in messages:
        if msg["role"] == "user":
            user_content = msg["content"]
            break

    user_ids: list[int] = []
    image_idx = 0
    for part in user_content:
        if part["type"] == "image":
            if image_idx < len(image_token_lists):
                raw_tokens = image_token_lists[image_idx]
                shifted = [t + VISUAL_TOKEN_OFFSET for t in raw_tokens]
                user_ids.append(VISION_START_ID)
                user_ids.extend(shifted)
                user_ids.append(VISION_END_ID)
                image_idx += 1
        elif part["type"] == "text" and part.get("text"):
            text_ids = tokenizer.encode(part["text"], add_special_tokens=False)
            user_ids.extend(text_ids)

    return user_ids


def build_vlm_prompt_completions(
    messages: list[dict],
    image_token_lists: list[list[int]],
    choices: list[str],
    tokenizer,
    segment_id_start: int,
) -> list[PromptCompletion]:
    """For an MC question with K options, create K PromptCompletion objects.

    Uses Modified MCQ format (TASK_OVERVIEW.md Format 3): each completion is
    the full "letter. choice_text" string (e.g., "B. London") rather than just
    the letter. This gives smoother loss curves (R²=0.61 vs 0.28 for letter-only)
    while preserving accuracy.

    Args:
        messages: Message list from parquet (user content with image placeholders + text).
        image_token_lists: Pre-encoded TokLIP image tokens per image.
        choices: List of choice strings (e.g., ['Paris', 'London', 'Berlin', 'Rome']).
        tokenizer: Tokenizer for encoding text.
        segment_id_start: Starting segment ID for this question's completions.

    Returns:
        List of K PromptCompletion objects, one per option.
    """
    prompt_ids = _build_user_token_ids(messages, image_token_lists, tokenizer)
    separator_ids = tokenizer.encode("\nAnswer: ", add_special_tokens=False)
    prompt_with_sep = prompt_ids + separator_ids

    completions = []
    for opt_idx in range(len(choices)):
        letter = MC_LETTERS[opt_idx] if opt_idx < len(MC_LETTERS) else chr(ord("A") + opt_idx)
        # Modified MCQ: completion = "A. choice_text" (not just "A")
        completion_text = f"{letter}. {choices[opt_idx]}"
        option_ids = tokenizer.encode(completion_text, add_special_tokens=False)

        all_ids = prompt_with_sep + option_ids
        completions.append(
            PromptCompletion(
                ids=all_ids,
                prompt_length=len(prompt_with_sep),
                segment_id=segment_id_start + opt_idx,
            )
        )

    return completions


def build_vlm_open_ended_completion(
    messages: list[dict],
    image_token_lists: list[list[int]],
    answer: str,
    tokenizer,
    segment_id: int,
) -> PromptCompletion:
    """For open-ended evaluation, create 1 PromptCompletion per question.

    Completion = full gold answer text (not a single option letter).
    Metric: length-normalized log P(answer | prompt + image).
    """
    prompt_ids = _build_user_token_ids(messages, image_token_lists, tokenizer)
    separator_ids = tokenizer.encode("\nAnswer: ", add_special_tokens=False)
    prompt_with_sep = prompt_ids + separator_ids
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    all_ids = prompt_with_sep + answer_ids
    return PromptCompletion(
        ids=all_ids,
        prompt_length=len(prompt_with_sep),
        segment_id=segment_id,
    )


def _resolve_correct_option_index(choices: list[str], answer: str | None) -> int | None:
    """Determine which option index (0-based) is correct.

    Handles both letter-based answers (A/B/C/D) and index-based answers.
    """
    if answer is None:
        return None

    # Letter-based: A=0, B=1, ...
    if len(answer) == 1 and answer.upper() in "ABCDEFGH":
        return ord(answer.upper()) - ord("A")

    # Integer index
    try:
        idx = int(answer)
        if 0 <= idx < len(choices):
            return idx
    except (ValueError, TypeError):
        pass

    # Direct match against choices
    for idx, choice in enumerate(choices):
        if choice.strip().lower() == answer.strip().lower():
            return idx

    logger.warning("Could not resolve correct option for answer=%r, choices=%r", answer, choices)
    return None


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------


def prepare_vlm_mc_data(
    parquet_paths: list[str],
    tokenizer,
    max_len: int,
) -> tuple[list[PromptCompletion], dict[int, tuple[int, int, bool, int]]]:
    """Load parquets and build PromptCompletion objects for MC evaluation.

    This is pure CPU/IO work (no JAX collectives) and should be called BEFORE
    entering the coordinator/worker pattern to avoid collective timeouts.

    Returns:
        Tuple of (all_completions, segment_metadata) where segment_metadata maps
        segment_id → (question_idx, option_idx, is_correct, num_completion_tokens).
    """
    all_completions: list[PromptCompletion] = []
    segment_metadata: dict[int, tuple[int, int, bool, int]] = {}
    question_idx = 0
    segment_counter = 0
    skipped = 0

    for parquet_path in parquet_paths:
        table = _load_parquet(parquet_path)
        messages_col = table.column("messages")
        image_tokens_col = table.column("image_tokens")
        task_type_col = table.column("task_type")

        has_choices = "choices" in table.column_names
        has_answer = "answer" in table.column_names
        if not has_choices or not has_answer:
            logger.warning("Parquet %s missing choices/answer columns, skipping", parquet_path)
            continue

        choices_col = table.column("choices")
        answer_col = table.column("answer")

        for i in range(len(table)):
            task_type = task_type_col[i].as_py()
            if task_type != "multiple_choice":
                continue

            choices = choices_col[i].as_py()
            if not choices or len(choices) < 2:
                skipped += 1
                continue

            answer = answer_col[i].as_py()
            correct_idx = _resolve_correct_option_index(choices, answer)
            if correct_idx is None:
                skipped += 1
                continue

            messages = messages_col[i].as_py()
            image_token_lists = image_tokens_col[i].as_py()

            completions = build_vlm_prompt_completions(
                messages,
                image_token_lists,
                choices,
                tokenizer,
                segment_id_start=segment_counter,
            )

            too_long = any(len(pc.ids) > max_len for pc in completions)
            if too_long:
                logger.warning(
                    "Question %d has sequences exceeding max_length=%d, skipping",
                    question_idx,
                    max_len,
                )
                skipped += 1
                continue

            for opt_idx, pc in enumerate(completions):
                num_completion_tokens = len(pc.ids) - pc.prompt_length
                segment_metadata[pc.segment_id] = (
                    question_idx,
                    opt_idx,
                    opt_idx == correct_idx,
                    num_completion_tokens,
                )

            all_completions.extend(completions)
            segment_counter += len(completions)
            question_idx += 1

    logger.info(
        "Built %d MC completions for %d questions (%d skipped)",
        len(all_completions),
        question_idx,
        skipped,
    )
    return all_completions, segment_metadata


def evaluate_vlm_mc_benchmark(
    jit_fn,
    model,
    all_completions: list[PromptCompletion],
    segment_metadata: dict[int, tuple[int, int, bool, int]],
    tokenizer,
    EvalPos,
    EvalBatch,
    max_packed_segments,
    axis_resources,
) -> dict:
    """Evaluate pre-prepared MC data using per-option log-likelihood comparison.

    For multi-benchmark evaluation, prefer evaluate_all_vlm_benchmarks() which
    runs all benchmarks in a single loop to avoid TPU crashes at transitions.
    """
    if not all_completions:
        return {"accuracy": 0.0, "loss": 0.0, "num_questions": 0, "num_correct": 0}

    result_probs, covered_points = _run_batch_loop(
        jit_fn,
        model,
        all_completions,
        tokenizer,
        EvalPos,
        EvalBatch,
        max_packed_segments,
        axis_resources,
        len(tokenizer),
    )
    return _compute_mc_metrics(result_probs, covered_points, segment_metadata)


# ---------------------------------------------------------------------------
# Open-ended evaluation
# ---------------------------------------------------------------------------


def prepare_vlm_open_ended_data(
    parquet_paths: list[str],
    tokenizer,
    max_len: int,
) -> tuple[list[PromptCompletion], dict[int, tuple[int, str, int]]]:
    """Load parquets and build PromptCompletion objects for open-ended evaluation.

    This is pure CPU/IO work (no JAX collectives) and should be called BEFORE
    entering the coordinator/worker pattern to avoid collective timeouts.

    Returns:
        Tuple of (all_completions, segment_metadata) where segment_metadata maps
        segment_id → (question_idx, answer_str, num_answer_tokens).
    """
    all_completions: list[PromptCompletion] = []
    segment_metadata: dict[int, tuple[int, str, int]] = {}
    question_idx = 0
    segment_counter = 0
    skipped = 0

    for parquet_path in parquet_paths:
        table = _load_parquet(parquet_path)
        messages_col = table.column("messages")
        image_tokens_col = table.column("image_tokens")
        task_type_col = table.column("task_type")

        has_answer = "answer" in table.column_names
        if not has_answer:
            logger.warning("Parquet %s missing answer column, skipping", parquet_path)
            continue

        answer_col = table.column("answer")

        for i in range(len(table)):
            task_type = task_type_col[i].as_py()
            if task_type != "open_ended":
                continue

            answer = answer_col[i].as_py()
            if not answer:
                skipped += 1
                continue

            messages = messages_col[i].as_py()
            image_token_lists = image_tokens_col[i].as_py()

            pc = build_vlm_open_ended_completion(
                messages,
                image_token_lists,
                answer,
                tokenizer,
                segment_id=segment_counter,
            )

            if len(pc.ids) > max_len:
                logger.warning(
                    "Question %d has %d tokens (max %d), skipping",
                    question_idx,
                    len(pc.ids),
                    max_len,
                )
                skipped += 1
                continue

            num_answer_tokens = len(pc.ids) - pc.prompt_length
            segment_metadata[pc.segment_id] = (question_idx, answer, num_answer_tokens)

            all_completions.append(pc)
            segment_counter += 1
            question_idx += 1

    logger.info(
        "Built %d open-ended completions for %d questions (%d skipped)",
        len(all_completions),
        question_idx,
        skipped,
    )
    return all_completions, segment_metadata


def evaluate_vlm_open_ended_benchmark(
    jit_fn,
    model,
    all_completions: list[PromptCompletion],
    segment_metadata: dict[int, tuple[int, str, int]],
    tokenizer,
    EvalPos,
    EvalBatch,
    max_packed_segments,
    axis_resources,
) -> dict:
    """Evaluate pre-prepared open-ended data using length-normalized log-likelihood.

    For multi-benchmark evaluation, prefer evaluate_all_vlm_benchmarks() which
    runs all benchmarks in a single loop to avoid TPU crashes at transitions.
    """
    if not all_completions:
        return {"mean_ll": 0.0, "num_questions": 0, "num_covered": 0}

    result_probs, covered_points = _run_batch_loop(
        jit_fn,
        model,
        all_completions,
        tokenizer,
        EvalPos,
        EvalBatch,
        max_packed_segments,
        axis_resources,
        len(tokenizer),
    )
    return _compute_open_ended_metrics(result_probs, covered_points, segment_metadata)


# ---------------------------------------------------------------------------
# Parquet loading
# ---------------------------------------------------------------------------


def _load_parquet(parquet_path: str):
    """Load a parquet table from a local or GCS path."""
    if parquet_path.startswith("gs://"):
        from experiments.unified.vlm_tokenize_captions import gcs_download

        tmp = tempfile.mkdtemp(prefix="vlm_mc_eval_")
        local = os.path.join(tmp, "shard.parquet")
        gcs_download(parquet_path, local)
        return pq.read_table(local)
    return pq.read_table(parquet_path)


def find_benchmark_parquets(benchmark: str, base_path: str = DEFAULT_EVAL_PARQUET_PATH) -> list[str]:
    """Find all parquet shards for a benchmark."""
    pattern = f"{base_path}/{benchmark}/eval-{benchmark}-*.parquet"
    paths = sorted(fsspec_glob(pattern))
    if not paths:
        logger.warning("No parquets found for %s (pattern: %s)", benchmark, pattern)
    return paths


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------


def _infer_wandb_info(checkpoint_path: str) -> tuple[str | None, int | None]:
    """Infer WandB run ID and training step from a checkpoint path.

    Handles two path layouts:

    1. Levanter standalone: ``{base_path}/{run_id}/step-{N}/``
    2. Marin executor: ``{bucket}/checkpoints/{run_id}/checkpoints/step-{N}/``
       or ``{bucket}/checkpoints/{run_id}/hf/step-{N}/``

    In both cases the WandB run ID equals the Marin executor step name (set by
    ``impute_run_id_from_output_path`` in ``TrainLmOnPodConfig``).  When the
    immediate parent of ``step-{N}`` is a known internal directory (``checkpoints``
    or ``hf``), we look one level higher for the actual run ID.
    """
    import re

    parts = checkpoint_path.rstrip("/").split("/")

    step = None
    step_idx = None
    for i, part in enumerate(parts):
        m = re.match(r"step[-_](\d+)$", part)
        if m:
            step = int(m.group(1))
            step_idx = i
            break

    run_id = None
    if step_idx is not None and step_idx > 0:
        candidate = parts[step_idx - 1]
        # Skip known Marin/Levanter internal subdirectory names
        if candidate in ("checkpoints", "hf") and step_idx > 1:
            run_id = parts[step_idx - 2]
        else:
            run_id = candidate

    return run_id, step


def main():
    parser = argparse.ArgumentParser(description="VLM benchmark evaluation using log-likelihood comparison.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--checkpoint_is_hf", action="store_true", help="Checkpoint is HuggingFace format")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=VLM_EVAL_BENCHMARKS,
        help=f"Benchmarks to evaluate (default: {VLM_EVAL_BENCHMARKS})",
    )
    parser.add_argument("--eval_parquet_path", type=str, default=DEFAULT_EVAL_PARQUET_PATH)
    parser.add_argument(
        "--eval_batch_size", type=int, default=None, help="Eval batch size (default: 1 per device = data_axis_size)"
    )
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--output_json", type=str, default=None, help="Path to write results JSON")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer path (default: inferred from checkpoint or unified tokenizer)",
    )
    parser.add_argument(
        "--wandb_project", type=str, default="marin", help="WandB project name (default: marin, same as training)"
    )
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity/team")
    parser.add_argument(
        "--wandb_run_id", type=str, default=None, help="WandB run ID (default: inferred from checkpoint path)"
    )
    parser.add_argument(
        "--train_step", type=int, default=None, help="Training step (default: inferred from checkpoint path)"
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging even if run_id can be inferred")
    args = parser.parse_args()

    import typing

    import equinox as eqx

    from levanter.checkpoint import load_checkpoint
    from levanter.compat.hf_checkpoints import HFCheckpointConverter, load_tokenizer
    from levanter.distributed import RayConfig
    from levanter.main.train_lm import TrainerConfig
    from levanter.models.lm_model import LmHeadModel
    from levanter.utils.jax_utils import use_cpu_device
    from levanter.utils.tree_utils import inference_mode

    from experiments.unified.unified_pretrain import UNIFIED_TOKENIZER_PATH

    tokenizer_path = args.tokenizer or UNIFIED_TOKENIZER_PATH
    tokenizer = load_tokenizer(tokenizer_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    import jmp

    # Match text eval harness: per_device_eval_parallelism=1 keeps memory safe.
    trainer_config = TrainerConfig(
        mp=jmp.get_policy("p=f32,c=bfloat16,o=bfloat16"),
        per_device_eval_parallelism=1,
        ray=RayConfig(auto_start_cluster=False),
    )
    trainer_config.initialize()

    compute_axis_mapping = trainer_config.compute_axis_mapping
    parameter_axis_mapping = trainer_config.parameter_axis_mapping

    # Default: 1 per device (matching text eval harness).
    # --eval_batch_size overrides, but must be >= data_axis_size and evenly divisible.
    data_axis_size = trainer_config.data_axis_size
    if args.eval_batch_size is not None:
        requested = args.eval_batch_size
        adjusted = max(requested, data_axis_size)
        adjusted = (adjusted // data_axis_size) * data_axis_size
        effective_eval_batch_size = adjusted
        if adjusted != requested:
            logger.info(
                "Adjusted eval_batch_size from %d to %d (must be multiple of data_axis_size=%d)",
                requested,
                adjusted,
                data_axis_size,
            )
    else:
        effective_eval_batch_size = trainer_config.eval_batch_size  # = 1 * data_axis_size
    EvalBatch = hax.Axis("batch", effective_eval_batch_size)

    with trainer_config.use_device_mesh(), hax.axis_mapping(parameter_axis_mapping):
        from haliax.partitioning import round_axis_for_partitioning

        key = jax.random.PRNGKey(0)
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(hax.Axis("vocab", vocab_size), compute_axis_mapping)

        if args.checkpoint_is_hf:
            # HuggingFace checkpoint loading
            from experiments.qwen3 import Qwen3Config as ModelConfig

            model_config = ModelConfig()
            converter: HFCheckpointConverter = model_config.hf_checkpoint_converter()
            converter = converter.replaced(reference_checkpoint=args.checkpoint_path, tokenizer=tokenizer)
            model = converter.load_pretrained(
                model_config.model_type,
                ref=args.checkpoint_path,
                dtype=trainer_config.mp.compute_dtype,
                axis_mapping=parameter_axis_mapping,
            )
        else:
            from experiments.qwen3 import Qwen3Config as ModelConfig

            # Default Qwen3Config() uses LlamaConfig defaults (4096 hidden, 32 layers).
            # For non-default model sizes, use --checkpoint_is_hf with an HF checkpoint
            # which reads the correct config from config.json.
            model_config = ModelConfig()
            logger.warning(
                "Using default Qwen3Config (hidden_dim=%d, num_layers=%d). "
                "If this doesn't match your model, use --checkpoint_is_hf instead.",
                model_config.hidden_dim,
                model_config.num_layers,
            )
            with use_cpu_device():
                model = eqx.filter_eval_shape(model_config.build, Vocab, key=key)
                model = load_checkpoint(model, args.checkpoint_path, subpath="model")
            model = hax.shard(model, parameter_axis_mapping)

        model = typing.cast(LmHeadModel, inference_mode(model, True))
        # Use default attention backend (SPLASH on TPU) — same as text eval harness.
        # Vanilla attention materializes O(seq²) attention matrices, causing OOM on v4-8.

        gc.collect()  # Free lingering state_dict buffers from load_pretrained

        EvalPos = model.Pos if args.max_length is None else model.Pos.resize(args.max_length)

        # All processes prepare identical data independently, then eval in lockstep.
        prepared: dict[str, tuple] = {}
        for bench_name in args.benchmarks:
            parquet_paths = find_benchmark_parquets(bench_name, args.eval_parquet_path)
            if not parquet_paths:
                continue

            task_type = _detect_benchmark_task_type(parquet_paths[0])
            logger.info("Preparing %s [%s] (%d parquet shards)", bench_name, task_type, len(parquet_paths))

            if task_type == "multiple_choice":
                completions, metadata = prepare_vlm_mc_data(parquet_paths, tokenizer, EvalPos.size)
                prepared[bench_name] = ("multiple_choice", completions, metadata)
            elif task_type == "open_ended":
                completions, metadata = prepare_vlm_open_ended_data(parquet_paths, tokenizer, EvalPos.size)
                prepared[bench_name] = ("open_ended", completions, metadata)
            else:
                logger.warning("Unsupported task_type=%s for %s, skipping", task_type, bench_name)

        max_packed_segments = 64
        jit_fn = _make_vlm_eval_jit(compute_axis_mapping, trainer_config.mp, EvalPos, EvalBatch, max_packed_segments)

        results = evaluate_all_vlm_benchmarks(
            jit_fn,
            model,
            prepared,
            tokenizer,
            EvalPos,
            EvalBatch,
            max_packed_segments,
            compute_axis_mapping,
            len(tokenizer),
        )

        if jax.process_index() == 0:
            # Print summary
            logger.info("=" * 60)
            logger.info("VLM Evaluation Results:")
            for bench, res in results.items():
                if "accuracy" in res:
                    logger.info(
                        "  %s [MC]: %.2f%% (%d/%d), loss=%.4f",
                        bench,
                        res["accuracy"] * 100,
                        res["num_correct"],
                        res["num_questions"],
                        res["loss"],
                    )
                elif "mean_ll" in res:
                    logger.info(
                        "  %s [open-ended]: mean_ll=%.4f (%d questions)",
                        bench,
                        res["mean_ll"],
                        res["num_questions"],
                    )

            if args.output_json:
                compact = {
                    bench: {k: v for k, v in res.items() if k != "per_question"} for bench, res in results.items()
                }
                with open(args.output_json, "w") as f:
                    json.dump(compact, f, indent=2)
                logger.info("Results written to %s", args.output_json)

            # Log to the training WandB run
            inferred_run_id, inferred_step = _infer_wandb_info(args.checkpoint_path)
            wandb_run_id = args.wandb_run_id or inferred_run_id
            train_step = args.train_step if args.train_step is not None else inferred_step

            if wandb_run_id and not args.no_wandb:
                import wandb

                wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    id=wandb_run_id,
                    resume="must",
                )

                metrics = {}
                for bench, res in results.items():
                    if "accuracy" in res:
                        metrics[f"eval/{bench}/accuracy"] = res["accuracy"]
                        metrics[f"eval/{bench}/loss"] = res["loss"]
                    elif "mean_ll" in res:
                        metrics[f"eval/{bench}/mean_ll"] = res["mean_ll"]
                    metrics[f"eval/{bench}/num_questions"] = res["num_questions"]

                wandb.log(metrics, step=train_step)
                wandb.finish()
                logger.info("Logged eval results to WandB run %s (step=%s)", wandb_run_id, train_step)
            elif not args.no_wandb:
                logger.info("Could not infer WandB run ID from checkpoint path. Use --wandb_run_id to log to WandB.")

        logger.info("Done.")


# ---------------------------------------------------------------------------
# Training callback factory
# ---------------------------------------------------------------------------


def _detect_benchmark_task_type(parquet_path: str) -> str:
    """Peek at the first row of a benchmark parquet to determine its task_type."""
    table = _load_parquet(parquet_path)
    return table.column("task_type")[0].as_py()


def vlm_mc_eval_callback(
    benchmarks: list[str],
    parquet_base_path: str,
    tokenizer,
    EvalBatch,
    axis_resources,
    mp,
    max_length: int | None = None,
):
    """Factory that returns a training callback for VLM evaluation.

    Auto-detects each benchmark's task type from its parquet data:
    - ``multiple_choice``: evaluates MC accuracy via log-likelihood comparison.
    - ``open_ended``: evaluates mean length-normalized log P(answer | prompt + image).

    Args:
        benchmarks: List of benchmark names (e.g., ["ai2d", "mmmu", "textvqa"]).
        parquet_base_path: Base GCS path to tokenized eval parquets.
        tokenizer: Tokenizer for the model.
        EvalBatch: Batch axis for evaluation.
        axis_resources: Resource mapping for distributed computation.
        mp: Mixed precision policy.
        max_length: Override for max sequence length (None = use model's Pos).
    """
    import levanter.tracker
    from levanter.callbacks import StepInfo
    from levanter.utils.jax_utils import parameter_count
    from levanter.utils.tree_utils import inference_mode

    # Cache JITs by (EvalPos, EvalBatch) to support VLM-only safe batch-size override.
    _cached_jit: dict[tuple[int, int], object] = {}

    # Pre-resolve parquet paths and detect task types
    bench_parquets: dict[str, list[str]] = {}
    bench_task_types: dict[str, str] = {}
    for bench_name in benchmarks:
        paths = find_benchmark_parquets(bench_name, parquet_base_path)
        if paths:
            bench_parquets[bench_name] = paths
            bench_task_types[bench_name] = _detect_benchmark_task_type(paths[0])
            logger.info("Benchmark %s: task_type=%s, %d shards", bench_name, bench_task_types[bench_name], len(paths))
        else:
            logger.warning("No parquets found for benchmark %s, skipping", bench_name)

    def vlm_eval(step: StepInfo, force=False):
        local_uuid = uuid.uuid4().hex[:8] if jax.process_index() == 0 else ""
        eval_run_uuid = multihost_broadcast_sync(local_uuid)
        model = inference_mode(step.eval_model, True)
        # Use default attention backend (inherited from training config).
        EvalPos = model.Pos if max_length is None else model.Pos.resize(max_length)
        # Keep multi-host defaults aligned with text-mc style settings unless
        # explicitly overridden via environment.
        default_safe_batch = "8" if jax.process_count() > 1 else "32"
        safe_eval_batch_size = int(os.environ.get("VLM_EVAL_SAFE_BATCH_SIZE", default_safe_batch))
        if safe_eval_batch_size <= 0:
            raise ValueError(f"VLM_EVAL_SAFE_BATCH_SIZE must be positive, got {safe_eval_batch_size}")
        effective_eval_batch_size = min(EvalBatch.size, safe_eval_batch_size)
        VlmEvalBatch = EvalBatch.resize(effective_eval_batch_size)

        try:
            abstract_mesh = jax.sharding.get_abstract_mesh()
        except Exception as e:
            abstract_mesh = f"<unavailable: {type(e).__name__}: {e}>"

        local_device_count = jax.local_device_count()
        platform = jax.devices()[0].platform if jax.devices() else "unknown"
        logger.info(
            "[%s][host=%d/%d] VLM eval env: local_device_count=%d platform=%s abstract_mesh=%s axis_resources=%s "
            "EvalBatch=%d VlmEvalBatch=%d EvalPos=%d",
            eval_run_uuid,
            jax.process_index(),
            jax.process_count(),
            local_device_count,
            platform,
            abstract_mesh,
            axis_resources,
            EvalBatch.size,
            VlmEvalBatch.size,
            EvalPos.size,
        )

        logger.info(
            "[%s] Running VLM eval at step %d (EvalPos=%d, EvalBatch=%d, VlmEvalBatch=%d, params=%s)",
            eval_run_uuid,
            step.step,
            EvalPos.size,
            EvalBatch.size,
            VlmEvalBatch.size,
            f"{parameter_count(model):,}",
        )
        run_file = __file__
        git_commit = os.environ.get("GIT_COMMIT", "unknown")
        ray_pkg_fragment = "n/a"
        marker = "_ray_pkg_"
        if marker in run_file:
            ray_pkg_fragment = run_file[run_file.find(marker) : run_file.find(marker) + 24]
        logger.info(
            "[%s][host=%d] Code provenance: file=%s git_commit=%s ray_pkg=%s",
            eval_run_uuid,
            jax.process_index(),
            run_file,
            git_commit,
            ray_pkg_fragment,
        )
        logger.warning(
            "VLM_EVAL_PROVENANCE eval_run_uuid=%s host=%d process_count=%d file=%s ray_pkg=%s git_commit=%s "
            "axis_resources_crc32=%s",
            eval_run_uuid,
            jax.process_index(),
            jax.process_count(),
            run_file,
            ray_pkg_fragment,
            git_commit,
            int(zlib.crc32(json.dumps(str(axis_resources), sort_keys=True).encode())),
        )
        logger.warning(
            "VLM_HLO_DUMP_WINDOW_START eval_run_uuid=%s host=%d xla_flags=%s expected_module_regex=%s "
            "expected_dump_dir=%s",
            eval_run_uuid,
            jax.process_index(),
            os.environ.get("XLA_FLAGS", ""),
            ".*vlm_eval_loglikelihood.*",
            "/tmp/xla_dumps_vlm_once",
        )
        # Process 0 prepares data; all hosts consume the same batch stream via
        # broadcast_shard(source=0) in _run_batch_loop.
        prepared: dict[str, tuple] = {}
        if jax.process_index() == 0:
            for bench_name, parquet_paths in bench_parquets.items():
                task_type = bench_task_types[bench_name]
                logger.info("Preparing %s [%s] (%d shards)", bench_name, task_type, len(parquet_paths))

                if task_type == "multiple_choice":
                    completions, metadata = prepare_vlm_mc_data(
                        parquet_paths,
                        tokenizer,
                        EvalPos.size,
                    )
                    prepared[bench_name] = ("multiple_choice", completions, metadata)
                elif task_type == "open_ended":
                    completions, metadata = prepare_vlm_open_ended_data(
                        parquet_paths,
                        tokenizer,
                        EvalPos.size,
                    )
                    prepared[bench_name] = ("open_ended", completions, metadata)
                else:
                    logger.warning("Unknown task_type=%s for %s, skipping", task_type, bench_name)

        default_max_packed_segments = "32" if jax.process_count() > 1 else "64"
        max_packed_segments = int(os.environ.get("VLM_EVAL_MAX_PACKED_SEGMENTS", default_max_packed_segments))
        if max_packed_segments <= 0:
            raise ValueError(f"VLM_EVAL_MAX_PACKED_SEGMENTS must be positive, got {max_packed_segments}")
        logger.warning(
            "VLM_EVAL_STABILITY_CONFIG eval_run_uuid=%s host=%d safe_batch=%d max_packed_segments=%d "
            "process_count=%d",
            eval_run_uuid,
            jax.process_index(),
            effective_eval_batch_size,
            max_packed_segments,
            jax.process_count(),
        )

        # Create JIT function on first call for this (Pos, Batch) shape pair.
        jit_key = (EvalPos.size, VlmEvalBatch.size)
        if jit_key not in _cached_jit:
            _cached_jit[jit_key] = _make_vlm_eval_jit(axis_resources, mp, EvalPos, VlmEvalBatch, max_packed_segments)
        jit_fn = _cached_jit[jit_key]

        # Run ALL benchmarks in a single continuous batch loop to avoid the
        # TPU crash that occurs at benchmark transitions.
        try:
            results = evaluate_all_vlm_benchmarks(
                jit_fn,
                model,
                prepared,
                tokenizer,
                EvalPos,
                VlmEvalBatch,
                max_packed_segments,
                axis_resources,
                len(tokenizer),
                eval_run_uuid=eval_run_uuid,
            )
            if jax.process_index() == 0:
                metrics_to_log: dict[str, float] = {}
                for bench_name, result in results.items():
                    task_type = prepared[bench_name][0]
                    if task_type == "multiple_choice":
                        accuracy = result["accuracy"]
                        loss = result["loss"]
                        logger.info(
                            "%s: accuracy=%.4f (%d/%d), loss=%.4f",
                            bench_name,
                            accuracy,
                            result["num_correct"],
                            result["num_questions"],
                            loss,
                        )
                        metrics_to_log[f"eval/{bench_name}/accuracy"] = accuracy
                        metrics_to_log[f"eval/{bench_name}/loss"] = loss
                        logger.warning(
                            "VLM_EVAL_RESULTS eval_run_uuid=%s host=%d benchmark=%s task=multiple_choice "
                            "accuracy=%.6f loss=%.6f num_correct=%d num_questions=%d",
                            eval_run_uuid,
                            jax.process_index(),
                            bench_name,
                            accuracy,
                            loss,
                            result["num_correct"],
                            result["num_questions"],
                        )
                    elif task_type == "open_ended":
                        mean_ll = result["mean_ll"]
                        logger.info(
                            "%s: mean_ll=%.4f (%d questions)",
                            bench_name,
                            mean_ll,
                            result["num_questions"],
                        )
                        metrics_to_log[f"eval/{bench_name}/mean_ll"] = mean_ll
                        logger.warning(
                            "VLM_EVAL_RESULTS eval_run_uuid=%s host=%d benchmark=%s task=open_ended "
                            "mean_ll=%.6f num_questions=%d",
                            eval_run_uuid,
                            jax.process_index(),
                            bench_name,
                            mean_ll,
                            result["num_questions"],
                        )
                if metrics_to_log:
                    # Match text-mc/eval-harness behavior: avoid host0-only reads
                    # of step/state inside this callback.
                    levanter.tracker.log(metrics_to_log, step=None)
        except Exception:
            logger.exception("vlm_mc_eval crashed")
            raise
        finally:
            if jax.process_index() == 0:
                _log_vlm_hlo_fingerprints(eval_run_uuid=eval_run_uuid, dump_dir="/tmp/xla_dumps_vlm_once")
            else:
                logger.warning(
                    "VLM_HLO_FINGERPRINT_SKIPPED eval_run_uuid=%s host=%d reason=non_source_host",
                    eval_run_uuid,
                    jax.process_index(),
                )
            logger.warning(
                "VLM_HLO_DUMP_WINDOW_END eval_run_uuid=%s host=%d",
                eval_run_uuid,
                jax.process_index(),
            )

        logger.warning(
            "VLM_EVAL_CALLBACK_SYNC_SKIPPED eval_run_uuid=%s host=%d reason=text_mc_alignment",
            eval_run_uuid,
            jax.process_index(),
        )
        logger.warning("VLM_EVAL_CALLBACK_EXIT eval_run_uuid=%s host=%d", eval_run_uuid, jax.process_index())
        logger.info("VLM eval done.")

    return vlm_eval


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
