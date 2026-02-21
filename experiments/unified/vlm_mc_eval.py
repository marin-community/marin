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
Each host independently builds identical completions from GCS parquet files;
_shard_identical_batch uses jax.make_array_from_callback so each host only
provides data for its local devices. Results are split by benchmark afterward.

Usage (standalone):
    uv run experiments/unified/vlm_mc_eval.py \
        --checkpoint_path gs://marin-vlm/checkpoints/unified-qwen3-0.6b/step_10000 \
        --benchmarks ai2d mmmu textvqa chartqa \
        --eval_batch_size 16

    # MC only
    uv run experiments/unified/vlm_mc_eval.py \
        --checkpoint_path path/to/hf/checkpoint \
        --checkpoint_is_hf \
        --benchmarks ai2d mmmu
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq
from tqdm_loggable.auto import tqdm

import haliax as hax

from levanter.data.loader import stack_batches
from levanter.data.packing import PromptCompletion, greedy_pack_prompt_completions, per_segment_loss
from levanter.eval_harness import get_padding_count_from_batch
from levanter.models.loss import next_token_loss
from levanter.utils.jax_utils import multihost_broadcast_sync

from experiments.unified.vlm_tokenize_captions import (
    VISION_END_ID,
    VISION_START_ID,
    VISUAL_TOKEN_OFFSET,
)
from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)


def _shard_identical_batch(batch, axis_resources):
    """Create global arrays from identical local data present on every host.

    Uses jax.make_array_from_callback so each host only provides data for its
    own local devices — the same pattern as Levanter's DataLoader (loader.py:408).
    """
    shardings = hax.partitioning.infer_resource_partitions(batch, axis_resources)

    def _make_global(leaf, sharding):
        if sharding is None:
            return leaf
        if isinstance(leaf, hax.NamedArray):
            raw = np.asarray(leaf.array)
            global_arr = jax.make_array_from_callback(raw.shape, sharding, lambda idx: raw[idx])
            return hax.NamedArray(global_arr, leaf.axes)
        raw = np.asarray(leaf)
        return jax.make_array_from_callback(raw.shape, sharding, lambda idx: raw[idx])

    return jax.tree.map(_make_global, batch, shardings, is_leaf=lambda x: isinstance(x, hax.NamedArray))


def _make_vlm_eval_jit(EvalBatch, EvalPos, max_packed_segments, axis_resources, mp):
    """Create a JIT loglikelihood function for VLM evaluation.

    Uses out_axis_resources={} to replicate outputs on every host (needed for
    correct multi-host metrics).  The caller must run a post-loop sync JIT
    (all-reduce + .item()) after the batch loop to synchronize all hosts
    before returning to training — see _run_batch_loop.
    """

    def _eval_loglikelihood(model, packed_example):
        if mp is not None:
            model = mp.cast_to_compute(model)

        logits = model(packed_example.tokens, attn_mask=packed_example.attn_mask)
        logits = logits.astype(jnp.float32)
        Pos = logits.resolve_axis(EvalPos.name)

        loss = next_token_loss(
            Pos=Pos,
            Vocab=model.Vocab,
            logits=logits,
            true_ids=packed_example.tokens,
            loss_weight=packed_example.loss_weight,
            reduction=None,
        )

        max_Segments = hax.Axis("Segments", size=max_packed_segments + 1)

        batched_segment_ids, batched_per_segment_losses = hax.vmap(per_segment_loss, EvalBatch)(
            packed_example, loss, max_Segments
        )

        segments = hax.flatten(batched_segment_ids, "segment")
        losses = hax.flatten(batched_per_segment_losses, "segment")

        return segments, -losses

    return hax.named_jit(_eval_loglikelihood, axis_resources=axis_resources, out_axis_resources={})


def _run_batch_loop(jit_fn, model, all_completions, tokenizer, EvalPos, EvalBatch, max_packed_segments, axis_resources):
    """Pack completions, batch, sync across hosts, and run one continuous loop.

    Returns (result_probs, covered_points) numpy arrays indexed by completion
    list position.  This is the shared "hot loop" — keeping it as a single
    uninterrupted iteration prevents the TPU-crash that occurs when separate
    benchmark loops are run back-to-back.
    """
    pad_token = tokenizer.pad_token_id
    if pad_token is None:
        pad_token = tokenizer.eos_token_id

    packed = greedy_pack_prompt_completions(
        EvalPos,
        all_completions,
        pad_token=pad_token,
        max_segments_per_example=max_packed_segments,
    )

    all_batches = list(stack_batches(iter(packed), EvalPos, EvalBatch))

    num_batches = multihost_broadcast_sync(len(all_batches))
    if len(all_batches) < num_batches:
        dummy = all_batches[-1] if all_batches else None
        all_batches.extend([dummy] * (num_batches - len(all_batches)))
    elif len(all_batches) > num_batches:
        all_batches = all_batches[:num_batches]

    result_probs = np.zeros(len(all_completions))
    covered_points = np.zeros(len(all_completions), dtype=bool)

    total_tokens = len(packed) * EvalPos.size
    pbar = tqdm(total=total_tokens, desc="vlm_eval", unit="tok")

    for batch in all_batches:
        _padding_count, batch_tokens = get_padding_count_from_batch(batch, pad_token)
        batch = _shard_identical_batch(batch, axis_resources)
        out_ids, out_lls = jit_fn(model, batch)

        out_ids = jax.device_get(out_ids.array)
        out_lls = jax.device_get(out_lls.array)

        valid_indices = out_ids != -1
        result_probs[out_ids[valid_indices]] = out_lls[valid_indices]
        covered_points[out_ids[valid_indices]] = True
        pbar.update(batch_tokens)

    pbar.close()

    # Sync all hosts before returning — matches standard eval's post-loop
    # pattern (eval.py:414).  The all-reduce forces all hosts to participate;
    # .item() blocks until it completes on every host.  Without this, hosts
    # diverge in timing during post-loop Python code and the next training JIT
    # sees mismatched launch groups → TPU HALT.
    dummy = hax.zeros(EvalBatch, dtype=jnp.float32)
    dummy = _shard_identical_batch(dummy, axis_resources)
    _sync = hax.named_jit(lambda x: hax.mean(x).array, axis_resources=axis_resources)(dummy)
    _sync.item()
    logger.info("vlm_eval post-loop sync complete")

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


def evaluate_all_vlm_benchmarks(jit_fn, model, prepared, tokenizer, EvalPos, EvalBatch, max_packed_segments, axis_resources):
    """Evaluate all VLM benchmarks in a single continuous batch loop.

    Merges all completions from all benchmarks into one flat list, runs one
    packing/batching/eval loop, then splits results by benchmark for
    per-benchmark metric computation.  Running everything in one loop avoids
    the benchmark-transition TPU crash ("unexpected peer with different launch
    id") that occurs when separate loops are run back-to-back.
    """
    all_completions = []
    bench_slices = {}
    offset = 0
    for bench_name, (task_type, completions, metadata) in prepared.items():
        if not completions:
            continue
        bench_slices[bench_name] = (offset, len(completions), task_type, metadata)
        # Each prepare_* function assigns 0-based segment IDs.  When merging
        # multiple benchmarks we must offset them so they don't collide in the
        # packer (which uses segment_id for attention masking and result mapping).
        if offset == 0:
            all_completions.extend(completions)
        else:
            all_completions.extend(
                PromptCompletion(
                    ids=pc.ids,
                    prompt_length=pc.prompt_length,
                    segment_id=(pc.segment_id if pc.segment_id is not None else 0) + offset,
                )
                for pc in completions
            )
        offset += len(completions)

    if not all_completions:
        return {}

    # One sync for the combined completion count
    local_n = len(all_completions)
    expected_n = multihost_broadcast_sync(local_n)
    if local_n != expected_n:
        raise RuntimeError(
            f"Host {jax.process_index()} has {local_n} completions but leader has {expected_n}. "
            "Likely a GCS download inconsistency — all hosts must see identical data."
        )

    result_probs, covered_points = _run_batch_loop(
        jit_fn, model, all_completions, tokenizer, EvalPos, EvalBatch, max_packed_segments, axis_resources
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
MC_BENCHMARKS = ["ai2d", "mmmu"]
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

    completions = []
    for opt_idx in range(len(choices)):
        letter = MC_LETTERS[opt_idx] if opt_idx < len(MC_LETTERS) else chr(ord("A") + opt_idx)
        # Modified MCQ: completion = "A. choice_text" (not just "A")
        completion_text = f"{letter}. {choices[opt_idx]}"
        option_ids = tokenizer.encode(completion_text, add_special_tokens=False)

        all_ids = prompt_ids + option_ids
        completions.append(
            PromptCompletion(
                ids=all_ids,
                prompt_length=len(prompt_ids),
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
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    all_ids = prompt_ids + answer_ids
    return PromptCompletion(
        ids=all_ids,
        prompt_length=len(prompt_ids),
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
        jit_fn, model, all_completions, tokenizer, EvalPos, EvalBatch, max_packed_segments, axis_resources
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
        jit_fn, model, all_completions, tokenizer, EvalPos, EvalBatch, max_packed_segments, axis_resources
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
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--output_json", type=str, default=None, help="Path to write results JSON")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer path (default: inferred from checkpoint or unified tokenizer)",
    )
    args = parser.parse_args()

    import typing

    import equinox as eqx

    from levanter.checkpoint import load_checkpoint
    from levanter.compat.hf_checkpoints import HFCheckpointConverter, load_tokenizer
    from levanter.main.train_lm import TrainerConfig
    from levanter.models.lm_model import LmHeadModel
    from levanter.utils.jax_utils import use_cpu_device
    from levanter.utils.tree_utils import inference_mode

    from experiments.unified.unified_pretrain import UNIFIED_TOKENIZER_PATH

    tokenizer_path = args.tokenizer or UNIFIED_TOKENIZER_PATH
    tokenizer = load_tokenizer(tokenizer_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    trainer_config = TrainerConfig()
    trainer_config.initialize()

    compute_axis_mapping = trainer_config.compute_axis_mapping
    parameter_axis_mapping = trainer_config.parameter_axis_mapping

    EvalBatch = hax.Axis("batch", args.eval_batch_size)

    with trainer_config.use_device_mesh(), hax.axis_mapping(parameter_axis_mapping):
        from levanter.utils.hf_utils import round_axis_for_partitioning

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

            model_config = ModelConfig()
            with use_cpu_device():
                model = eqx.filter_eval_shape(model_config.build, Vocab, key=key)
                model = load_checkpoint(model, args.checkpoint_path, subpath="model")
            model = hax.shard(model, parameter_axis_mapping)

        model = typing.cast(LmHeadModel, inference_mode(model, True))

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
        jit_fn = _make_vlm_eval_jit(EvalBatch, EvalPos, max_packed_segments, compute_axis_mapping, trainer_config.mp)

        results = evaluate_all_vlm_benchmarks(
            jit_fn, model, prepared, tokenizer, EvalPos, EvalBatch, max_packed_segments, compute_axis_mapping
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
        model = inference_mode(step.eval_model, True)
        EvalPos = model.Pos if max_length is None else model.Pos.resize(max_length)

        logger.info(
            "Running VLM eval at step %d (EvalPos=%d, EvalBatch=%d, params=%s)",
            step.step,
            EvalPos.size,
            EvalBatch.size,
            f"{parameter_count(model):,}",
        )

        # All processes prepare identical data independently, then run eval
        # in lockstep. hax.shard places each host's data onto its devices.
        prepared: dict[str, tuple] = {}
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

        max_packed_segments = 64
        jit_fn = _make_vlm_eval_jit(EvalBatch, EvalPos, max_packed_segments, axis_resources, mp)

        # Run ALL benchmarks in a single continuous batch loop to avoid the
        # TPU crash that occurs at benchmark transitions.
        try:
            results = evaluate_all_vlm_benchmarks(
                jit_fn, model, prepared, tokenizer, EvalPos, EvalBatch, max_packed_segments, axis_resources
            )
            if jax.process_index() == 0:
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
                        levanter.tracker.log(
                            {
                                f"eval/{bench_name}/accuracy": accuracy,
                                f"eval/{bench_name}/loss": loss,
                            },
                            step=step.step,
                        )
                    elif task_type == "open_ended":
                        mean_ll = result["mean_ll"]
                        logger.info(
                            "%s: mean_ll=%.4f (%d questions)",
                            bench_name,
                            mean_ll,
                            result["num_questions"],
                        )
                        levanter.tracker.log(
                            {f"eval/{bench_name}/mean_ll": mean_ll},
                            step=step.step,
                        )
        except Exception:
            logger.exception("vlm_mc_eval crashed")

        logger.info("VLM eval done.")

    return vlm_eval


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
