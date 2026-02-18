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
PromptCompletion → greedy_pack → _LmEvalHarnessWorker.dispatch_loglikelihood.

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
import numpy as np
import pyarrow.parquet as pq
from tqdm_loggable.auto import tqdm

import haliax as hax

from levanter.data.loader import stack_batches
from levanter.data.packing import PromptCompletion, greedy_pack_prompt_completions
from levanter.eval_harness import (
    _LmEvalHarnessWorker,
    get_padding_count_from_batch,
)
from levanter.utils.background_iterable import BackgroundIterator

from experiments.unified.vlm_tokenize_captions import (
    VISION_END_ID,
    VISION_START_ID,
    VISUAL_TOKEN_OFFSET,
)
from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)

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


def evaluate_vlm_mc_benchmark(
    worker: _LmEvalHarnessWorker,
    parquet_paths: list[str],
    tokenizer,
) -> dict:
    """Evaluate an MC benchmark using per-option log-likelihood comparison.

    Reuses the same loglikelihood pipeline as text LLM evaluation (MMLU etc.):
    PromptCompletion → greedy_pack → dispatch_loglikelihood → group by question → accuracy.

    Args:
        worker: Levanter eval harness worker (holds model + JIT loglikelihood).
        parquet_paths: Paths to benchmark parquet shards (with image_tokens column).
        tokenizer: Tokenizer for encoding text parts.

    Returns:
        Dict with accuracy, loss (mean normalized LL of correct answers),
        num_questions, num_correct, and per-question details.
    """
    # Step 1: Build all PromptCompletion objects from parquet rows
    all_completions: list[PromptCompletion] = []
    # segment_id → (question_idx, option_idx, is_correct, num_completion_tokens)
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

            # Check none exceed max length
            max_len = worker.EvalPos.size
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

    num_questions = question_idx
    if num_questions == 0:
        logger.warning("No MC questions found in parquets")
        return {"accuracy": 0.0, "num_questions": 0, "num_correct": 0}

    logger.info(
        "Built %d completions for %d questions (%d skipped)",
        len(all_completions),
        num_questions,
        skipped,
    )

    # Step 2: Pack and run loglikelihood (same pipeline as text MMLU)
    pad_token = tokenizer.pad_token_id
    if pad_token is None:
        pad_token = tokenizer.eos_token_id

    packed = greedy_pack_prompt_completions(
        worker.EvalPos,
        all_completions,
        pad_token=pad_token,
        max_segments_per_example=worker.max_packed_segments,
    )

    packed_iterator = stack_batches(iter(packed), worker.EvalPos, worker.EvalBatch)
    packed_iterator = BackgroundIterator(packed_iterator, max_capacity=64)

    result_probs = np.zeros(len(all_completions))
    covered_points = np.zeros(len(all_completions), dtype=bool)

    total_tokens_expected = len(packed) * worker.EvalPos.size
    pbar = tqdm(total=total_tokens_expected, desc="vlm_mc_eval", unit="tok")

    for batch in packed_iterator:
        batch = hax.shard(batch, worker.axis_resources)

        _padding_count, batch_tokens = get_padding_count_from_batch(batch, pad_token)
        batch = jax.device_put(batch)

        out_ids, out_lls, _out_correct = worker.dispatch_loglikelihood(batch)

        out_ids = jax.device_get(out_ids.array)
        out_lls = jax.device_get(out_lls.array)

        valid_indices = out_ids != -1
        result_probs[out_ids[valid_indices]] = out_lls[valid_indices]
        covered_points[out_ids[valid_indices]] = True

        pbar.update(batch_tokens)

    pbar.close()

    missing_points = np.where(~covered_points)[0]
    if len(missing_points) > 0:
        logger.warning("Missing %d segments: %s", len(missing_points), missing_points[:10])

    # Step 3: Group by question, pick highest log-likelihood → accuracy
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

        # Pick option with highest log-likelihood
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

    # Step 4: Compute mean length-normalized LL of correct answers (loss metric)
    # per_segment_loss() returns summed token-level losses, so we normalize by
    # the number of completion tokens for comparability across questions.
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


# ---------------------------------------------------------------------------
# Open-ended evaluation
# ---------------------------------------------------------------------------


def evaluate_vlm_open_ended_benchmark(
    worker: _LmEvalHarnessWorker,
    parquet_paths: list[str],
    tokenizer,
) -> dict:
    """Evaluate an open-ended benchmark using length-normalized log-likelihood.

    For each question, computes log P(gold_answer | prompt + image) and normalizes
    by the number of answer tokens. Reports mean normalized log-likelihood across
    all questions.

    Args:
        worker: Levanter eval harness worker (holds model + JIT loglikelihood).
        parquet_paths: Paths to benchmark parquet shards (with image_tokens column).
        tokenizer: Tokenizer for encoding text parts.

    Returns:
        Dict with mean_ll, num_questions, and per-question details.
    """
    # Step 1: Build all PromptCompletion objects (1 per question)
    all_completions: list[PromptCompletion] = []
    # segment_id → (question_idx, answer_str, num_answer_tokens)
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

            # Check length limit
            max_len = worker.EvalPos.size
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

    num_questions = question_idx
    if num_questions == 0:
        logger.warning("No open-ended questions found in parquets")
        return {"mean_ll": 0.0, "num_questions": 0}

    logger.info(
        "Built %d completions for %d open-ended questions (%d skipped)",
        len(all_completions),
        num_questions,
        skipped,
    )

    # Step 2: Pack and run loglikelihood (same pipeline as MC)
    pad_token = tokenizer.pad_token_id
    if pad_token is None:
        pad_token = tokenizer.eos_token_id

    packed = greedy_pack_prompt_completions(
        worker.EvalPos,
        all_completions,
        pad_token=pad_token,
        max_segments_per_example=worker.max_packed_segments,
    )

    packed_iterator = stack_batches(iter(packed), worker.EvalPos, worker.EvalBatch)
    packed_iterator = BackgroundIterator(packed_iterator, max_capacity=64)

    result_probs = np.zeros(len(all_completions))
    covered_points = np.zeros(len(all_completions), dtype=bool)

    total_tokens_expected = len(packed) * worker.EvalPos.size
    pbar = tqdm(total=total_tokens_expected, desc="vlm_open_ended_eval", unit="tok")

    for batch in packed_iterator:
        batch = hax.shard(batch, worker.axis_resources)

        _padding_count, batch_tokens = get_padding_count_from_batch(batch, pad_token)
        batch = jax.device_put(batch)

        out_ids, out_lls, _out_correct = worker.dispatch_loglikelihood(batch)

        out_ids = jax.device_get(out_ids.array)
        out_lls = jax.device_get(out_lls.array)

        valid_indices = out_ids != -1
        result_probs[out_ids[valid_indices]] = out_lls[valid_indices]
        covered_points[out_ids[valid_indices]] = True

        pbar.update(batch_tokens)

    pbar.close()

    missing_points = np.where(~covered_points)[0]
    if len(missing_points) > 0:
        logger.warning("Missing %d segments: %s", len(missing_points), missing_points[:10])

    # Step 3: Compute length-normalized log-likelihood per question
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

        # Create worker (same as eval_harness.py line 1300)
        worker = _LmEvalHarnessWorker(
            EvalBatch,
            EvalPos,
            model,
            compute_axis_mapping,
            tokenizer,
            trainer_config.mp,
            max_packed_segments=64,
        )

        if jax.process_index() == 0:
            results = {}
            for bench_name in args.benchmarks:
                parquet_paths = find_benchmark_parquets(bench_name, args.eval_parquet_path)
                if not parquet_paths:
                    continue

                task_type = _detect_benchmark_task_type(parquet_paths[0])
                logger.info("Evaluating %s [%s] (%d parquet shards)", bench_name, task_type, len(parquet_paths))

                if task_type == "multiple_choice":
                    result = evaluate_vlm_mc_benchmark(worker, parquet_paths, tokenizer)
                    results[bench_name] = result
                    logger.info(
                        "%s: accuracy=%.4f (%d/%d), loss=%.4f",
                        bench_name,
                        result["accuracy"],
                        result["num_correct"],
                        result["num_questions"],
                        result["loss"],
                    )
                elif task_type == "open_ended":
                    result = evaluate_vlm_open_ended_benchmark(worker, parquet_paths, tokenizer)
                    results[bench_name] = result
                    logger.info(
                        "%s: mean_ll=%.4f (%d questions)",
                        bench_name,
                        result["mean_ll"],
                        result["num_questions"],
                    )
                else:
                    logger.warning("Unsupported task_type=%s for %s, skipping", task_type, bench_name)

            worker.stop()

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
                # Strip per_question details for compact output
                compact = {
                    bench: {k: v for k, v in res.items() if k != "per_question"} for bench, res in results.items()
                }
                with open(args.output_json, "w") as f:
                    json.dump(compact, f, indent=2)
                logger.info("Results written to %s", args.output_json)
        else:
            worker.worker_message_loop()

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
        if step.step == 0 and not force:
            return

        model = inference_mode(step.eval_model, True)
        EvalPos = model.Pos if max_length is None else model.Pos.resize(max_length)

        logger.info(
            "Running VLM eval at step %d (EvalPos=%d, EvalBatch=%d, params=%s)",
            step.step,
            EvalPos.size,
            EvalBatch.size,
            f"{parameter_count(model):,}",
        )

        worker = _LmEvalHarnessWorker(
            EvalBatch,
            EvalPos,
            model,
            axis_resources,
            tokenizer,
            mp,
            max_packed_segments=64,
        )

        if jax.process_index() == 0:
            for bench_name, parquet_paths in bench_parquets.items():
                task_type = bench_task_types[bench_name]
                logger.info("Evaluating %s [%s] (%d shards)", bench_name, task_type, len(parquet_paths))

                if task_type == "multiple_choice":
                    result = evaluate_vlm_mc_benchmark(worker, parquet_paths, tokenizer)
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
                    result = evaluate_vlm_open_ended_benchmark(worker, parquet_paths, tokenizer)
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

                else:
                    logger.warning("Unknown task_type=%s for %s, skipping", task_type, bench_name)

            worker.stop()
        else:
            worker.worker_message_loop()

        logger.info("VLM eval done.")

    return vlm_eval


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
