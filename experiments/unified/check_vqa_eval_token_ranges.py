#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check VQA eval parquet rows for token-id range violations.

This script validates the exact tokenization path used by
`experiments/unified/vlm_mc_eval.py`:
1. Build user prompt tokens by interleaving image tokens and text tokens.
2. Build completions (multiple-choice or open-ended).
3. Verify every constructed token id is within tokenizer vocab range.

It also validates raw TokLIP code ranges before applying VISUAL_TOKEN_OFFSET.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from dataclasses import dataclass

import pyarrow.parquet as pq

from experiments.unified.unified_pretrain import TOKLIP_CODEBOOK_SIZE, UNIFIED_TOKENIZER_PATH
from experiments.unified.vlm_tokenize_captions import VISUAL_TOKEN_OFFSET, gcs_download
from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)

MC_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]
DEFAULT_BENCHMARKS = ["ai2d", "mmmu", "textvqa", "chartqa"]
DEFAULT_INPUT_PATH = "gs://marin-vlm/eval_benchmarks_tokenized"


@dataclass
class Violation:
    benchmark: str
    shard: str
    row: int
    task_type: str
    reason: str
    detail: str


def _load_table(parquet_path: str):
    if parquet_path.startswith("gs://"):
        tmp = tempfile.mkdtemp(prefix="check_vqa_eval_")
        local = os.path.join(tmp, "shard.parquet")
        gcs_download(parquet_path, local)
        return pq.read_table(local)
    return pq.read_table(parquet_path)


def _build_user_token_ids_with_tokenizer(
    messages: list[dict],
    image_token_lists: list[list[int]],
    tokenizer,
) -> tuple[list[int], list[Violation]]:
    user_content = []
    for msg in messages:
        if msg["role"] == "user":
            user_content = msg["content"]
            break

    user_ids: list[int] = []
    violations: list[Violation] = []
    image_idx = 0
    for part in user_content:
        if part.get("type") == "image":
            if image_idx < len(image_token_lists):
                raw_tokens = image_token_lists[image_idx]
                shifted = []
                for t in raw_tokens:
                    if not isinstance(t, int):
                        violations.append(
                            Violation("", "", -1, "", "non_int_image_token", f"type={type(t).__name__}, value={t!r}")
                        )
                        continue
                    if t < 0 or t >= TOKLIP_CODEBOOK_SIZE:
                        violations.append(
                            Violation(
                                "",
                                "",
                                -1,
                                "",
                                "raw_image_token_out_of_codebook_range",
                                f"token={t}, expected=[0,{TOKLIP_CODEBOOK_SIZE - 1}]",
                            )
                        )
                    shifted.append(t + VISUAL_TOKEN_OFFSET)

                user_ids.append(128_004)  # VISION_START_ID
                user_ids.extend(shifted)
                user_ids.append(128_005)  # VISION_END_ID
                image_idx += 1
        elif part.get("type") == "text" and part.get("text"):
            text_ids = tokenizer.encode(part["text"], add_special_tokens=False)
            user_ids.extend(text_ids)

    return user_ids, violations


def _resolve_choices(choices_obj) -> list[str]:
    if choices_obj is None:
        return []
    if isinstance(choices_obj, list):
        return [str(x) for x in choices_obj]
    return [str(x) for x in list(choices_obj)]


def _check_row(
    benchmark: str,
    shard: str,
    row_idx: int,
    task_type: str,
    messages,
    image_token_lists,
    choices,
    answer,
    tokenizer,
    vocab_size: int,
) -> list[Violation]:
    violations: list[Violation] = []
    prompt_ids, prompt_violations = _build_user_token_ids_with_tokenizer(messages, image_token_lists, tokenizer)
    for v in prompt_violations:
        v.benchmark = benchmark
        v.shard = shard
        v.row = row_idx
        v.task_type = task_type
    violations.extend(prompt_violations)

    if task_type == "multiple_choice":
        option_texts = _resolve_choices(choices)
        for opt_idx, choice in enumerate(option_texts):
            letter = MC_LETTERS[opt_idx] if opt_idx < len(MC_LETTERS) else chr(ord("A") + opt_idx)
            completion_text = f"{letter}. {choice}"
            option_ids = tokenizer.encode(completion_text, add_special_tokens=False)
            all_ids = prompt_ids + option_ids
            if all_ids:
                mx = max(all_ids)
                mn = min(all_ids)
                if mn < 0 or mx >= vocab_size:
                    violations.append(
                        Violation(
                            benchmark,
                            shard,
                            row_idx,
                            task_type,
                            "constructed_input_id_out_of_vocab",
                            f"min_id={mn}, max_id={mx}, vocab_size={vocab_size}, option_idx={opt_idx}",
                        )
                    )
    elif task_type == "open_ended":
        answer_text = "" if answer is None else str(answer)
        answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)
        all_ids = prompt_ids + answer_ids
        if all_ids:
            mx = max(all_ids)
            mn = min(all_ids)
            if mn < 0 or mx >= vocab_size:
                violations.append(
                    Violation(
                        benchmark,
                        shard,
                        row_idx,
                        task_type,
                        "constructed_input_id_out_of_vocab",
                        f"min_id={mn}, max_id={mx}, vocab_size={vocab_size}",
                    )
                )
    else:
        violations.append(
            Violation(
                benchmark,
                shard,
                row_idx,
                task_type,
                "unknown_task_type",
                f"task_type={task_type!r}",
            )
        )

    return violations


def check_benchmark(
    benchmark: str,
    input_path: str,
    tokenizer,
    max_rows_per_shard: int | None,
) -> tuple[int, list[Violation]]:
    pattern = f"{input_path.rstrip('/')}/{benchmark}/eval-{benchmark}-*.parquet"
    shard_paths = sorted(fsspec_glob(pattern))
    if not shard_paths:
        logger.warning("No shards found for %s (pattern=%s)", benchmark, pattern)
        return 0, []

    vocab_size = len(tokenizer)
    checked_rows = 0
    violations: list[Violation] = []

    for shard_path in shard_paths:
        table = _load_table(shard_path)
        n_rows = len(table)
        if max_rows_per_shard is not None:
            n_rows = min(n_rows, max_rows_per_shard)

        messages_col = table.column("messages")
        image_tokens_col = table.column("image_tokens")
        task_type_col = table.column("task_type")
        choices_col = table.column("choices") if "choices" in table.column_names else None
        answer_col = table.column("answer") if "answer" in table.column_names else None

        for i in range(n_rows):
            messages = messages_col[i].as_py()
            image_token_lists = image_tokens_col[i].as_py()
            task_type = task_type_col[i].as_py()
            choices = choices_col[i].as_py() if choices_col is not None else None
            answer = answer_col[i].as_py() if answer_col is not None else None

            row_violations = _check_row(
                benchmark=benchmark,
                shard=shard_path,
                row_idx=i,
                task_type=task_type,
                messages=messages,
                image_token_lists=image_token_lists,
                choices=choices,
                answer=answer,
                tokenizer=tokenizer,
                vocab_size=vocab_size,
            )
            violations.extend(row_violations)
            checked_rows += 1

    return checked_rows, violations


def main():
    parser = argparse.ArgumentParser(description="Check VQA eval token ranges against tokenizer vocab.")
    parser.add_argument("--input_path", type=str, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS)
    parser.add_argument("--tokenizer", type=str, default=UNIFIED_TOKENIZER_PATH)
    parser.add_argument("--max_rows_per_shard", type=int, default=None)
    parser.add_argument("--max_print", type=int, default=50)
    args = parser.parse_args()

    from levanter.compat.hf_checkpoints import load_tokenizer

    tokenizer = load_tokenizer(args.tokenizer)
    vocab_size = len(tokenizer)
    logger.info("Loaded tokenizer=%s vocab_size=%d", args.tokenizer, vocab_size)
    logger.info(
        "Checking benchmarks=%s input_path=%s visual_offset=%d codebook_size=%d",
        args.benchmarks,
        args.input_path,
        VISUAL_TOKEN_OFFSET,
        TOKLIP_CODEBOOK_SIZE,
    )

    total_rows = 0
    total_violations: list[Violation] = []
    for bench in args.benchmarks:
        rows, violations = check_benchmark(
            benchmark=bench,
            input_path=args.input_path,
            tokenizer=tokenizer,
            max_rows_per_shard=args.max_rows_per_shard,
        )
        total_rows += rows
        total_violations.extend(violations)
        logger.info("Benchmark %s: checked_rows=%d violations=%d", bench, rows, len(violations))

    print("=" * 80)
    print("VQA EVAL TOKEN RANGE CHECK SUMMARY")
    print(f"tokenizer: {args.tokenizer}")
    print(f"vocab_size: {vocab_size}")
    print(f"input_path: {args.input_path}")
    print(f"benchmarks: {args.benchmarks}")
    print(f"checked_rows: {total_rows}")
    print(f"violations: {len(total_violations)}")

    if total_violations:
        print("-" * 80)
        print("First violations:")
        for v in total_violations[: args.max_print]:
            print(
                f"[{v.reason}] bench={v.benchmark} row={v.row} task={v.task_type} " f"shard={v.shard} detail={v.detail}"
            )
        raise SystemExit(2)

    print("No out-of-range token ids detected.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
