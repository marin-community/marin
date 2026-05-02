# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage small LM-eval-style HF datasets into raw text for PPL probes."""

from __future__ import annotations

import json
import os
import posixpath
import string
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from datasets import load_dataset
from rigging.filesystem import open_url, url_to_fs
from zephyr.writers import atomic_rename

from marin.datakit.ingestion_manifest import (
    IngestionSourceManifest,
    MaterializedOutputMetadata,
    write_ingestion_metadata_json,
)
from marin.utils import fsspec_mkdirs


class LmEvalRawRenderer(StrEnum):
    MMLU = "mmlu_multiple_choice"
    GSM8K = "gsm8k_qa"


MMLU_DEFAULT_NUM_FEWSHOT = 5
MMLU_DEFAULT_FEWSHOT_SPLIT = "dev"
GSM8K_COT_DEFAULT_NUM_FEWSHOT = 8
GSM8K_COT_FEWSHOT_EXAMPLES: tuple[tuple[str, str], ...] = (
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, "
        "there will be 21 trees. How many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more were planted. "
        "So there must have been 21 - 15 = 6. The answer is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. "
        "After eating 35, they had 74 - 35 = 39. The answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
        "How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
        "So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
        "How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. "
        "5 + 4 = 9. The answer is 9.",
    ),
    (
        "There were nine computers in the server room. Five more computers were installed each day, "
        "from monday to thursday. How many computers are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more computers were added. "
        "So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. "
        "After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
        "So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    ),
)


@dataclass(frozen=True)
class LmEvalRawStagingConfig:
    """Configuration for staging small LM-eval dataset slices into JSONL."""

    input_path: str
    output_path: str
    source_label: str
    renderer_name: LmEvalRawRenderer
    split: str
    subset: str | None = None
    output_filename: str = "staged.jsonl.gz"
    max_examples: int | None = None
    num_fewshot: int = 0
    fewshot_split: str | None = None
    extra_metadata: dict[str, str] = field(default_factory=dict)
    source_manifest: IngestionSourceManifest | None = None
    content_fingerprint: str = ""


def _fsspec_url(fs: Any, path: str) -> str:
    protocol = fs.protocol
    if isinstance(protocol, (list, tuple)):
        protocol = protocol[0]
    if protocol in (None, "file"):
        return path
    if path.startswith(f"{protocol}://"):
        return path
    return f"{protocol}://{path}"


def _parquet_file_matches_split(path: str, split: str) -> bool:
    filename = os.path.basename(path)
    if not filename.endswith(".parquet"):
        return False
    return filename == f"{split}.parquet" or filename.startswith(f"{split}-")


def _find_split_parquet_files(input_path: str, split: str, subset: str | None) -> list[str]:
    fs, root = url_to_fs(input_path)
    roots: list[str] = []
    if subset and subset != "default":
        subset_root = posixpath.join(root, subset)
        if fs.exists(subset_root):
            roots.append(subset_root)
    if not roots:
        roots.append(root)

    matches: list[str] = []
    for candidate_root in roots:
        if fs.isfile(candidate_root):
            candidates = [candidate_root]
            selected = [path for path in candidates if path.endswith(".parquet")]
        else:
            candidates = list(fs.find(candidate_root, withdirs=False))
            selected = [path for path in candidates if _parquet_file_matches_split(path, split)]
        matches.extend(selected)

    if not matches:
        raise FileNotFoundError(f"No parquet files found for split {split!r} under {input_path}")

    return [_fsspec_url(fs, path) for path in sorted(set(matches))]


def _load_hf_iterable(input_path: str, split: str, subset: str | None) -> Iterable[dict[str, Any]]:
    data_files = _find_split_parquet_files(input_path, split, subset)
    dataset = load_dataset("parquet", data_files={split: data_files}, split=split, streaming=True)
    return dataset


def _render_mmlu_question_block(example: dict[str, Any], *, include_answer: bool) -> str:
    question = str(example.get("question") or "").strip()
    if not question:
        return ""
    choices = example.get("choices") or []
    answer_index = int(example["answer"])
    subject = str(example.get("subject") or "").strip()

    lines: list[str] = []
    if subject:
        lines.append(f"Subject: {subject}")
        lines.append("")
    lines.append(f"Question: {question}")
    lines.append("")
    lines.append("Choices:")
    for label, choice in zip(string.ascii_uppercase, choices, strict=False):
        lines.append(f"{label}. {choice}")
    lines.append("")
    lines.append("Answer:")
    if include_answer:
        lines.append(f"{string.ascii_uppercase[answer_index]}. {choices[answer_index]}")
    return "\n".join(lines)


def _mmlu_signature(example: dict[str, Any]) -> str:
    return json.dumps(
        {
            "question": example.get("question"),
            "subject": example.get("subject"),
            "choices": list(example.get("choices") or []),
            "answer": example.get("answer"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def _build_mmlu_fewshot_index(
    input_path: str,
    split: str,
    subset: str | None,
    *,
    num_fewshot: int,
) -> dict[str, list[tuple[str, str]]]:
    by_subject: dict[str, list[tuple[str, str]]] = {}
    if num_fewshot <= 0:
        return by_subject

    # Keep one extra example per subject so dev->dev rendering can usually exclude
    # the query example and still remain close to the target shot count.
    target_count = num_fewshot + 1
    for example in _load_hf_iterable(input_path, split, subset):
        subject = str(example.get("subject") or "").strip()
        if not subject:
            continue
        rendered = _render_mmlu_question_block(example, include_answer=True)
        if not rendered:
            continue
        supports = by_subject.setdefault(subject, [])
        if len(supports) < target_count:
            supports.append((_mmlu_signature(example), rendered))

    return by_subject


def _render_mmlu_example(
    example: dict[str, Any],
    *,
    fewshot_index: dict[str, list[tuple[str, str]]],
    num_fewshot: int,
) -> str:
    query = _render_mmlu_question_block(example, include_answer=True)
    if not query:
        return ""
    if num_fewshot <= 0:
        return query

    subject = str(example.get("subject") or "").strip()
    signature = _mmlu_signature(example)
    supports = [
        rendered for support_signature, rendered in fewshot_index.get(subject, []) if support_signature != signature
    ][:num_fewshot]
    if not supports:
        return query
    return "\n\n".join([*supports, query])


def _render_gsm8k_question_block(example: dict[str, Any], *, include_answer: bool) -> str:
    question = str(example.get("question") or "").strip()
    answer = str(example.get("answer") or "").strip()
    if not question or (include_answer and not answer):
        return ""
    if include_answer:
        return f"Q: {question}\nA: {answer}"
    return f"Q: {question}\nA:"


def _render_gsm8k_example(example: dict[str, Any], *, num_fewshot: int) -> str:
    query = _render_gsm8k_question_block(example, include_answer=True)
    if not query:
        return ""
    if num_fewshot <= 0:
        return query
    supports = [f"Q: {question}\nA: {answer}" for question, answer in GSM8K_COT_FEWSHOT_EXAMPLES[:num_fewshot]]
    return "\n\n".join([*supports, query])


RENDERERS = {
    LmEvalRawRenderer.MMLU: _render_mmlu_example,
    LmEvalRawRenderer.GSM8K: _render_gsm8k_example,
}


def stage_lm_eval_source(cfg: LmEvalRawStagingConfig) -> dict[str, int | str]:
    """Stage one LM-eval-style dataset split into raw-text JSONL."""
    if cfg.source_manifest is not None and cfg.content_fingerprint:
        expected = cfg.source_manifest.fingerprint()
        if cfg.content_fingerprint != expected:
            raise ValueError(
                f"content_fingerprint mismatch: config has {cfg.content_fingerprint}, source manifest has {expected}"
            )

    renderer = RENDERERS[cfg.renderer_name]
    fsspec_mkdirs(cfg.output_path, exist_ok=True)
    out_file = posixpath.join(cfg.output_path, cfg.output_filename)
    compression = "gzip" if out_file.endswith(".gz") else None

    mmlu_fewshot_index: dict[str, list[tuple[str, str]]] = {}
    if cfg.renderer_name is LmEvalRawRenderer.MMLU and cfg.num_fewshot > 0:
        mmlu_fewshot_index = _build_mmlu_fewshot_index(
            cfg.input_path,
            cfg.fewshot_split or cfg.split,
            cfg.subset,
            num_fewshot=cfg.num_fewshot,
        )

    record_count = 0
    with atomic_rename(out_file) as temp_path:
        with open_url(temp_path, "wt", encoding="utf-8", compression=compression) as outfile:
            for index, example in enumerate(_load_hf_iterable(cfg.input_path, cfg.split, cfg.subset)):
                if cfg.renderer_name is LmEvalRawRenderer.MMLU:
                    text = renderer(example, fewshot_index=mmlu_fewshot_index, num_fewshot=cfg.num_fewshot)
                elif cfg.renderer_name is LmEvalRawRenderer.GSM8K:
                    text = renderer(example, num_fewshot=cfg.num_fewshot)
                else:
                    text = renderer(example)
                if not text:
                    continue
                record = {
                    "id": f"{cfg.source_label}:{cfg.split}:{index:08d}",
                    "text": text,
                    "source": cfg.source_label,
                    "provenance": {
                        "dataset": cfg.input_path,
                        "split": cfg.split,
                        "subset": cfg.subset,
                        "renderer": cfg.renderer_name.value,
                        "num_fewshot": cfg.num_fewshot,
                        "fewshot_split": cfg.fewshot_split,
                        "index": index,
                        **cfg.extra_metadata,
                    },
                }
                json.dump(record, outfile, ensure_ascii=False)
                outfile.write("\n")
                record_count += 1
                if cfg.max_examples is not None and record_count >= cfg.max_examples:
                    break

    fs, _ = url_to_fs(out_file)
    output_size = int(fs.info(out_file)["size"])
    result: dict[str, int | str] = {
        "record_count": record_count,
        "bytes_written": output_size,
        "output_file": out_file,
    }

    if cfg.source_manifest is not None:
        metadata_path = write_ingestion_metadata_json(
            manifest=cfg.source_manifest,
            materialized_output=MaterializedOutputMetadata(
                input_path=cfg.input_path,
                output_path=cfg.output_path,
                output_file=out_file,
                record_count=record_count,
                bytes_written=output_size,
                metadata={
                    "renderer": cfg.renderer_name.value,
                    "num_fewshot": cfg.num_fewshot,
                    "fewshot_split": cfg.fewshot_split,
                },
            ),
        )
        result["metadata_file"] = metadata_path

    return result
