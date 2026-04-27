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
    from datasets import load_dataset

    data_files = _find_split_parquet_files(input_path, split, subset)
    dataset = load_dataset("parquet", data_files={split: data_files}, split=split, streaming=True)
    return dataset


def _render_mmlu_example(example: dict[str, Any]) -> str:
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
    lines.append(f"{string.ascii_uppercase[answer_index]}. {choices[answer_index]}")
    return "\n".join(lines)


def _render_gsm8k_example(example: dict[str, Any]) -> str:
    question = str(example.get("question") or "").strip()
    answer = str(example.get("answer") or "").strip()
    if not question or not answer:
        return ""
    return f"Question: {question}\n\nAnswer:\n{answer}"


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

    record_count = 0
    with atomic_rename(out_file) as temp_path:
        with open_url(temp_path, "wt", encoding="utf-8", compression=compression) as outfile:
            for index, example in enumerate(_load_hf_iterable(cfg.input_path, cfg.split, cfg.subset)):
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
                metadata={"renderer": cfg.renderer_name.value},
            ),
        )
        result["metadata_file"] = metadata_path

    return result
