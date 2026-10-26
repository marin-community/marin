# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The four SFT sets of the midtraining probe, as prepared-messages artifacts and chat caches.

Liu, Neubig and Xiong (2025, arXiv:2510.14865) fine-tune on Pycode (a 5,000-function CodeSearchNet Python subset,
no prompt masking), GSM8K (prompt masked), LIMA (no masking) and SciQ (prompt masked). LIMA is gated on the Hub, so
the general-instruction set here is a 1,000-example sample of no_robots, the closest open human-written set.

Each prepared artifact holds ``train/data.jsonl.gz`` and ``validation/data.jsonl.gz`` with canonical OpenAI
``messages`` plus ``manifest.json`` (counts, source revision, format). A masked set is a user prompt followed by an
assistant completion; an unmasked set puts the whole example in one assistant message, so the base-model template
below supervises every token of it. The validation split is fixed per set and identical for every checkpoint.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
from dataclasses import dataclass
from typing import Any

import fsspec
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from levanter.data.text.formats import ChatLmDatasetFormat
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.processing.tokenize.tokenize import TokenizeConfig, TokenizedCache
from marin.processing.tokenize.tokenize import tokenize as run_tokenize
from rigging.filesystem.storage_path import prefix_join

from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

DATA_VERSION = "2026.10.14.1"  # .1: candidate-pool sampling before the length filter
CACHE_VERSION = "2026.10.14"
TOKENIZER = llama3_tokenizer
# Examples longer than this (tokens, including bos and eos) are dropped at preparation time: the unpacked chat
# dataset raises on an over-long example instead of truncating it.
MAX_TOKENS = 2048
SPLIT_SEED = 20_261_014

# Base-model template: prompt text followed by the completion; only the generation span (completion + eos) is
# supervised. An unmasked set is a single assistant message, so the whole example is in the span.
BASE_CHAT_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "{% if message['role'] == 'user' %}{{ message['content'] }}"
    "{% elif message['role'] == 'assistant' %}{% generation %}{{ message['content'] }}{{ eos_token }}{% endgeneration %}"
    "{% endif %}{% endfor %}"
)
CHAT_FORMAT = ChatLmDatasetFormat(
    messages_field="messages", chat_template=BASE_CHAT_TEMPLATE, mask_user_turns=True, pack=False
)


@dataclass(frozen=True)
class ProbeDataset:
    """One SFT set: its Hub source, prompt masking, and fixed train/validation sizes (``None`` keeps a whole split)."""

    slug: str
    hf_dataset_id: str
    revision: str
    mask_prompt: bool
    train_size: int | None
    validation_size: int | None
    description: str


GSM8K = ProbeDataset("gsm8k", "openai/gsm8k", "740312a", True, None, None, "GSM8K main train; validation = test")
SCIQ = ProbeDataset(
    "sciq", "allenai/sciq", "2c94ad3", True, None, None, "SciQ train with support; validation = validation"
)
PYCODE = ProbeDataset(
    "pycode", "code-search-net/code_search_net", "bd0cf26", False, 5000, 500, "CodeSearchNet Python functions"
)
NOROBOTS = ProbeDataset(
    "norobots", "HuggingFaceH4/no_robots", "e6f9a4a", False, 1000, 300, "no_robots conversations (LIMA substitute)"
)
PROBE_DATASETS = (GSM8K, SCIQ, PYCODE, NOROBOTS)


@dataclass(frozen=True)
class PrepareDatasetConfig:
    output_path: str
    dataset: ProbeDataset
    tokenizer: str
    max_tokens: int
    split_seed: int


def _parquet(dataset: ProbeDataset, filename: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = hf_hub_download(dataset.hf_dataset_id, filename, repo_type="dataset", revision=dataset.revision)
    return pd.read_parquet(path, columns=columns)


def _gsm8k_rows(split: str) -> list[dict[str, Any]]:
    frame = _parquet(GSM8K, f"main/{split}-00000-of-00001.parquet")
    return [
        {
            "messages": [
                {"role": "user", "content": f"Question: {q.strip()}\nAnswer:"},
                {"role": "assistant", "content": f" {a.strip()}"},
            ]
        }
        for q, a in zip(frame["question"], frame["answer"], strict=True)
    ]


def _sciq_rows(split: str) -> list[dict[str, Any]]:
    frame = _parquet(SCIQ, f"data/{split}-00000-of-00001.parquet")
    rows = []
    for row in frame.to_dict(orient="records"):
        support = str(row["support"]).strip()
        prompt = (f"{support}\n" if support else "") + f"Question: {str(row['question']).strip()}\nAnswer:"
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": f" {str(row['correct_answer']).strip()}"},
                ]
            }
        )
    return rows


def _pycode_rows(split: str) -> list[dict[str, Any]]:
    frame = _parquet(PYCODE, f"python/{split}-00000-of-00001.parquet", columns=["whole_func_string"])
    return [
        {"messages": [{"role": "assistant", "content": str(code).rstrip() + "\n"}]}
        for code in frame["whole_func_string"]
    ]


def _norobots_rows(split: str) -> list[dict[str, Any]]:
    frame = _parquet(NOROBOTS, f"data/{split}_sft-00000-of-00001.parquet")
    rows = []
    for messages in frame["messages"]:
        turns = []
        for message in messages:
            speaker = {"user": "User", "assistant": "Assistant", "system": "System"}[message["role"]]
            turns.append(f"{speaker}: {str(message['content']).strip()}")
        rows.append({"messages": [{"role": "assistant", "content": "\n\n".join(turns) + "\n"}]})
    return rows


_SOURCE_ROWS = {
    "gsm8k": (lambda: _gsm8k_rows("train"), lambda: _gsm8k_rows("test")),
    "sciq": (lambda: _sciq_rows("train"), lambda: _sciq_rows("validation")),
    "pycode": (lambda: _pycode_rows("train"), lambda: _pycode_rows("validation")),
    "norobots": (lambda: _norobots_rows("train"), lambda: _norobots_rows("test")),
}


_TOKENIZE_CHUNK = 2000
# A sampled set measures lengths on a seeded candidate pool of this many times its size, not on the whole split
# (tokenizing all 412k CodeSearchNet functions at once OOM-killed the 8 GB coordinator on 2026-10-14).
_POOL_FACTOR = 4


def _token_lengths(rows: list[dict[str, Any]], tokenizer: str) -> np.ndarray:
    from levanter.tokenizers import load_tokenizer  # noqa: PLC0415  # the coordinator loads it once per set

    tok = load_tokenizer(tokenizer)
    lengths: list[int] = []
    for start in range(0, len(rows), _TOKENIZE_CHUNK):
        chunk = [row["messages"] for row in rows[start : start + _TOKENIZE_CHUNK]]
        encoded = tok.apply_chat_template_with_masks(chunk, chat_template=BASE_CHAT_TEMPLATE)
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return np.asarray(lengths)


def _select(
    rows: list[dict[str, Any]], size: int | None, rng: np.random.Generator, max_tokens: int, tokenizer: str
) -> tuple[list[dict[str, Any]], int]:
    if size is not None and len(rows) > _POOL_FACTOR * size:
        pool = np.sort(rng.choice(len(rows), size=_POOL_FACTOR * size, replace=False))
        rows = [rows[i] for i in pool]
    lengths = _token_lengths(rows, tokenizer)
    keep = np.flatnonzero(lengths <= max_tokens)
    dropped = len(rows) - len(keep)
    if size is not None:
        if len(keep) < size:
            raise ValueError(f"Only {len(keep)} examples fit in {max_tokens} tokens; {size} requested")
        keep = np.sort(rng.choice(keep, size=size, replace=False))
    return [rows[i] for i in keep], dropped


def _write_jsonl_gz(path: str, rows: list[dict[str, Any]]) -> str:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as gz:
        for row in rows:
            gz.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
    payload = buffer.getvalue()
    with fsspec.open(path, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def prepare_dataset(config: PrepareDatasetConfig) -> Artifact:
    """Download one set, format its messages, apply the length filter and fixed sampling, and write both splits."""
    dataset = config.dataset
    train_source, validation_source = _SOURCE_ROWS[dataset.slug]
    rng = np.random.default_rng(config.split_seed)
    train_rows, train_dropped = _select(train_source(), dataset.train_size, rng, config.max_tokens, config.tokenizer)
    validation_rows, validation_dropped = _select(
        validation_source(), dataset.validation_size, rng, config.max_tokens, config.tokenizer
    )
    train_sha = _write_jsonl_gz(prefix_join(config.output_path, "train/data.jsonl.gz"), train_rows)
    validation_sha = _write_jsonl_gz(prefix_join(config.output_path, "validation/data.jsonl.gz"), validation_rows)
    manifest = {
        "slug": dataset.slug,
        "hf_dataset_id": dataset.hf_dataset_id,
        "revision": dataset.revision,
        "description": dataset.description,
        "mask_prompt": dataset.mask_prompt,
        "tokenizer": config.tokenizer,
        "max_tokens": config.max_tokens,
        "split_seed": config.split_seed,
        "n_train": len(train_rows),
        "n_validation": len(validation_rows),
        "n_dropped_over_length": {"train": train_dropped, "validation": validation_dropped},
        "candidate_pool_factor": _POOL_FACTOR,
        "train_sha256": train_sha,
        "validation_sha256": validation_sha,
    }
    with fsspec.open(prefix_join(config.output_path, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    logger.info("Prepared %s: %d train / %d validation examples", dataset.slug, len(train_rows), len(validation_rows))
    return Artifact(path=config.output_path)


def prepare_dataset_step(dataset: ProbeDataset) -> ArtifactStep[Artifact]:
    return ArtifactStep(
        name=f"sft_midtraining_probe/data/{dataset.slug}",
        version=DATA_VERSION,
        artifact_type=Artifact,
        run=prepare_dataset,
        build_config=lambda ctx: PrepareDatasetConfig(
            output_path=ctx.output_path,
            dataset=dataset,
            tokenizer=TOKENIZER,
            max_tokens=MAX_TOKENS,
            split_seed=SPLIT_SEED,
        ),
    )


def chat_cache_step(dataset: ProbeDataset, prepared: ArtifactStep[Artifact]) -> ArtifactStep[TokenizedCache]:
    """Tokenize both splits with the base-model template and completion masking, off the training pod."""
    key = hashlib.md5(f"{TOKENIZER}|{BASE_CHAT_TEMPLATE}|unpacked".encode()).hexdigest()[:6]

    def build_config(ctx: StepContext) -> TokenizeConfig:
        root = ctx.artifact_path(prepared)
        return TokenizeConfig(
            train_paths=[prefix_join(root, "train/*.jsonl.gz")],
            validation_paths=[prefix_join(root, "validation/*.jsonl.gz")],
            cache_path=ctx.output_path,
            tokenizer=TOKENIZER,
            format=CHAT_FORMAT,
            tags=[dataset.slug],
        )

    return ArtifactStep(
        name=f"sft_midtraining_probe/tokenized/{dataset.slug}-chat-{key}",
        version=CACHE_VERSION,
        artifact_type=TokenizedCache,
        run=run_tokenize,
        build_config=build_config,
        deps=(prepared,),
    )


def read_manifest(prepared_path: str) -> dict[str, Any]:
    with fsspec.open(prefix_join(prepared_path, "manifest.json"), "r") as handle:
        return json.load(handle)
