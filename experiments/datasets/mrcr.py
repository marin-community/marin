# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenAI MRCR long-context perplexity evals as paired validation datasets."""

import gzip
import io
import json
import posixpath
from contextlib import ExitStack
from dataclasses import dataclass
from typing import TextIO

import pyarrow.parquet as pq
import tiktoken
from levanter.data.text.formats import SupervisedLmDatasetFormat
from levanter.eval import EvalLossContrast
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.experiment.data import dataset_main, hf_download, tokenized
from marin.processing.tokenize.tokenize import TokenizedCache
from rigging.filesystem import StoragePath

from experiments.llama import llama3_tokenizer

_HF_REVISION = "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d"
_VERSION = "2026.07.14"
MRCR_NEEDLE_COUNTS = (2, 4, 8)
MRCR_CONTEXT_LENGTH_BINS = (
    ("4k-8k", 4_096, 8_192),
    ("8k-16k", 8_192, 16_384),
    ("16k-32k", 16_384, 32_768),
    ("32k-64k", 32_768, 65_536),
    ("64k-128k", 65_536, 131_072),
    ("128k-256k", 131_072, 262_144),
    ("256k-512k", 262_144, 524_288),
    ("512k-1m", 524_288, 1_048_576),
)
MRCR_CONDITIONS = ("full_context", "final_user_only")


@dataclass(frozen=True)
class MrcrTransformConfig:
    """Paths for converting MRCR parquet files to paired supervised records."""

    input_path: str
    output_path: str


def _context_length_bin(token_count: int) -> str:
    for name, lower, upper in MRCR_CONTEXT_LENGTH_BINS:
        if lower < token_count <= upper or (lower == 4_096 and token_count == lower):
            return name
    raise ValueError(f"MRCR example has {token_count} tokens, outside the benchmark bins")


def _render_prompt(messages: list[dict[str, str]]) -> str:
    turns = "".join(f"{message['role'].capitalize()}: {message['content']}\n" for message in messages)
    return f"{turns}Assistant: "


def _writer(
    stack: ExitStack,
    writers: dict[tuple[int, str, str], TextIO],
    output_path: str,
    needles: int,
    context_length_bin: str,
    condition: str,
) -> TextIO:
    key = (needles, context_length_bin, condition)
    if key not in writers:
        path = posixpath.join(
            output_path,
            f"{needles}needle",
            context_length_bin,
            f"{condition}.jsonl.gz",
        )
        StoragePath(posixpath.dirname(path)).mkdirs(exist_ok=True)
        raw = stack.enter_context(StoragePath(path).open("wb"))
        compressed = stack.enter_context(gzip.GzipFile(fileobj=raw, mode="wb", mtime=0))
        writers[key] = stack.enter_context(io.TextIOWrapper(compressed, encoding="utf-8"))
    return writers[key]


def transform_mrcr(config: MrcrTransformConfig) -> None:
    """Convert MRCR into full-context and final-user-only target pairs."""

    encoding = tiktoken.get_encoding("o200k_base")
    input_files = sorted(str(path) for path in StoragePath(f"{config.input_path}/**/*.parquet").glob())
    if not input_files:
        raise FileNotFoundError(f"No MRCR parquet files found under {config.input_path}")

    with ExitStack() as stack:
        writers: dict[tuple[int, str, str], TextIO] = {}
        for input_file in input_files:
            with StoragePath(input_file).open("rb") as source:
                parquet = pq.ParquetFile(source)
                for batch in parquet.iter_batches(batch_size=1, columns=["prompt", "answer", "n_needles"]):
                    row = batch.to_pylist()[0]
                    messages = json.loads(row["prompt"])
                    answer = row["answer"]
                    needles = row["n_needles"]
                    token_count = sum(len(encoding.encode(message["content"])) for message in messages)
                    token_count += len(encoding.encode(answer))
                    context_length_bin = _context_length_bin(token_count)
                    prompts = {
                        "full_context": _render_prompt(messages),
                        "final_user_only": _render_prompt(messages[-1:]),
                    }

                    for condition, prompt in prompts.items():
                        output = _writer(
                            stack,
                            writers,
                            config.output_path,
                            needles,
                            context_length_bin,
                            condition,
                        )
                        output.write(json.dumps({"input": prompt, "target": answer}, ensure_ascii=False) + "\n")


def _mrcr_tags(needles: int, context_length_bin: str, condition: str) -> tuple[str, ...]:
    needle = f"{needles}needle"
    return (
        f"mrcr/{condition}",
        f"mrcr/{needle}/{condition}",
        f"mrcr/{context_length_bin}/{condition}",
        f"mrcr/{needle}/{context_length_bin}/{condition}",
    )


def mrcr_loss_contrasts() -> tuple[EvalLossContrast, ...]:
    """Return context-gain contrasts for every MRCR aggregate and bin."""

    groups = ["mrcr"]
    groups.extend(f"mrcr/{needles}needle" for needles in MRCR_NEEDLE_COUNTS)
    groups.extend(f"mrcr/{context_length_bin}" for context_length_bin, _, _ in MRCR_CONTEXT_LENGTH_BINS)
    groups.extend(
        f"mrcr/{needles}needle/{context_length_bin}"
        for needles in MRCR_NEEDLE_COUNTS
        for context_length_bin, _, _ in MRCR_CONTEXT_LENGTH_BINS
    )
    return tuple(
        EvalLossContrast(
            name=group,
            baseline_tag=f"{group}/final_user_only",
            conditioned_tag=f"{group}/full_context",
        )
        for group in groups
    )


def mrcr_datasets(*, tokenizer: str = llama3_tokenizer) -> dict[str, ArtifactStep[TokenizedCache]]:
    """Return paired MRCR validation datasets keyed by needles, length bin, and condition."""

    raw = hf_download(
        "raw/openai/mrcr",
        hf_id="openai/mrcr",
        revision=_HF_REVISION,
        urls_glob=["*needle/*.parquet"],
        version=_VERSION,
    )
    processed = ArtifactStep(
        name="processed/openai/mrcr",
        version=_VERSION,
        artifact_type=Artifact,
        run=transform_mrcr,
        build_config=lambda ctx: MrcrTransformConfig(
            input_path=ctx.artifact_path(raw),
            output_path=ctx.output_path,
        ),
        deps=(raw,),
    )
    dataset_format = SupervisedLmDatasetFormat(pack=True, slice_strategy="right")
    return {
        f"{needles}needle/{context_length_bin}/{condition}": tokenized(
            f"mrcr/{needles}needle/{context_length_bin}/{condition}-llama3",
            tokenizer=tokenizer,
            version=_VERSION,
            raw=processed,
            glob=f"{needles}needle/{context_length_bin}/{condition}.jsonl.gz",
            validation=True,
            dataset_format=dataset_format,
            tags=_mrcr_tags(needles, context_length_bin, condition),
        )
        for needles in MRCR_NEEDLE_COUNTS
        for context_length_bin, _, _ in MRCR_CONTEXT_LENGTH_BINS
        for condition in MRCR_CONDITIONS
    }


if __name__ == "__main__":
    dataset_main(mrcr_datasets())
