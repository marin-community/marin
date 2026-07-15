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
from levanter.data.text.formats import SupervisedLmDatasetFormat
from levanter.eval import EvalLossContrast
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.experiment.data import dataset_main, hf_download, tokenized
from marin.processing.tokenize.tokenize import TokenizedCache
from rigging.filesystem import StoragePath

from experiments.llama import llama3_tokenizer

_HF_REVISION = "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d"
_VERSION = "2026.07.14.1"
MRCR_NEEDLE_COUNTS = (2, 4, 8)
MRCR_CONDITIONS = ("full_context", "final_user_only")


@dataclass(frozen=True)
class MrcrTransformConfig:
    """Paths for converting MRCR parquet files to paired supervised records."""

    input_path: str
    output_path: str


def _render_prompt(messages: list[dict[str, str]]) -> str:
    turns = "".join(f"{message['role'].capitalize()}: {message['content']}\n" for message in messages)
    return f"{turns}Assistant: "


def _writer(
    stack: ExitStack,
    writers: dict[tuple[int, str], TextIO],
    output_path: str,
    needles: int,
    condition: str,
) -> TextIO:
    key = (needles, condition)
    if key not in writers:
        path = posixpath.join(
            output_path,
            f"{needles}needle",
            f"{condition}.jsonl.gz",
        )
        StoragePath(posixpath.dirname(path)).mkdirs(exist_ok=True)
        raw = stack.enter_context(StoragePath(path).open("wb"))
        compressed = stack.enter_context(gzip.GzipFile(fileobj=raw, mode="wb", mtime=0))
        writers[key] = stack.enter_context(io.TextIOWrapper(compressed, encoding="utf-8"))
    return writers[key]


def transform_mrcr(config: MrcrTransformConfig) -> None:
    """Convert MRCR into full-context and final-user-only target pairs."""

    input_files = sorted(str(path) for path in StoragePath(f"{config.input_path}/**/*.parquet").glob())
    if not input_files:
        raise FileNotFoundError(f"No MRCR parquet files found under {config.input_path}")

    with ExitStack() as stack:
        writers: dict[tuple[int, str], TextIO] = {}
        for input_file in input_files:
            with StoragePath(input_file).open("rb") as source:
                parquet = pq.ParquetFile(source)
                for batch in parquet.iter_batches(batch_size=1, columns=["prompt", "answer", "n_needles"]):
                    row = batch.to_pylist()[0]
                    messages = json.loads(row["prompt"])
                    answer = row["answer"]
                    needles = row["n_needles"]
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
                            condition,
                        )
                        output.write(json.dumps({"input": prompt, "target": answer}, ensure_ascii=False) + "\n")


def _mrcr_tags(needles: int, condition: str) -> tuple[str, ...]:
    needle = f"{needles}needle"
    return (
        f"mrcr/{condition}",
        f"mrcr/{needle}/{condition}",
    )


def mrcr_loss_contrasts() -> tuple[EvalLossContrast, ...]:
    """Return context-gain contrasts for the MRCR aggregate and needle counts."""

    groups = ["mrcr"]
    groups.extend(f"mrcr/{needles}needle" for needles in MRCR_NEEDLE_COUNTS)
    return tuple(
        EvalLossContrast(
            name=group,
            baseline_tag=f"{group}/final_user_only",
            conditioned_tag=f"{group}/full_context",
        )
        for group in groups
    )


def mrcr_datasets(*, tokenizer: str = llama3_tokenizer) -> dict[str, ArtifactStep[TokenizedCache]]:
    """Return paired MRCR validation datasets keyed by needle count and condition."""

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
        f"{needles}needle/{condition}": tokenized(
            f"mrcr/{needles}needle/{condition}-llama3",
            tokenizer=tokenizer,
            version=_VERSION,
            raw=processed,
            glob=f"{needles}needle/{condition}.jsonl.gz",
            validation=True,
            dataset_format=dataset_format,
            tags=_mrcr_tags(needles, condition),
        )
        for needles in MRCR_NEEDLE_COUNTS
        for condition in MRCR_CONDITIONS
    }


if __name__ == "__main__":
    dataset_main(mrcr_datasets())
