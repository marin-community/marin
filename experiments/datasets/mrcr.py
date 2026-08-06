# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenAI MRCR long-context perplexity datasets for tagged evaluation."""

import gzip
import io
import json
from contextlib import ExitStack
from dataclasses import dataclass
from typing import TextIO, TypedDict

import pyarrow.parquet as pq
from levanter.data.text.formats import ChatLmDatasetFormat
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.data import dataset_main, hf_download
from marin.processing.tokenize.tokenize import TokenizeConfig, TokenizedCache, tokenize
from rigging.filesystem import StoragePath, prefix_join

from experiments.llama import llama3_tokenizer

_HF_REVISION = "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d"
_VERSION = "2026.07.14.1"
MRCR_NEEDLE_COUNTS = (2, 4, 8)
_FULL_CONTEXT = "full_context"
_FINAL_USER_ONLY = "final_user_only"
MRCR_CONDITIONS = (_FULL_CONTEXT, _FINAL_USER_ONLY)
_MRCR_CHAT_TEMPLATE = (
    "{{ messages[0]['content'] }}" "{% generation %}{{ messages[1]['content'] + ' ' + eos_token }}{% endgeneration %}"
)


class _MrcrMessage(TypedDict):
    role: str
    content: str


@dataclass(frozen=True)
class MrcrTransformConfig:
    """Paths for converting MRCR parquet files to paired chat records."""

    input_path: str
    output_path: str


def _mrcr_format() -> ChatLmDatasetFormat:
    return ChatLmDatasetFormat(chat_template=_MRCR_CHAT_TEMPLATE, pack=True, slice_strategy="right")


class MrcrTokenizedCache(TokenizedCache):
    """MRCR cache consumed as packed, right-sliced tagged-eval examples."""

    @property
    def format(self) -> ChatLmDatasetFormat:
        return _mrcr_format()


def _render_prompt(messages: list[_MrcrMessage]) -> str:
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
        path = prefix_join(output_path, f"{needles}needle/{condition}.jsonl.gz")
        StoragePath(path).parent.mkdirs(exist_ok=True)
        raw = stack.enter_context(StoragePath(path).open("wb"))
        compressed = stack.enter_context(gzip.GzipFile(fileobj=raw, mode="wb", mtime=0))
        writers[key] = stack.enter_context(io.TextIOWrapper(compressed, encoding="utf-8"))
    return writers[key]


def transform_mrcr(config: MrcrTransformConfig) -> None:
    """Convert MRCR into full-context and final-user-only chat pairs."""

    input_files = sorted(str(path) for path in StoragePath(prefix_join(config.input_path, "**/*.parquet")).glob())
    if not input_files:
        raise FileNotFoundError(f"No MRCR parquet files found under {config.input_path}")

    with ExitStack() as stack:
        writers: dict[tuple[int, str], TextIO] = {}
        for input_file in input_files:
            with StoragePath(input_file).open("rb") as source:
                parquet = pq.ParquetFile(source)
                for batch in parquet.iter_batches(batch_size=1, columns=["prompt", "answer", "n_needles"]):
                    row = batch.to_pylist()[0]
                    messages: list[_MrcrMessage] = json.loads(row["prompt"])
                    answer = row["answer"]
                    needles = row["n_needles"]
                    prompts = {
                        _FULL_CONTEXT: _render_prompt(messages),
                        _FINAL_USER_ONLY: _render_prompt(messages[-1:]),
                    }

                    for condition, prompt in prompts.items():
                        output = _writer(
                            stack,
                            writers,
                            config.output_path,
                            needles,
                            condition,
                        )
                        output.write(
                            json.dumps(
                                {
                                    "messages": [
                                        {"role": "user", "content": prompt},
                                        {"role": "assistant", "content": answer},
                                    ]
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )


def _mrcr_tags(needles: int, condition: str) -> tuple[str, ...]:
    needle = f"{needles}needle"
    return (
        f"mrcr/{condition}",
        f"mrcr/{needle}/{condition}",
    )


def _tokenized_mrcr(
    *,
    name: str,
    tokenizer: str,
    raw: ArtifactStep[Artifact],
    glob: str,
    tags: tuple[str, ...],
) -> ArtifactStep[TokenizedCache]:
    def build_config(ctx: StepContext) -> TokenizeConfig:
        return TokenizeConfig(
            train_paths=[],
            validation_paths=[prefix_join(ctx.artifact_path(raw), glob)],
            cache_path=ctx.output_path,
            tokenizer=tokenizer,
            format=_mrcr_format(),
            tags=list(tags),
        )

    return ArtifactStep(
        name=name,
        version=_VERSION,
        artifact_type=MrcrTokenizedCache,
        run=tokenize,
        build_config=build_config,
        deps=(raw,),
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
    return {
        f"{needles}needle/{condition}": _tokenized_mrcr(
            name=f"mrcr/{needles}needle/{condition}-llama3",
            tokenizer=tokenizer,
            raw=processed,
            glob=f"{needles}needle/{condition}.jsonl.gz",
            tags=_mrcr_tags(needles, condition),
        )
        for needles in MRCR_NEEDLE_COUNTS
        for condition in MRCR_CONDITIONS
    }


if __name__ == "__main__":
    dataset_main(mrcr_datasets())
