# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Structured-chat Datakit sources and lowering to ordinary text sources."""

from dataclasses import asdict, dataclass

from fray.types import ResourceConfig
from levanter.data.text.formats import ChatLmDatasetFormat
from levanter.tokenizers import MarinTokenizer, load_tokenizer
from rigging.filesystem.storage_path import prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.readers import load_parquet

from marin.datakit.download.rollout_transforms import text_document
from marin.datakit.normalize import NormalizedData, normalize_step
from marin.datakit.sources import DatakitSource
from marin.execution.artifact import read_artifact
from marin.execution.step_spec import StepSpec


@dataclass(frozen=True)
class DatakitChatSource:
    """A source whose terminal normalized artifact contains canonical messages."""

    name: str
    normalize_steps: tuple[StepSpec, ...]
    format: ChatLmDatasetFormat
    rough_token_count_b: float

    @property
    def normalized(self) -> StepSpec:
        return self.normalize_steps[-1]


def _render_messages(record: dict, tokenizer: MarinTokenizer, chat_format: ChatLmDatasetFormat) -> list[dict]:
    kwargs = record.get(chat_format.chat_template_kwargs) if chat_format.chat_template_kwargs is not None else None
    rendered = tokenizer.apply_chat_template(
        record[chat_format.messages_field],
        tokenize=False,
        add_generation_prompt=False,
        **(kwargs or {}),
    )
    assert isinstance(rendered, str)
    if tokenizer.bos_token and rendered.startswith(tokenizer.bos_token):
        rendered = rendered.removeprefix(tokenizer.bos_token)
    return [text_document(rendered, record.get("source", ""))]


def _render_chat_artifact(
    input_path: str,
    output_path: str,
    tokenizer_name: str,
    chat_format: ChatLmDatasetFormat,
) -> None:
    normalized = read_artifact(input_path, NormalizedData)
    tokenizer = load_tokenizer(tokenizer_name)
    if chat_format.chat_template is not None:
        tokenizer = tokenizer.with_chat_template(chat_format.chat_template)
    pipeline = (
        Dataset.from_files(prefix_join(normalized.main_output_dir.resolve(), "*.parquet"))
        .flat_map(load_parquet)
        .flat_map(lambda record: _render_messages(record, tokenizer, chat_format))
        .write_parquet(prefix_join(output_path, "data-{shard:05d}-of-{total:05d}.parquet"), skip_existing=True)
    )
    ZephyrContext(name="render-chat", resources=ResourceConfig(cpu=1, ram="8g")).execute(pipeline)


def render_chat_source(source: DatakitChatSource, *, tokenizer: str) -> DatakitSource:
    """Render a structured-chat source and return an ordinary text Datakit source."""
    rendered = StepSpec(
        name=f"rendered-chat/{source.name}",
        deps=[source.normalized],
        fn=lambda output_path: _render_chat_artifact(
            source.normalized.output_path, output_path, tokenizer, source.format
        ),
        hash_attrs={"tokenizer": tokenizer, "format": asdict(source.format)},
    )
    normalized = normalize_step(name=f"normalized-rendered-chat/{source.name}", download=rendered)
    return DatakitSource(
        name=source.name,
        normalize_steps=(*source.normalize_steps, rendered, normalized),
        rough_token_count_b=source.rough_token_count_b,
    )
