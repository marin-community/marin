# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Interfaces shared by local inference backends."""

from dataclasses import dataclass
from typing import Protocol

OPENAI_API_SUFFIX = "/v1"
CONCAT_CHAT_TEMPLATE = "{%- for message in messages -%}{{ message['content'] }}\n\n{%- endfor -%}"


@dataclass(frozen=True)
class ModelSpec:
    """Resolved model inputs required by every local backend.

    ``revision`` belongs to ``weights``; ``tokenizer_revision`` belongs to
    ``tokenizer_source`` and stays pinned when Iris mirrors the weights.
    """

    weights: str
    api_model: str
    num_chips: int | None
    tensor_parallel_size: int | None
    dtype: str
    max_model_len: int | None
    chat_template_content: str | None
    tokenizer: str | None = None
    revision: str | None = None
    tokenizer_revision: str | None = None

    @property
    def tokenizer_source(self) -> str:
        return self.tokenizer or self.weights


class ServedModel(Protocol):
    """A running OpenAI-compatible server on the current host."""

    @property
    def base_url(self) -> str: ...

    @property
    def model_id(self) -> str: ...

    @property
    def chat_template_content(self) -> str | None: ...

    def check_alive(self) -> None: ...
