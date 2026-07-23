# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Interfaces shared by local inference backends."""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol

OPENAI_API_SUFFIX = "/v1"
CONCAT_CHAT_TEMPLATE = "{%- for message in messages -%}{{ message['content'] }}\n\n{%- endfor -%}"


@dataclass(frozen=True)
class ModelSpec:
    """Resolved model inputs required by every local backend."""

    model: str
    model_path: str
    num_chips: int | None
    tensor_parallel_size: int | None
    dtype: str
    max_model_len: int | None
    chat_template_content: str | None


class ServedModel(Protocol):
    """A running OpenAI-compatible server on the current host."""

    @property
    def base_url(self) -> str: ...

    @property
    def model_id(self) -> str: ...

    def check_alive(self) -> None: ...


class InferenceBackend(Protocol):
    """Starts one OpenAI-compatible server on the current host."""

    @property
    def name(self) -> str: ...

    def serve(self, spec: ModelSpec) -> AbstractContextManager[ServedModel]: ...
