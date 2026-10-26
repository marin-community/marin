# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generated evaluation sampling fields shared by recipes and callbacks."""

from __future__ import annotations

from pydantic import NonNegativeInt

from .model import Section, field, unset_field


class EvalSamplingParams(Section):
    max_generate_length: int = unset_field()
    repetition_penalty: int | float = field(1.0)
    temperature: int | float = field(0.0)
    top_p: int | float = field(1.0)
    min_p: int | float = field(0.0)
    top_k: int = field(-1)
    logprobs: NonNegativeInt | None = field(None)
    stop: tuple[str, ...] | None = field(None)
    min_tokens: NonNegativeInt = unset_field()
