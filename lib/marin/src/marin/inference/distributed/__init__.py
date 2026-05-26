# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Distributed inference: multi-region vLLM-on-TPU library.

Public surface:

* `inference(model, dataset, config) -> InferenceResult` — stateless API
* `ModelSpec`, `SamplingParams`, `InferenceConfig` — caller configuration
* `InferenceResult`, `ResponseRecord` — result handle and per-record output

See ``api.py`` for the top-level entry point. Designed to be called from
ExecutorStep pipelines or directly from scripts.
"""
from .api import InferenceInput, inference
from .config import (
    DEFAULT_HEARTBEAT_TIMEOUT,
    DEFAULT_TPU_SHAPES,
    InferenceConfig,
    ModelSpec,
    SamplingParams,
)
from .input import (
    PAYLOAD_KIND_MESSAGES,
    PAYLOAD_KIND_TEXT,
    PromptRecord,
)
from .output import InferenceResult, ResponseRecord
