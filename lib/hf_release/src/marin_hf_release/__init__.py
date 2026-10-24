# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Contracts for portable Hugging Face inference releases."""

from marin_hf_release.policy import (
    POLICY_VERSION,
    PrecisionMode,
    ReleasePolicy,
    TensorSpec,
    canonical_grug_config,
    grug_tensor_specs,
)
from marin_hf_release.request import (
    HF_COMMIT_HEX_LENGTH,
    ReleaseRequest,
    TokenizerMetadata,
    completion_manifest,
    exporter_revision,
    generation_metadata,
    is_hf_commit,
    persist_request,
    resolve_metadata,
    tokenizer_special_token_ids,
    validate_release_metadata,
    write_completion_manifest,
)
from marin_hf_release.validation import validate_release_weights
