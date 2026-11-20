# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned retrieval state for the licensed OpenResearcher data and pinned BM25 index."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from taskcompendium.models import ProviderRequirement

OPENRESEARCHER_ACTION_INTERFACE = "openresearcher-bm25:v1"


class OpenResearcherState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    implementation_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    corpus_revision: Literal["8367703babe6e080899a48782f365c284c32b668"]
    index_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    top_k: Literal[5]
    snippet_max_tokens: Literal[512]
    include_get_document: Literal[True]


def openresearcher_state(requirement: ProviderRequirement) -> OpenResearcherState:
    if requirement.action_interface != OPENRESEARCHER_ACTION_INTERFACE:
        raise ValueError(f"Unsupported OpenResearcher interface: {requirement.action_interface}")
    return OpenResearcherState.model_validate(requirement.initial_state)
