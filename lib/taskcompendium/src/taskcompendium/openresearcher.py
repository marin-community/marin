# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned retrieval state for the licensed OpenResearcher data and its retrieval indexes."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from taskcompendium.models import FunctionDefinition, ProviderRequirement

OPENRESEARCHER_ACTION_INTERFACE = "openresearcher-bm25:v1"
OPENRESEARCHER_MODULE = "openresearcher.tool"
OPENRESEARCHER_HYBRID_ACTION_INTERFACE = "openresearcher-hybrid:v1"


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


class OpenResearcherHybridState(BaseModel):
    """Frozen hybrid-retrieval state over the same corpus, BM25 plus a dense index."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    implementation_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    corpus_revision: Literal["8367703babe6e080899a48782f365c284c32b668"]
    bm25_index_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dense_index_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding_model: Literal["Qwen/Qwen3-Embedding-8B"]
    embedding_model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    embedding_dim: Literal[4096]
    max_embed_tokens: int = Field(ge=128, le=32768)
    fusion: Literal["rrf"]
    fusion_rrf_k: Literal[60]
    candidate_depth: Literal[25]
    top_k: Literal[5]
    snippet_max_tokens: Literal[512]
    include_get_document: Literal[True]


def openresearcher_hybrid_state(requirement: ProviderRequirement) -> OpenResearcherHybridState:
    if requirement.action_interface != OPENRESEARCHER_HYBRID_ACTION_INTERFACE:
        raise ValueError(f"Unsupported OpenResearcher interface: {requirement.action_interface}")
    return OpenResearcherHybridState.model_validate(requirement.initial_state)


def _retrieval_tools(top_k: int, include_get_document: bool) -> tuple[FunctionDefinition, ...]:
    search = FunctionDefinition(
        name="search",
        description=f"Perform a search on a knowledge source. Returns top-{top_k} hits with docid, score, "
        "and snippet. The snippet contains the document's contents (may be truncated based on token limits).",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )
    if not include_get_document:
        return (search,)
    return (
        search,
        FunctionDefinition(
            name="get_document",
            description="Retrieve a full document by its docid.",
            parameters={
                "type": "object",
                "properties": {"docid": {"type": "string"}},
                "required": ["docid"],
                "additionalProperties": False,
            },
        ),
    )


def openresearcher_tools(state: OpenResearcherState) -> tuple[FunctionDefinition, ...]:
    return _retrieval_tools(state.top_k, state.include_get_document)


def openresearcher_hybrid_tools(state: OpenResearcherHybridState) -> tuple[FunctionDefinition, ...]:
    return _retrieval_tools(state.top_k, state.include_get_document)
