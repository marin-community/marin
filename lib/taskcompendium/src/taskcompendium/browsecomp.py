# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The pinned offline BrowseComp-Plus BM25 action interface."""

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from taskcompendium.models import FunctionDefinition, ProviderRequirement

BROWSECOMP_ACTION_INTERFACE = "browsecomp-plus-bm25:v1"
BROWSECOMP_MODULE = "browsecomp.tool"


class BrowseCompState(BaseModel):
    """Public retrieval settings; the declared actor image contains the frozen index."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    corpus_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    index_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    top_k: int = Field(gt=0)
    snippet_max_tokens: int = Field(gt=0)
    include_get_document: bool


def browsecomp_state(requirement: ProviderRequirement) -> BrowseCompState:
    if requirement.action_interface != BROWSECOMP_ACTION_INTERFACE:
        raise ValueError(f"Unsupported BrowseComp interface: {requirement.action_interface}")
    return BrowseCompState.model_validate(requirement.initial_state)


class OfflineSearchSettings(Protocol):
    @property
    def top_k(self) -> int: ...

    @property
    def include_get_document(self) -> bool: ...


def browsecomp_tools(state: OfflineSearchSettings) -> tuple[FunctionDefinition, ...]:
    search = FunctionDefinition(
        name="search",
        description=f"Perform a search on a knowledge source. Returns top-{state.top_k} hits with docid, score, "
        "and snippet. The snippet contains the document's contents (may be truncated based on token limits).",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )
    if not state.include_get_document:
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
