# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``fetch-v1`` tool-provider contract: a ``fetch_url`` tool served from pages a task declares.

A task declares the pages in ``ProviderRequirement.initial_state`` as ``{"pages": {url: page}}``. A page
is ``{"status": int, "body": str}`` for an HTTP response, or ``{"error": str}`` for a network failure.
A provider answers every call with JSON content, so an HTTP error, a network failure and an unknown
URL each reach the model and the grader as distinct observations instead of interrupting the attempt.
"""

from pydantic import BaseModel, ConfigDict, Field

from taskcompendium.models import FunctionDefinition, ProviderRequirement

FETCH_ACTION_INTERFACE = "fetch-v1"
FETCH_TOOL = FunctionDefinition(
    name="fetch_url",
    description="Retrieve a URL and return its HTTP status and body.",
    parameters={
        "type": "object",
        "properties": {"url": {"type": "string", "description": "The URL to retrieve."}},
        "required": ["url"],
        "additionalProperties": False,
    },
)
NOT_FOUND_STATUS = 404


class HttpPage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: int = Field(ge=100, le=599)
    body: str


class NetworkFailure(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    error: str = Field(min_length=1)


class FetchState(BaseModel):
    """The pages one fetch requirement serves."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pages: dict[str, HttpPage | NetworkFailure]


def fetch_requirement(pages: dict[str, HttpPage | NetworkFailure]) -> ProviderRequirement:
    """A ``fetch-v1`` requirement that serves ``pages``."""
    return ProviderRequirement(
        action_interface=FETCH_ACTION_INTERFACE, initial_state=FetchState(pages=pages).model_dump(mode="json")
    )
