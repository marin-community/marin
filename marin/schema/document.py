"""
document.py

Pydantic Schema definitions for all "content" classes (derivations of a top-level Document, e.g., DiscussionThread,
StackExchangeThread, ArxivPaper, etc.).
"""

from abc import ABC
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, HttpUrl, SerializeAsAny


# === Document :: Atomic Unit, following Dolma (https://github.com/allenai/dolma/blob/main/docs/data-format.md) ===
#   =>> We additionally define an abstract `DocumentMetadata` class =>> this is to be overridden by each "domain"
class DocumentMetadata(BaseModel, ABC):
    source: str


class Document(BaseModel):
    id: str
    text: str
    source: str

    # Note =>> All dates/times are assumed to be UTC
    added: datetime
    created: datetime

    # Note =>> Mark as `SerializeAsAny` to fully expand any subclass of DocumentMetadata
    #   See Pydantic docs: https://docs.pydantic.dev/latest/concepts/serialization/#serializing-with-duck-typing
    metadata: SerializeAsAny[DocumentMetadata]


# === Domain-Specific Metadata (e.g., ThreadedMetadata for StackExchange/Reddit) ===
class StackExchangeAnswer(BaseModel):
    id: str
    body: str
    creation_time_utc: datetime
    votes: int


class StackExchangeThreadMetadata(DocumentMetadata):
    source: str = "stackexchange"
    subdomain: str

    id: str
    url: HttpUrl
    title: str
    question: str
    tags: list[str]
    creation_time_utc: datetime

    votes: int
    accepted_answer_id: Optional[str]
    answers: list[StackExchangeAnswer]
