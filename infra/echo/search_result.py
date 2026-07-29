# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checked federated-search result values for the Echo CLI."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    id: str
    domain: str
    title: str
    subtitle: str
    url: str
    snippet: str
    score: float
    distance: float | None
    lexical_score: float | None

    @classmethod
    def from_json(cls, value: object) -> "SearchResult":
        if not isinstance(value, dict):
            raise ValueError("search result must be an object")
        try:
            return cls(
                id=str(value["id"]),
                domain=str(value["domain"]),
                title=str(value["title"]),
                subtitle=str(value["subtitle"]),
                url=str(value["url"]),
                snippet=str(value["snippet"]),
                score=float(value["score"]),
                distance=optional_float(value.get("distance")),
                lexical_score=optional_float(value.get("lexical_score")),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid search result: {error}") from error


def optional_float(value: object) -> float | None:
    return None if value is None else float(value)
