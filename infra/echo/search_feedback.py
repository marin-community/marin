# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validation shared by Echo search-feedback clients and the API."""

from dataclasses import dataclass

import search_config

MAX_QUERY_CHARACTERS = 2_000
MAX_NOTE_CHARACTERS = 2_000
MAX_RESULT_ID_CHARACTERS = 4_096
MAX_GRADES = search_config.RERANK_MAX_CANDIDATES


def checked_result_id(value: str) -> str:
    domain, separator, detail = value.partition(":")
    if not separator or domain not in search_config.SEARCH_DOMAINS or not detail:
        raise ValueError("must be <wiki|file|discord|pr|issue>:<id>")
    if len(value) > MAX_RESULT_ID_CHARACTERS:
        raise ValueError(f"must be at most {MAX_RESULT_ID_CHARACTERS} characters")
    if domain != "file" and not detail.isdecimal():
        raise ValueError(f"{domain} result IDs are numeric")
    return value


@dataclass(frozen=True)
class FeedbackGrade:
    result_id: str
    grade: int

    @classmethod
    def from_spec(cls, value: str) -> "FeedbackGrade":
        result_id, separator, grade_text = value.rpartition("=")
        if not separator:
            raise ValueError("must be <result-id>=<0-10>")
        checked_result_id(result_id)
        try:
            grade = int(grade_text)
        except ValueError as error:
            raise ValueError("grade must be an integer from 0 through 10") from error
        if not 0 <= grade <= 10:
            raise ValueError("grade must be an integer from 0 through 10")
        return cls(result_id=result_id, grade=grade)
