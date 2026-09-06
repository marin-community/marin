# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validation shared by Echo search-feedback clients and the API."""

from dataclasses import dataclass

from . import search_config

MAX_QUERY_CHARACTERS = 2_000
MAX_NOTE_CHARACTERS = 2_000
MAX_RESULT_KEY_CHARACTERS = 32
MAX_ARTIFACT_ID_CHARACTERS = 4_096
MAX_GRADES = search_config.RERANK_MAX_CANDIDATES
MAX_NUMERIC_RESULT_ID = 2**63 - 1
MIN_GRADE = search_config.SEARCH_FEEDBACK_MIN_GRADE
MAX_GRADE = search_config.SEARCH_FEEDBACK_MAX_GRADE


def checked_artifact_id(value: str) -> str:
    domain, separator, detail = value.partition(":")
    if not separator or domain not in search_config.SEARCH_DOMAINS or not detail:
        raise ValueError("must be <wiki|file|discord|pr|issue>:<id>")
    if len(value) > MAX_ARTIFACT_ID_CHARACTERS:
        raise ValueError(f"must be at most {MAX_ARTIFACT_ID_CHARACTERS} characters")
    if domain != "file":
        if not detail.isdecimal():
            raise ValueError(f"{domain} result IDs are numeric")
        if int(detail) > MAX_NUMERIC_RESULT_ID:
            raise ValueError(f"{domain} result IDs must fit a signed 64-bit integer")
    return value


def checked_result_key(value: str) -> str:
    domain, separator, detail = value.partition(":")
    if not separator or domain not in search_config.SEARCH_DOMAINS or not detail:
        raise ValueError("must be <wiki|file|discord|pr|issue>:<numeric-key>")
    if len(value) > MAX_RESULT_KEY_CHARACTERS:
        raise ValueError(f"must be at most {MAX_RESULT_KEY_CHARACTERS} characters")
    if not detail.isdecimal():
        raise ValueError("search result keys are numeric")
    if not 0 < int(detail) <= MAX_NUMERIC_RESULT_ID:
        raise ValueError("search result keys must be positive signed 64-bit integers")
    return value


def result_key_parts(value: str) -> tuple[str, int]:
    checked_result_key(value)
    domain, _, detail = value.partition(":")
    return domain, int(detail)


@dataclass(frozen=True)
class FeedbackGrade:
    key: str
    grade: int

    @classmethod
    def from_spec(cls, value: str) -> "FeedbackGrade":
        key, separator, grade_text = value.rpartition("=")
        if not separator:
            raise ValueError(f"must be <result-key>=<{MIN_GRADE}-{MAX_GRADE}>")
        checked_result_key(key)
        try:
            grade = int(grade_text)
        except ValueError as error:
            raise ValueError(f"grade must be an integer from {MIN_GRADE} through {MAX_GRADE}") from error
        if not MIN_GRADE <= grade <= MAX_GRADE:
            raise ValueError(f"grade must be an integer from {MIN_GRADE} through {MAX_GRADE}")
        return cls(key=key, grade=grade)
