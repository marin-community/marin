# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Translate open-instruct IFEval descriptors into verifyit constraints."""

import json
from collections.abc import Mapping
from typing import Any

from verifyit.modes.grade_ifeval import grade_ifeval_candidate, resolve_checks
from verifyit.spec import Constraint, IfevalSpec

# Each entry names the verifyit check and maps required source arguments to its parameters.
# The source records also carry unrelated null fields; only the check's arguments are retained.
OPEN_INSTRUCT_CHECKS: dict[str, tuple[str, dict[str, str], dict[str, Any]]] = {
    "verify_keywords": ("keywords:existence", {"keyword_list": "keywords"}, {"word_boundary": False}),
    "verify_keyword_frequency": ("keywords:frequency", {"word": "keyword", "N": "frequency"}, {"relation": "exactly"}),
    "verify_keyword_frequency_relation": (
        "open_instruct:keyword_frequency_relation",
        {"keyword_list": "keyword_list", "N": "N", "quantifier": "quantifier"},
        {},
    ),
    "validate_forbidden_words": (
        "keywords:forbidden_words",
        {"forbidden_words": "forbidden_words"},
        {"word_boundary": False},
    ),
    "verify_letter_frequency": ("open_instruct:letter_frequency", {"letter": "letter", "N": "N"}, {}),
    "validate_response_language": ("language:response_language", {"language": "language"}, {}),
    "verify_paragraph_count": ("length_constraints:number_paragraphs", {"N": "num_paragraphs"}, {}),
    "validate_word_constraint": ("open_instruct:word_constraint", {"N": "N", "quantifier": "quantifier"}, {}),
    "verify_sentence_constraint": ("open_instruct:sentence_constraint", {"N": "N", "quantifier": "quantifier"}, {}),
    "validate_paragraphs": (
        "length_constraints:nth_paragraph_first_word",
        {"N": "num_paragraphs", "first_word": "first_word", "i": "nth_paragraph"},
        {},
    ),
    "verify_postscript": ("detectable_content:postscript", {"postscript_marker": "postscript_marker"}, {}),
    "validate_placeholders": ("detectable_content:number_placeholders", {"N": "num_placeholders"}, {}),
    "verify_bullet_points": ("detectable_format:number_bullet_lists", {"N": "num_bullets"}, {}),
    "validate_title": ("detectable_format:title", {}, {}),
    "validate_choice": ("open_instruct:choice", {"options": "options"}, {}),
    "validate_highlighted_sections": ("detectable_format:number_highlighted_sections", {"N": "num_highlights"}, {}),
    "validate_sections": (
        "detectable_format:multiple_sections",
        {"N": "num_sections", "section_splitter": "section_splitter"},
        {},
    ),
    "validate_json_format": ("detectable_format:json_format", {}, {}),
    "validate_repeat_prompt": ("open_instruct:repeat_prompt", {"original_prompt": "original_prompt"}, {}),
    "validate_two_responses": ("combination:two_responses", {}, {}),
    "validate_uppercase": ("change_case:english_capital", {}, {}),
    "validate_lowercase": ("change_case:english_lowercase", {}, {}),
    "validate_frequency_capital_words": ("open_instruct:capital_words", {"N": "N", "quantifier": "quantifier"}, {}),
    "validate_end": ("startend:end_checker", {"end_phrase": "end_phrase"}, {}),
    "validate_quotation": ("startend:quotation", {}, {}),
    "validate_no_commas": ("punctuation:no_comma", {}, {}),
}


def normalized_descriptors(ground_truth: Any) -> list[dict[str, Any]]:
    """Validate a source record and retain only arguments required by its checks."""
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    descriptors = ground_truth if isinstance(ground_truth, list) else [ground_truth]
    if not descriptors or any(not isinstance(item, Mapping) for item in descriptors):
        raise ValueError("IFeval ground truth requires a nonempty constraint mapping or list")
    normalized = []
    for descriptor in descriptors:
        name = descriptor.get("func_name")
        if not isinstance(name, str) or name not in OPEN_INSTRUCT_CHECKS:
            raise ValueError(f"Unknown IFEval func_name: {name!r}")
        _, arguments, _ = OPEN_INSTRUCT_CHECKS[name]
        missing = [key for key in arguments if descriptor.get(key) is None]
        if missing:
            raise ValueError(f"IFeval {name!r} requires non-null {missing}")
        for key in arguments:
            value = descriptor[key]
            if key in {"N", "i"} and (type(value) is not int or value < (1 if key == "i" else 0)):
                raise ValueError(f"IFeval {name!r} has an invalid {key}")
            if key in {"keyword_list", "forbidden_words", "options"} and (
                not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value)
            ):
                raise ValueError(f"IFeval {name!r} requires a nonempty {key} list")
            if key not in {"N", "i", "keyword_list", "forbidden_words", "options"} and (
                not isinstance(value, str) or not value
            ):
                raise ValueError(f"IFeval {name!r} requires nonempty text for {key}")
        if descriptor.get("N") == 0 and (
            name in {"validate_placeholders", "validate_highlighted_sections", "validate_sections"}
            or descriptor.get("quantifier") == "at least"
        ):
            raise ValueError(f"IFeval {name!r} has a vacuous minimum count")
        normalized.append({"func_name": name, **{key: descriptor[key] for key in arguments}})
    return normalized


def normalize_ground_truth(ground_truth: Any) -> str:
    """Return canonical source descriptors after checking verifyit can resolve them."""
    descriptors = normalized_descriptors(ground_truth)
    constraints = constraints_from_descriptors(descriptors)
    resolve_checks(constraints)
    decoded = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    output: Any = descriptors if isinstance(decoded, list) else descriptors[0]
    return json.dumps(output, sort_keys=True, separators=(",", ":"))


def constraints_from_descriptors(ground_truth: Any) -> tuple[Constraint, ...]:
    """Translate source descriptors to the canonical verifyit check names."""
    constraints = []
    for descriptor in normalized_descriptors(ground_truth):
        name, arguments, constants = OPEN_INSTRUCT_CHECKS[descriptor["func_name"]]
        params = {**constants, **{target: descriptor[source] for source, target in arguments.items()}}
        constraints.append(Constraint(name, params))
    return tuple(constraints)


def compute_score(response: str, ground_truth: Any) -> dict[str, Any]:
    """Score source constraints through verifyit; report invalid source data separately."""
    try:
        descriptors = normalized_descriptors(ground_truth)
    except (ValueError, TypeError):
        return {
            "score": 0.0,
            "acc": False,
            "func_name": None,
            "constraints_satisfied": 0,
            "constraints_total": 0,
            "error_type": "schema_error",
        }
    verdict = grade_ifeval_candidate(IfevalSpec(constraints_from_descriptors(descriptors)), response)
    passed = sum(result["passed"] for result in verdict.detail.get("constraints", []))
    total = len(descriptors)
    return {
        "score": passed / total,
        "acc": passed == total,
        "func_name": (
            descriptors[0]["func_name"]
            if total == 1
            and not isinstance(json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth, list)
            else None
        ),
        "constraints_satisfied": passed,
        "constraints_total": total,
    }
