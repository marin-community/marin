# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submission decoding with explicit, unambiguous answer conventions."""

import json
import re
import xml.etree.ElementTree as ET

from taskcompendium.models import BoxedLatex, Extractor, JsonPath, PlainText, XmlPath

_JSON_PATH = re.compile(r"\$(?:\.[A-Za-z_][A-Za-z0-9_]*|\[[0-9]+\])*")
_JSON_COMPONENT = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)|\[([0-9]+)\]")
_XML_PATH = re.compile(r"/(?:[A-Za-z_][A-Za-z0-9_-]*)(?:/[A-Za-z_][A-Za-z0-9_-]*)*")


class ExtractionError(ValueError):
    """The attempt does not satisfy the selected submission protocol."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ExtractionError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ExtractionError(f"Non-JSON numeric constant: {value}")


def validate_extractor(extractor: Extractor) -> None:
    if isinstance(extractor, JsonPath) and not _JSON_PATH.fullmatch(extractor.path):
        raise ValueError("JSON paths support only named fields and nonnegative array indices")
    if isinstance(extractor, XmlPath) and not _XML_PATH.fullmatch(extractor.path):
        raise ValueError("XML paths must name a root and optional child elements")


def extract(text: str, extractor: Extractor) -> str:
    """Decode exactly the configured wrapper into a canonical candidate string."""
    validate_extractor(extractor)
    if not text.strip():
        raise ExtractionError("Submission is empty")
    if isinstance(extractor, PlainText):
        return text
    if isinstance(extractor, BoxedLatex):
        starts = list(re.finditer(r"\\boxed\s*\{", text))
        if len(starts) != 1:
            raise ExtractionError("Expected exactly one boxed answer")
        start = starts[0].end()
        depth = 1
        for index in range(start, len(text)):
            if text[index] == "{" and (index == 0 or text[index - 1] != "\\"):
                depth += 1
            elif text[index] == "}" and (index == 0 or text[index - 1] != "\\"):
                depth -= 1
            if depth == 0:
                value = text[start:index]
                if not value.strip():
                    raise ExtractionError("Boxed answer is empty")
                return value
        raise ExtractionError("Unclosed boxed answer")
    if isinstance(extractor, JsonPath):
        try:
            value = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
            for match in _JSON_COMPONENT.finditer(extractor.path):
                key, index = match.groups()
                if key is not None:
                    if not isinstance(value, dict):
                        raise ExtractionError("JSON path expected an object")
                    value = value[key]
                else:
                    if not isinstance(value, list):
                        raise ExtractionError("JSON path expected an array")
                    value = value[int(index)]
        except (json.JSONDecodeError, KeyError, IndexError) as error:
            raise ExtractionError(f"Invalid JSON submission: {error}") from error
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ExtractionError("JSON answer is empty or null")
        return value if isinstance(value, str) else json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(extractor, XmlPath):
        if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
            raise ExtractionError("XML declarations and entities are not allowed")
        try:
            root = ET.fromstring(text)
        except ET.ParseError as error:
            raise ExtractionError(f"Invalid XML submission: {error}") from error
        components = extractor.path.strip("/").split("/")
        matches = [root] if root.tag == components[0] else []
        for component in components[1:]:
            matches = [child for parent in matches for child in parent if child.tag == component]
        if len(matches) != 1 or len(matches[0]) or not (matches[0].text or "").strip():
            raise ExtractionError("XML path must select exactly one nonempty text element")
        return matches[0].text or ""
    raise TypeError(f"Unsupported extractor: {type(extractor)}")


def rendering_instruction(extractor: Extractor) -> str:
    validate_extractor(extractor)
    if isinstance(extractor, PlainText):
        return "Return only the answer in the requested format."
    if isinstance(extractor, BoxedLatex):
        return r"Give exactly one final answer inside \boxed{...}."
    if isinstance(extractor, JsonPath):
        return f"Return valid JSON with the answer at {extractor.path}; do not use Markdown fences."
    return f"Return valid XML with the answer text at {extractor.path}; do not use Markdown fences."
