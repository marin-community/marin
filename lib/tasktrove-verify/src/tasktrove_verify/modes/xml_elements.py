# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mode xml-elements: the answer must be well-formed XML carrying the field names the task asks for.

The document is parsed with ElementTree, and every element tag and attribute name in it, namespace
prefix stripped, becomes one name. ``required`` names must all be there; ``any_of`` names, used
where the task marks nothing required, need one hit. Values, nesting and order are not checked:
this grades the shape of an XML answer, not its content.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

from tasktrove_verify.modes.extract import unwrap_fence
from tasktrove_verify.output import read_output
from tasktrove_verify.reward import InvalidTask, Reward, scored
from tasktrove_verify.spec import XmlElementsSpec

MAX_REPORTED_NAMES = 8


def local_name(name: str) -> str:
    """``{namespace}tag`` reduced to ``tag``; a name without a namespace is unchanged."""
    return name.rsplit("}", 1)[-1]


def document_names(root: ET.Element) -> set[str]:
    """Every element tag and attribute name in the tree. Comments and processing instructions,
    whose tag is a callable rather than a string, contribute nothing."""
    names: set[str] = set()
    for element in root.iter():
        if isinstance(element.tag, str):
            names.add(local_name(element.tag))
        names.update(local_name(attribute) for attribute in element.attrib)
    return names


def grade(spec: XmlElementsSpec, tests_dir: Path, workspace: Path) -> Reward:
    if not spec.required and not spec.any_of:
        raise InvalidTask("xml-elements expects required or any_of names")

    text = read_output(spec, workspace)
    if text is None:
        return scored(0.0, reason="no_output")
    try:
        root = ET.fromstring(unwrap_fence(text).strip())
    except ET.ParseError as error:
        return scored(0.0, reason="parse_error", error=str(error))

    names = document_names(root)
    missing = [name for name in spec.required if name not in names]
    if missing:
        return scored(0.0, reason="missing_elements", missing=missing[:MAX_REPORTED_NAMES])
    if spec.any_of and names.isdisjoint(spec.any_of):
        return scored(0.0, reason="no_expected_element", expected=list(spec.any_of[:MAX_REPORTED_NAMES]))
    return scored(1.0, reason="names_present", names=len(names))
