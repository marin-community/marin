# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read the current RL loop vocabulary from its public Google Doc."""

import hashlib
import re
import urllib.request
from dataclasses import dataclass

RULE_CODE = "ml-rl-loop-vocabulary"
DOC_ID = "1hi-EZaEDPMlpmCoPAJVfdVxsvTXy3gPFVFraiYol-Uc"
DOC_URL = f"https://docs.google.com/document/d/{DOC_ID}/edit"
EXPORT_URL = f"https://docs.google.com/document/d/{DOC_ID}/export?format=md"
EXPORT_TIMEOUT = 20
# The Markdown export appends each image as a base64 reference definition.
INLINE_IMAGE = re.compile(r"^!\[\]\[image\d+\][ \t]*$|^\[image\d+\]: <data:image/[^>]*>$", re.MULTILINE)


@dataclass(frozen=True)
class GlossarySource:
    content_sha: str
    text: str


def glossary_from_markdown(markdown: str) -> GlossarySource:
    """Drop inline images and keep the `~~OldName~~` proposed renames."""
    text = INLINE_IMAGE.sub("", markdown).strip()
    if not text:
        raise ValueError("RL Loop Glossary has no readable text")
    return GlossarySource(hashlib.sha256(text.encode()).hexdigest()[:12], text)


def fetch_glossary() -> GlossarySource:
    with urllib.request.urlopen(EXPORT_URL, timeout=EXPORT_TIMEOUT) as response:
        return glossary_from_markdown(response.read().decode())
