# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Open Knowledge Format (OKF) documents for Echo wiki entries.

OKF represents a unit of knowledge as one markdown file with a YAML frontmatter block
(``type`` required; ``title``/``description``/``resource``/``tags``/``timestamp`` optional).
See https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing.

Echo stores wiki entries in Postgres; this module is the file-format boundary. It parses an
OKF document into the fields a wiki write needs and renders a stored entry back as OKF, so an
entry round-trips through a ``.md`` file that a human or agent can author in an editor.

An Echo wiki note is the OKF document::

    ---
    type: wiki-note
    title: <title>
    use_when: <one sentence: when an agent should load this note>
    author: <author>          # set by the server on write; ignored on input
    timestamp: <ISO 8601>     # updated_at; ignored on input
    resource: <browse URL>    # optional
    ---

    <markdown body>
"""

import re
from dataclasses import dataclass

import yaml

WIKI_TYPE = "wiki-note"

# Leading `---` fenced YAML frontmatter, then the markdown body.
_FRONTMATTER = re.compile(r"^\s*---\n(.*?)\n---\n?(.*)$", re.DOTALL)


@dataclass(frozen=True)
class OkfDocument:
    """An OKF concept file: its ``type``, the full frontmatter mapping, and the markdown body."""

    type: str
    frontmatter: dict
    body: str


@dataclass(frozen=True)
class WikiFields:
    """The wiki-entry fields carried by an OKF document."""

    title: str
    use_when: str
    tags: tuple[str, ...]
    body: str


def parse(text: str) -> OkfDocument:
    """Parse an OKF markdown document into its frontmatter and body.

    Raises ValueError when the leading ``---`` YAML block or the required ``type`` field is
    absent, so a malformed file fails loudly instead of writing empty columns.
    """
    match = _FRONTMATTER.match(text)
    if match is None:
        raise ValueError("not an OKF document: expected a leading '---' YAML frontmatter block")
    frontmatter = yaml.safe_load(match.group(1))
    if not isinstance(frontmatter, dict):
        raise ValueError("OKF frontmatter must be a YAML mapping")
    entry_type = frontmatter.get("type")
    if not entry_type:
        raise ValueError("OKF frontmatter is missing the required 'type' field")
    return OkfDocument(type=str(entry_type), frontmatter=frontmatter, body=match.group(2).strip())


def parse_wiki(text: str) -> WikiFields:
    """Parse an OKF wiki note into the fields a wiki write needs."""
    document = parse(text)
    title = str(document.frontmatter.get("title", "")).strip()
    use_when = str(document.frontmatter.get("use_when", "")).strip()
    raw_tags = document.frontmatter.get("tags", [])
    if not isinstance(raw_tags, list) or not all(isinstance(tag, str) for tag in raw_tags):
        raise ValueError("OKF wiki note tags must be a YAML list of strings")
    missing = [name for name, value in (("title", title), ("use_when", use_when), ("body", document.body)) if not value]
    if missing:
        raise ValueError(f"OKF wiki note is missing: {', '.join(missing)}")
    return WikiFields(title=title, use_when=use_when, tags=tuple(raw_tags), body=document.body)


def emit(frontmatter: dict, body: str) -> str:
    """Render a frontmatter mapping and body as an OKF markdown document."""
    yaml_block = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{yaml_block}\n---\n\n{body.strip()}\n"


def wiki_to_okf(entry: dict, *, resource: str | None = None) -> str:
    """Render a stored wiki entry as an OKF document."""
    frontmatter: dict = {"type": WIKI_TYPE, "title": entry["title"], "use_when": entry["use_when"]}
    if entry.get("tags"):
        frontmatter["tags"] = entry["tags"]
    if entry.get("author"):
        frontmatter["author"] = entry["author"]
    if entry.get("updated_at"):
        frontmatter["timestamp"] = entry["updated_at"]
    if resource:
        frontmatter["resource"] = resource
    return emit(frontmatter, entry.get("body", ""))
