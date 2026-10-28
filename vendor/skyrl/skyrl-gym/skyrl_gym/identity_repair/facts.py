"""Versioned identity fact manifest supplied by the campaign operator.

The manifest carries the canonical assistant name, creator, organization, gradeable office locations, and the
provenance of every fact. Alias forms are matched case-insensitively after whitespace collapsing and trailing
punctuation stripping; `normalize_entity` is the single normalization used everywhere in this package.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

TRAILING_PUNCTUATION = " \t.,!?;:\"'“”‘’"


def normalize_entity(text: str) -> str:
    """Return the alias-comparison form: lowercased, whitespace-collapsed, trailing punctuation stripped."""
    return " ".join(text.split()).lower().rstrip(TRAILING_PUNCTUATION).rstrip()


def _alias_forms(canonical: str, aliases: tuple[str, ...]) -> tuple[str, ...]:
    forms: list[str] = []
    for surface in (canonical, *aliases):
        form = normalize_entity(surface)
        if form and form not in forms:
            forms.append(form)
    return tuple(forms)


@dataclass(frozen=True)
class FactManifest:
    """The campaign's canonical identity facts and who supplied each one."""

    version: str
    identity_name: str
    identity_aliases: tuple[str, ...]
    creator: str
    creator_aliases: tuple[str, ...]
    organization: str
    organization_aliases: tuple[str, ...]
    offices: tuple[str, ...]
    provenance: Mapping[str, str]

    @property
    def identity_forms(self) -> tuple[str, ...]:
        """Lowercased acceptable surface forms of the assistant name, canonical first."""
        return _alias_forms(self.identity_name, self.identity_aliases)

    @property
    def creator_forms(self) -> tuple[str, ...]:
        """Lowercased acceptable surface forms of the creator, canonical first."""
        return _alias_forms(self.creator, self.creator_aliases)

    @property
    def organization_forms(self) -> tuple[str, ...]:
        """Lowercased acceptable surface forms of the organization, canonical first."""
        return _alias_forms(self.organization, self.organization_aliases)

    def sha256(self) -> str:
        """Return the sha256 of the canonical sort_keys JSON serialization of every field."""
        payload = {
            "version": self.version,
            "identity_name": self.identity_name,
            "identity_aliases": list(self.identity_aliases),
            "creator": self.creator,
            "creator_aliases": list(self.creator_aliases),
            "organization": self.organization,
            "organization_aliases": list(self.organization_aliases),
            "offices": list(self.offices),
            "provenance": dict(self.provenance),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


MANIFEST_FIELDS = (
    "version",
    "identity_name",
    "identity_aliases",
    "creator",
    "creator_aliases",
    "organization",
    "organization_aliases",
    "offices",
    "provenance",
)

_CANONICAL_ALIASES = (
    ("identity_name", "identity_aliases"),
    ("creator", "creator_aliases"),
    ("organization", "organization_aliases"),
)


def _validate_manifest(manifest: FactManifest) -> None:
    if not manifest.version.strip():
        raise ValueError("Fact manifest version must be non-empty.")
    for canonical_field, aliases_field in _CANONICAL_ALIASES:
        canonical = getattr(manifest, canonical_field)
        aliases = getattr(manifest, aliases_field)
        if not aliases:
            raise ValueError(f"Fact manifest {aliases_field} must be non-empty.")
        forms = {normalize_entity(alias) for alias in aliases}
        if normalize_entity(canonical) not in forms:
            raise ValueError(f"Fact manifest {canonical_field} must appear in {aliases_field}.")


def load_fact_manifest(path: Path) -> FactManifest:
    """Load and validate a fact manifest JSON file, rejecting unknown fields."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Fact manifest {path} must be a JSON object.")
    unknown = sorted(set(data) - set(MANIFEST_FIELDS))
    if unknown:
        raise ValueError(f"Fact manifest {path} has unknown fields: {unknown}")
    missing = [field for field in MANIFEST_FIELDS if field not in data]
    if missing:
        raise ValueError(f"Fact manifest {path} is missing fields: {missing}")
    for field in ("identity_aliases", "creator_aliases", "organization_aliases", "offices"):
        if not isinstance(data[field], list) or not all(isinstance(entry, str) for entry in data[field]):
            raise ValueError(f"Fact manifest {path} field {field} must be a list of strings.")
    for field in ("version", "identity_name", "creator", "organization"):
        if not isinstance(data[field], str):
            raise ValueError(f"Fact manifest {path} field {field} must be a string.")
    if not isinstance(data["provenance"], dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in data["provenance"].items()
    ):
        raise ValueError(f"Fact manifest {path} field provenance must map fact keys to supplier strings.")
    manifest = FactManifest(
        version=data["version"],
        identity_name=data["identity_name"],
        identity_aliases=tuple(data["identity_aliases"]),
        creator=data["creator"],
        creator_aliases=tuple(data["creator_aliases"]),
        organization=data["organization"],
        organization_aliases=tuple(data["organization_aliases"]),
        offices=tuple(data["offices"]),
        provenance=dict(data["provenance"]),
    )
    _validate_manifest(manifest)
    return manifest
