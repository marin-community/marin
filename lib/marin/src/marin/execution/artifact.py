# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The realized artifact, its on-disk record, and the drift check.

An :class:`Artifact` is the produced, persisted output of a step; a
:class:`marin.execution.lazy.Lazy` is the inert handle that builds one. This module owns:

- :class:`Artifact` (and :class:`Dataset`/:class:`Checkpoint`/:class:`JsonArtifact`) —
  the produced, persisted value. ``load(path)`` reconstructs it; data refs return a
  path-bearing handle (no weights pulled), a :class:`JsonArtifact` reads its value out of
  the record.
- :class:`ArtifactRecord` — the single descriptor written next to a step's output: its
  config, fingerprint, provenance, and (for a value artifact) its ``result``.
- ``read_record``/``write_record`` (the full record) and ``read_artifact``/``write_artifact``
  (the manual typed-payload API), two entry points over one serialization scheme.
- :func:`check_drift` — the advisory recipe-drift guard the runner applies before serving
  a cached output.
"""

import functools
import json
import logging
from dataclasses import asdict, is_dataclass
from typing import Self, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem import marin_prefix, open_url, url_to_fs

from marin.execution.step_spec import StepSpec, _is_relative_path

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)

# JSON-shaped value, used for the human-readable config and the value payload.
type JSONValue = None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]

# The record file written next to every output. Legacy names are read for back-compat
# with already-materialized outputs, never written.
RECORD_FILENAME = "artifact.json"
_LEGACY_RECORD_FILENAMES = (".artifact_record.json", ".artifact")
_LEGACY_PAYLOAD_FILENAMES = (".artifact.json", ".artifact")

# Keys under ``StepSpec.hash_attrs`` carrying the artifact's identity, so the runner can
# apply the drift check without knowing about the lazy layer.
FINGERPRINT_KEY = "fingerprint"
VERSION_KEY = "version"
RESULT_TYPE_KEY = "result_type"
EXPECTED_FINGERPRINT_KEY = "expected_fingerprint"

# Cap on how many changed config values a drift message spells out before summarizing the
# remainder, so a wholesale recipe change stays readable.
_MAX_DIFF_LINES = 20


class FingerprintMismatchError(Exception):
    """The opt-in hard identity gate: an ``expected_fingerprint`` pin differs from the
    computed fingerprint (at ``lower``) or from a pinned artifact's recorded fingerprint
    (in :func:`check_drift`)."""


class ArtifactTypeMismatchError(Exception):
    """A served record's ``result_type`` differs from the requested handle's ``result_type``."""


class Artifact(BaseModel):
    """A produced, persisted artifact: its ``path`` plus a lazily-read provenance ``record``.

    ``load`` is concrete (a data ref) so the default ``result_type=Artifact`` is resolvable
    and weights/caches never enter the launcher. Not frozen: ``load`` sets ``path``.
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    path: str = ""

    @functools.cached_property
    def record(self) -> "ArtifactRecord | None":
        """The record sidecar at ``path`` (read once), or ``None`` if absent."""
        return read_record(self.path)

    @classmethod
    def load(cls, source: str) -> Self:
        """A data ref: return a handle into ``source`` without reading anything."""
        return cls(path=source)


class Dataset(Artifact):
    """A tokenized Levanter cache at ``path`` (inherits the path-ref ``load``)."""


class Checkpoint(Artifact):
    """A Levanter checkpoint dir at ``path`` (inherits the path-ref ``load``)."""


class JsonArtifact(Artifact):
    """A computed value persisted in the record's ``result``.

    Authors subclass this (declaring the value's fields) instead of writing ``load``.
    """

    @classmethod
    def load(cls, source: str) -> Self:
        rec = read_record(source)
        if rec is None:
            raise FileNotFoundError(f"no artifact record at {source}")
        obj = cls.model_validate(rec.result)
        obj.path = source
        return obj


class ArtifactRecord(BaseModel):
    """The single descriptor written next to a step's output.

    All fields except by-default-empty identity fields carry a default, so a minimal manual
    record (:func:`write_artifact`) and a pre-existing legacy file both load without error;
    the lazy runner fills them all.
    """

    name: str = ""
    version: str = ""
    fingerprint: str = ""
    result_type: str = ""
    output_path: str = ""
    deps: list[str] = Field(default_factory=list)
    """Dependency identities as ``name@version`` strings."""
    config: dict[str, JSONValue] | None = None
    """The materialized config that ran (canonical-encoded), for humans."""
    command_line: list[str] | None = None
    git_commit: str | None = None
    user: str | None = None
    created_at: str = ""
    source: str | None = None
    """For an adopted artifact, the pre-existing data location this ``name@version`` aliases."""
    result: dict[str, JSONValue] | None = None
    """``JsonArtifact.model_dump()`` for a value artifact; ``None`` for a data artifact."""
    fingerprint_payload: str | None = None
    """The canonical config JSON the ``fingerprint`` hashes, kept for the drift diff."""


def is_mutable_version(version: str) -> bool:
    """A ``dev`` version is mutable: the drift check is skipped and it always rebuilds."""
    return version == "dev" or version.endswith("-dev")


def _resolved(output_path: str) -> str:
    """A relative output path is rooted at ``marin_prefix()``; an absolute/URL path is used as-is.

    Mirrors the launcher's path resolution so a manual ``read_artifact``/``read_record`` of a
    relative step name reads the same location the runner wrote.
    """
    return f"{marin_prefix()}/{output_path}" if _is_relative_path(output_path) else output_path


def _join(output_path: str, filename: str) -> str:
    return f"{output_path.rstrip('/')}/{filename}"


def _read_text(output_path: str, filename: str) -> str | None:
    path = _join(output_path, filename)
    fs = url_to_fs(path, use_listings_cache=False)[0]
    if not fs.exists(path):
        return None
    with open_url(path, "r") as f:
        return f.read()


def read_record(output_path: str) -> ArtifactRecord | None:
    """The full record at ``{output_path}/artifact.json`` (or a legacy name), else ``None``.

    A corrupt/partial file raises :class:`pydantic.ValidationError`.
    """
    output_path = _resolved(output_path)
    for filename in (RECORD_FILENAME, *_LEGACY_RECORD_FILENAMES):
        text = _read_text(output_path, filename)
        if text is not None:
            return ArtifactRecord.model_validate_json(text)
    return None


def write_record(record: ArtifactRecord) -> None:
    """Write ``record`` to ``{record.output_path}/artifact.json``."""
    with open_url(_join(record.output_path, RECORD_FILENAME), "w") as f:
        f.write(record.model_dump_json(indent=2))


def _payload_json(value: object) -> JSONValue:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value  # pyrefly: ignore[bad-return]


def read_artifact(output_path: str, schema: type[M]) -> M:
    """Load a typed payload: ``read_record(output_path).result`` validated as ``schema``.

    Falls back to a legacy bare-payload sidecar when the record carries no ``result``.
    Raises :class:`FileNotFoundError` if nothing is present.
    """
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        raise TypeError(f"schema must be a pydantic BaseModel subclass, got {schema!r}")
    output_path = _resolved(output_path)
    record = read_record(output_path)
    if record is not None and record.result is not None:
        return cast(M, schema.model_validate(record.result))
    for filename in _LEGACY_PAYLOAD_FILENAMES:
        text = _read_text(output_path, filename)
        if text is not None:
            return cast(M, schema.model_validate_json(text))
    raise FileNotFoundError(f"no artifact payload at {output_path}")


def write_artifact(value: object, output_path: str) -> None:
    """Write a minimal record carrying ``value`` as its ``result`` — the manual save API."""
    write_record(ArtifactRecord(output_path=output_path, result=_payload_json(value)))


def _diff_json(old: object, new: object, prefix: str = "") -> list[str]:
    """Dotted-path descriptions of where ``old`` and ``new`` (parsed JSON) differ."""
    if isinstance(old, dict) and isinstance(new, dict):
        changes: list[str] = []
        for key in sorted(set(old) | set(new)):
            sub = f"{prefix}.{key}" if prefix else key
            if key not in old:
                changes.append(f"{sub}: (added) {new[key]!r}")
            elif key not in new:
                changes.append(f"{sub}: {old[key]!r} (removed)")
            else:
                changes.extend(_diff_json(old[key], new[key], sub))
        return changes
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [f"{prefix}: list of {len(old)} -> list of {len(new)}"]
        changes = []
        for i, (a, b) in enumerate(zip(old, new, strict=True)):
            changes.extend(_diff_json(a, b, f"{prefix}[{i}]"))
        return changes
    if old != new:
        return [f"{prefix or '(root)'}: {old!r} -> {new!r}"]
    return []


def _describe_change(existing: ArtifactRecord, payload: str | None) -> str:
    """A field-level summary of how the current recipe differs from the recorded one,
    or an empty string when neither side carries a payload to diff."""
    if payload is None or existing.fingerprint_payload is None:
        return ""
    changes = _diff_json(json.loads(existing.fingerprint_payload), json.loads(payload))
    if not changes:
        return ""
    shown = changes[:_MAX_DIFF_LINES]
    body = "\n".join(f"  {c}" for c in shown)
    if len(changes) > len(shown):
        body += f"\n  …and {len(changes) - len(shown)} more"
    return f"\nChanged config values:\n{body}"


def check_drift(step: StepSpec) -> bool:
    """Advisory recipe-drift guard, run before serving a cached SUCCESS.

    Returns ``False`` for a non-lazy step (no fingerprint). Returns ``True`` for a mutable
    (``dev``) version so the caller rebuilds. Otherwise, if a record exists whose fingerprint
    differs from the step's: raises :class:`FingerprintMismatchError` if the step carries an
    ``expected_fingerprint`` pin, else logs a field-level warning and returns ``False`` (the
    cached output is served).
    """
    fingerprint = step.hash_attrs.get(FINGERPRINT_KEY)
    if fingerprint is None:
        return False
    version = step.hash_attrs.get(VERSION_KEY, "")
    if is_mutable_version(version):
        return True
    record = read_record(step.output_path)
    if record is None or record.fingerprint == fingerprint:
        return False

    change = _describe_change(record, step.fingerprint_payload)
    if step.hash_attrs.get(EXPECTED_FINGERPRINT_KEY) is not None:
        raise FingerprintMismatchError(
            f"{step.name}@{version} is pinned to expected_fingerprint, but its recorded build has "
            f"fingerprint {record.fingerprint} (now {fingerprint}).{change} "
            f"Update the pin and bump the version if this is meant to be a different artifact."
        )
    logger.warning(
        "%s@%s: recipe drift — recorded fingerprint %s, now %s; serving the cached output. "
        "Bump the version to build the new recipe.%s",
        step.name,
        version,
        record.fingerprint,
        fingerprint,
        change,
    )
    return False
