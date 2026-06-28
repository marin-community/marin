# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The realized artifact, its on-disk record, and the drift check.

An :class:`Artifact` is the produced, persisted output of a step; an ``ArtifactStep`` is the
inert handle that builds one. This module owns the
framework — one base type and the record around it — while concrete artifact types live with
their producers (``LevanterCheckpoint`` in ``marin.training.training``, ``TokenizedCache`` in
``marin.processing.tokenize.tokenize``):

- :class:`Artifact` — a directory with a record (provenance + an optional JSON payload) and a
  ``load`` that reads it back. The default ``load`` returns a handle into the path; a subclass
  that declares value fields round-trips them through the record's ``result`` automatically.
- :class:`ArtifactRecord` — the single descriptor written next to a step's output: its config,
  fingerprint, provenance, and (for a value artifact) its ``result``.
- ``read_record``/``write_record`` (the full record) and ``read_artifact``/``write_artifact``
  (the manual typed-payload API), two entry points over one serialization scheme.
- :func:`check_drift` — the advisory recipe-drift guard the runner applies before serving a
  cached output.
"""

import functools
import logging
from dataclasses import asdict, is_dataclass
from typing import Self, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem import marin_prefix, open_url, url_to_fs

from marin.execution.fingerprint import describe_drift
from marin.execution.provenance import Provenance
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


class FingerprintMismatchError(Exception):
    """The opt-in hard identity gate: an ``expected_fingerprint`` pin differs from the
    computed fingerprint (at ``lower``) or from a pinned artifact's recorded fingerprint
    (in :func:`check_drift`)."""


class ArtifactTypeMismatchError(Exception):
    """A served record's ``result_type`` differs from the requested handle's ``result_type``."""


class Artifact(BaseModel):
    """A produced, persisted artifact: a directory with a record and a ``load``.

    The default ``load`` is a data ref — it returns a handle into ``path`` whose ``.record``
    carries provenance and the run's config, pulling no weights/caches into the launcher. A
    subclass that declares value fields persists and reloads them through ``record.result`` with
    no override (see :meth:`result_payload`). Not frozen: ``load`` sets ``path``.
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    path: str = ""

    @functools.cached_property
    def record(self) -> "ArtifactRecord | None":
        """The record sidecar at ``path`` (read once), or ``None`` if absent."""
        return read_record(self.path)

    def result_payload(self) -> dict | None:
        """What the record stores as ``result``: this artifact's *declared* value fields
        (every field but ``path``), or ``None`` for a pure data ref that declares none.

        Uses ``type(self).model_fields`` rather than ``model_dump()`` so ``extra="allow"`` extras
        (e.g. the cached ``record``) never leak into the payload. Override to persist something
        else.
        """
        keys = {name for name in type(self).model_fields if name != "path"}
        if not keys:
            return None
        return self.model_dump(mode="json", include=keys) or None

    @classmethod
    def load(cls, source: str) -> Self:
        """A handle into ``source``; a subclass with value fields repopulates them from
        ``record.result``."""
        rec = read_record(source)
        data = (rec.result if rec is not None else None) or {}
        return cls(path=source, **data)


class ArtifactRecord(BaseModel):
    """The single descriptor written next to a step's output.

    All fields carry a default, so a minimal manual record (:func:`write_artifact`) and a
    pre-existing legacy file both load without error; the lazy runner fills them all.
    """

    name: str = ""
    version: str = ""
    fingerprint: str = ""
    result_type: str = ""
    output_path: str = ""
    deps: list[str] = Field(default_factory=list)
    """Dependency identities as ``name@version`` strings."""
    config: dict[str, JSONValue] | None = None
    """The materialized config that ran (canonical-encoded), for humans and consumer metadata."""
    source: str | None = None
    """For an adopted artifact, the pre-existing data location this ``name@version`` aliases."""
    result: dict[str, JSONValue] | None = None
    """A value artifact's declared fields; ``None`` for a data artifact."""
    fingerprint_payload: str | None = None
    """The canonical config JSON the ``fingerprint`` hashes, kept for the drift diff."""
    provenance: Provenance | None = None
    """Who/when/which-commit/which-argv produced this — ``None`` for a minimal manual write."""


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

    change = describe_drift(record.fingerprint_payload, step.fingerprint_payload)
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
