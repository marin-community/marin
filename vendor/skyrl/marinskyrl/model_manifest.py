"""Portable manifests for immutable Hugging Face model exports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import struct
from typing import BinaryIO, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator

from marinskyrl.hf_model import (
    normalize_fast_tokenizer_metadata,
    validate_hf_model_weights,
    validate_portable_hf_model_files,
)

MODEL_MANIFEST_FILENAME = ".marinskyrl-model-manifest.json"
HF_WEIGHT_INDEX_FILENAME = "model.safetensors.index.json"
MAX_SAFETENSORS_HEADER_BYTES = 100 * 2**20
_HEADER_READ_CHUNK_BYTES = 8 * 2**20


class ModelManifestFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    size: int = Field(ge=0, strict=True)
    sha256: str

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return validate_model_file_path(value)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        return validate_sha256_digest(value)


class ModelManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str | None
    revision: str | None
    tokenizer_mode: Literal["embedded", "policy"] = "embedded"
    identity: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    files: tuple[ModelManifestFile, ...] = Field(min_length=1)
    format_version: Literal[1] = 1

    @classmethod
    def from_mapping(cls, value: object, source: str) -> Self:
        try:
            return cls.model_validate(value, context={"source": source})
        except ValidationError as error:
            raise ValueError(f"Invalid model manifest at {source}: {error}") from error

    @model_validator(mode="after")
    def validate_manifest(self, info: ValidationInfo) -> Self:
        source = (info.context or {}).get("source", "model manifest")
        paths = [entry.path for entry in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError(f"Model manifest contains duplicate paths: {source}")
        expected = _manifest_identity(self.model_id, self.revision, self.files, self.tokenizer_mode)
        if self.identity != expected:
            raise ValueError(f"Model manifest identity mismatch at {source}: {self.identity} != {expected}")
        validate_model_file_names(set(paths), source, self.tokenizer_mode)
        if HF_WEIGHT_INDEX_FILENAME not in paths:
            raise ValueError(f"Model manifest is missing {HF_WEIGHT_INDEX_FILENAME}: {source}")
        return self


def validate_model_file_path(value: str) -> str:
    """Validate one relative path stored in a model manifest."""
    path = PurePosixPath(value)
    if value in ("", ".") or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"model file path must be relative and contained: {value!r}")
    return value


def validate_sha256_digest(value: str) -> str:
    """Validate and return a lowercase SHA-256 digest."""
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"invalid SHA-256 digest: {value!r}")
    return value


def validate_model_file_names(
    names: set[str],
    source: str,
    tokenizer_mode: Literal["embedded", "policy"],
) -> None:
    """Validate a model file inventory for its tokenizer contract."""
    if tokenizer_mode == "embedded":
        validate_portable_hf_model_files(names, source)
    else:
        validate_hf_model_weights(names, source)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_identity(
    model_id: str | None,
    revision: str | None,
    files: tuple[ModelManifestFile, ...],
    tokenizer_mode: Literal["embedded", "policy"] = "embedded",
) -> str:
    value = {
        "files": [entry.model_dump(mode="json") for entry in files],
        "format_version": 1,
        "model_id": model_id,
        "revision": revision,
    }
    # Preserve identities written before draft manifests declared their shared
    # policy-tokenizer contract. The non-default mode remains identity-bound.
    if tokenizer_mode != "embedded":
        value["tokenizer_mode"] = tokenizer_mode
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


SAFETENSORS_LENGTH_PREFIX_BYTES = 8


def read_safetensors_header(source: BinaryIO, source_name: str) -> tuple[bytes, tuple[str, ...]]:
    """Consume and validate one safetensors header, returning its bytes and tensor keys."""

    def read_exact(size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining:
            chunk = source.read(min(remaining, _HEADER_READ_CHUNK_BYTES))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    prefix = read_exact(SAFETENSORS_LENGTH_PREFIX_BYTES)
    if len(prefix) != SAFETENSORS_LENGTH_PREFIX_BYTES:
        raise ValueError(f"Truncated safetensors header: {source_name}")
    header_size = struct.unpack("<Q", prefix)[0]
    if header_size > MAX_SAFETENSORS_HEADER_BYTES:
        raise ValueError(f"Safetensors header exceeds {MAX_SAFETENSORS_HEADER_BYTES} bytes: {source_name}")
    header_bytes = read_exact(header_size)
    if len(header_bytes) != header_size:
        raise ValueError(f"Truncated safetensors header: {source_name}")
    header = json.loads(header_bytes)
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header: {source_name}")
    return prefix + header_bytes, tuple(sorted(key for key in header if key != "__metadata__"))


def build_safetensors_weight_index(
    shards: dict[str, tuple[int, tuple[str, ...]]],
    source: str,
    *,
    existing: bytes | None = None,
) -> bytes:
    """Build or validate a Transformers safetensors index from shard headers."""
    if not shards:
        raise ValueError(f"Hugging Face mirror requires safetensors weights: {source}")
    weight_map: dict[str, str] = {}
    for shard_name, (_size, keys) in sorted(shards.items()):
        for key in keys:
            if key in weight_map:
                raise ValueError(f"Duplicate tensor {key!r} in {source}")
            weight_map[key] = shard_name
    if not weight_map:
        raise ValueError(f"Hugging Face mirror has no indexed tensors: {source}")
    if existing is not None:
        parsed = json.loads(existing)
        if parsed.get("weight_map") != weight_map:
            raise ValueError(f"Safetensors weight index does not match its shards: {source}")
        return existing
    index = {
        "metadata": {"total_size": sum(size for size, _keys in shards.values())},
        "weight_map": weight_map,
    }
    return (json.dumps(index, indent=2, sort_keys=True) + "\n").encode()


def build_model_manifest(
    files: tuple[ModelManifestFile, ...],
    model_id: str | None,
    revision: str | None,
    *,
    tokenizer_mode: Literal["embedded", "policy"] = "embedded",
) -> ModelManifest:
    """Validate manifest entries and bind them to an immutable identity."""
    return ModelManifest(
        model_id=model_id,
        revision=revision,
        tokenizer_mode=tokenizer_mode,
        identity=_manifest_identity(model_id, revision, files, tokenizer_mode),
        files=files,
    )


def _ensure_weight_index(snapshot: Path) -> None:
    shards = sorted(snapshot.glob("*.safetensors"))
    shard_headers = {}
    for shard in shards:
        with shard.open("rb") as source:
            _header, keys = read_safetensors_header(source, str(shard))
        shard_headers[shard.name] = (shard.stat().st_size, keys)
    index_path = snapshot / HF_WEIGHT_INDEX_FILENAME
    index_bytes = build_safetensors_weight_index(
        shard_headers,
        str(index_path),
        existing=index_path.read_bytes() if index_path.is_file() else None,
    )
    if not index_path.is_file():
        index_path.write_bytes(index_bytes)


def snapshot_model_manifest(
    snapshot: Path,
    model_id: str | None,
    revision: str | None,
    *,
    tokenizer_mode: Literal["embedded", "policy"] = "embedded",
) -> ModelManifest:
    normalize_fast_tokenizer_metadata(snapshot)
    _ensure_weight_index(snapshot)
    paths = sorted(
        path for path in snapshot.rglob("*") if path.is_file() and path.relative_to(snapshot).parts[0] != ".cache"
    )
    names = {path.relative_to(snapshot).as_posix() for path in paths}
    validate_model_file_names(names, str(snapshot), tokenizer_mode)
    files = tuple(
        ModelManifestFile(
            path=path.relative_to(snapshot).as_posix(),
            size=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for path in paths
    )
    return build_model_manifest(
        files,
        model_id,
        revision,
        tokenizer_mode=tokenizer_mode,
    )


def write_local_model_manifest(
    model_dir: str | Path,
    *,
    model_id: str | None = None,
    revision: str | None = None,
) -> ModelManifest:
    """Validate a completed local HF export and write its immutable manifest."""
    root = Path(model_dir)
    marker = root / MODEL_MANIFEST_FILENAME
    if marker.exists():
        marker.unlink()
    manifest = snapshot_model_manifest(root, model_id, revision)
    marker.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")
    return manifest
