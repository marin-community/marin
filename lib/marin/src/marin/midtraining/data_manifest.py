# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned JSON data-cache manifests consumed by the training launch.

Training never walks live ``StepSpec`` graphs. ``data-manifest build`` runs
the existing Marin/Zephyr cache code, verifies the produced caches, and
writes a content-addressed JSON file. ``launch`` reads the manifest, checks
fingerprints, samples BOS tokens, and refuses on any mismatch.

Two physical layers:

- Approved manifest:
  ``gs://marin-<region>/midtrain-manifests/data/<mix>/<fingerprint>.json``
- In-repo pointer:
  ``experiments/midtraining_data_manifests/<mix>.json``

The pointer file is small and trivially diff-reviewable; the approved
manifest is the data of record.
"""

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from marin.midtraining.tokenizers import TokenizerRef

SCHEMA_VERSION = 1
MANIFEST_PREFIX = "midtrain-manifests/data"
MAX_LOCAL_MANIFEST_BYTES = 1_000_000

_MIX_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_./-]*$")


@dataclass(frozen=True)
class DataCacheComponent:
    """One tokenized cache backing a mixture component.

    Args:
        logical_name: Stable name from the mix registry, e.g.
            ``"nemotron_cc_math_v1/4plus"``.
        cache_path: Resolved ``gs://...`` directory of the cache.
        cache_digest: Content fingerprint over ``(cache_path, length,
            tokenizer ref, BOS sample, sampled content)``. Used to detect
            silent rebuilds of the underlying tokenized data.
        total_sequences: Number of sequences in the cache (optional metric).
        total_tokens: Total token count (optional metric).
        bos_sample: First few token ids from a sampled sequence; must start
            with the tokenizer BOS.
        validation_fingerprint: Optional digest of the held-out validation
            partition, when one was carved out at build time.
    """

    logical_name: str
    cache_path: str
    cache_digest: str
    tokenizer: TokenizerRef
    total_sequences: int | None = None
    total_tokens: int | None = None
    bos_sample: tuple[int, ...] = field(default_factory=tuple)
    validation_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not self.logical_name:
            raise ValueError("DataCacheComponent.logical_name must be non-empty")
        if not self.cache_path.startswith("gs://"):
            raise ValueError(f"DataCacheComponent.cache_path must be a gs:// URI, got {self.cache_path!r}")
        if not self.cache_digest:
            raise ValueError("DataCacheComponent.cache_digest must be non-empty")
        if self.total_sequences is not None and self.total_sequences <= 0:
            raise ValueError(f"total_sequences must be positive, got {self.total_sequences!r}")
        if self.total_tokens is not None and self.total_tokens <= 0:
            raise ValueError(f"total_tokens must be positive, got {self.total_tokens!r}")
        # BOS sample must start with the tokenizer's BOS to catch the missing-BOS
        # class of bugs that bit us with the us-central1 4plus cache in 2026-04.
        if self.bos_sample and self.bos_sample[0] != self.tokenizer.bos_token_id:
            raise ValueError(
                f"BOS sample for {self.logical_name!r} does not start with "
                f"tokenizer BOS {self.tokenizer.bos_token_id}: {list(self.bos_sample[:4])!r}"
            )


@dataclass(frozen=True)
class DataCacheManifest:
    """Pinned mixture of tokenized caches for one midtraining launch.

    Args:
        mix_name: Stable identifier from ``experiments.midtraining_mixes``.
        mix_spec_digest: Hash over logical component names, weights, val
            carve-out policy, seq_len, tokenizer ref, and split policy.
        region: GCP region of all listed cache paths. Must equal the
            output region of the launch.
        components: Tokenized caches for the mixture.
        weights: Mapping from logical component name to mixing weight;
            weights sum to 1.0.
        seq_len: Sequence length the caches were built for.
        shuffle_before_trainval_split: Mirror of the build-time flag. The
            launcher refuses ``False`` unless safety overrides are set.
    """

    mix_name: str
    mix_spec_digest: str
    region: str
    components: tuple[DataCacheComponent, ...]
    weights: dict[str, float]
    seq_len: int
    shuffle_before_trainval_split: bool = True
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not _MIX_NAME_PATTERN.match(self.mix_name):
            raise ValueError(f"DataCacheManifest.mix_name {self.mix_name!r} must match {_MIX_NAME_PATTERN.pattern}")
        if not self.mix_spec_digest:
            raise ValueError("DataCacheManifest.mix_spec_digest must be non-empty")
        if not self.region:
            raise ValueError("DataCacheManifest.region must be non-empty")
        if not self.components:
            raise ValueError("DataCacheManifest.components must be non-empty")
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len!r}")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported manifest schema_version={self.schema_version!r}; expected {SCHEMA_VERSION}")

        component_names = {c.logical_name for c in self.components}
        weight_names = set(self.weights)
        if component_names != weight_names:
            missing = sorted(component_names - weight_names)
            extra = sorted(weight_names - component_names)
            raise ValueError(f"DataCacheManifest weights do not match components: missing={missing}, extra={extra}")

        weight_sum = sum(self.weights.values())
        if not (0.999 < weight_sum < 1.001):
            raise ValueError(f"DataCacheManifest weights must sum to ~1.0, got {weight_sum!r}")
        for name, weight in self.weights.items():
            if weight < 0:
                raise ValueError(f"DataCacheManifest weight for {name!r} is negative: {weight!r}")

        # All cache paths must live in the manifest's declared region. The
        # parent-retry incident in 2026-04 happened because executor identity
        # silently consumed paths from a different region; here we make the
        # mismatch an explicit failure at load time.
        region_prefix = f"gs://marin-{self.region}/"
        for component in self.components:
            if not component.cache_path.startswith(region_prefix):
                raise ValueError(
                    f"DataCacheComponent {component.logical_name!r} cache_path {component.cache_path!r} "
                    f"is not in declared region {self.region!r}"
                )
            if not component.bos_sample:
                raise ValueError(
                    f"DataCacheComponent {component.logical_name!r} has no bos_sample; "
                    "rebuild the manifest with a non-empty BOS sample to defend against the missing-BOS bug."
                )

        # Tokenizer agreement across all components — the BOS sample is
        # already validated against each component's own tokenizer ref, but
        # the mixture itself must be tokenizer-uniform.
        canonical = self.components[0].tokenizer
        for component in self.components[1:]:
            if component.tokenizer.key != canonical.key:
                raise ValueError(
                    f"Mixed tokenizers in manifest {self.mix_name!r}: " f"{canonical.key} vs {component.tokenizer.key}"
                )

    @property
    def tokenizer(self) -> TokenizerRef:
        """Shared tokenizer for every component in the mixture."""
        return self.components[0].tokenizer

    @property
    def total_tokens(self) -> int | None:
        if any(c.total_tokens is None for c in self.components):
            return None
        return sum(c.total_tokens or 0 for c in self.components)

    def fingerprint(self) -> str:
        """Stable sha256 over the manifest's normalized JSON form."""
        return _stable_sha256(_to_jsonable(self))


@dataclass(frozen=True)
class DataManifestPointer:
    """In-repo pointer file referencing an approved GCS manifest.

    The pointer is the diff-reviewable record; the approved manifest itself
    is content-addressed in GCS so silent overwrites cannot succeed.
    """

    mix_name: str
    approved_manifest_uri: str
    approved_at: str

    def __post_init__(self) -> None:
        if not _MIX_NAME_PATTERN.match(self.mix_name):
            raise ValueError(f"DataManifestPointer.mix_name {self.mix_name!r} is not a canonical mix name")
        if not self.approved_manifest_uri.startswith("gs://"):
            raise ValueError(
                f"DataManifestPointer.approved_manifest_uri must be a gs:// URI, got {self.approved_manifest_uri!r}"
            )
        if MANIFEST_PREFIX not in self.approved_manifest_uri:
            raise ValueError(
                f"approved_manifest_uri must live under '/{MANIFEST_PREFIX}/', got {self.approved_manifest_uri!r}"
            )


def load_data_manifest_pointer(pointer_path: str | os.PathLike[str]) -> DataManifestPointer:
    """Load a local pointer JSON. ``pointer_path`` is a filesystem path."""
    raw = Path(pointer_path).read_text(encoding="utf-8")
    if len(raw) > MAX_LOCAL_MANIFEST_BYTES:
        raise ValueError(f"Pointer file {pointer_path!s} is unexpectedly large ({len(raw)} bytes)")
    data = json.loads(raw)
    return DataManifestPointer(
        mix_name=data["mix_name"],
        approved_manifest_uri=data["approved_manifest_uri"],
        approved_at=data["approved_at"],
    )


def load_data_manifest(manifest_path: str | os.PathLike[str]) -> DataCacheManifest:
    """Load a manifest JSON from a local path or via GCS-aware fsspec.

    Accepts ``gs://`` URIs (uses ``gcsfs`` if available) and local paths.
    """
    raw = _read_text(str(manifest_path))
    data = json.loads(raw)
    return _manifest_from_dict(data)


def dump_data_manifest(manifest: DataCacheManifest) -> str:
    """Serialize a manifest to canonical JSON (sorted keys, 2-space indent)."""
    return json.dumps(_to_jsonable(manifest), indent=2, sort_keys=True)


def _read_text(uri: str) -> str:
    if uri.startswith("gs://"):
        import fsspec  # imported lazily so non-GCS environments still load the module

        with fsspec.open(uri, "r", encoding="utf-8") as f:
            return f.read()
    return Path(uri).read_text(encoding="utf-8")


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (DataCacheManifest, DataCacheComponent, TokenizerRef, DataManifestPointer)):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _manifest_from_dict(data: dict[str, Any]) -> DataCacheManifest:
    if data.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported manifest schema_version={data.get('schema_version')!r}")
    components = tuple(
        DataCacheComponent(
            logical_name=c["logical_name"],
            cache_path=c["cache_path"],
            cache_digest=c["cache_digest"],
            tokenizer=TokenizerRef(**c["tokenizer"]),
            total_sequences=c.get("total_sequences"),
            total_tokens=c.get("total_tokens"),
            bos_sample=tuple(c.get("bos_sample") or ()),
            validation_fingerprint=c.get("validation_fingerprint"),
        )
        for c in data["components"]
    )
    return DataCacheManifest(
        mix_name=data["mix_name"],
        mix_spec_digest=data["mix_spec_digest"],
        region=data["region"],
        components=components,
        weights=dict(data["weights"]),
        seq_len=int(data["seq_len"]),
        shuffle_before_trainval_split=bool(data.get("shuffle_before_trainval_split", True)),
        schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
    )


def _stable_sha256(payload: Any) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(body).hexdigest()


def approved_manifest_uri(*, mix_name: str, region: str, fingerprint: str) -> str:
    """Canonical path for an approved manifest under its content fingerprint."""
    # The fingerprint may already include a ``sha256:`` prefix from
    # :func:`DataCacheManifest.fingerprint`; strip it so the on-disk filename
    # stays plain hex.
    digest = fingerprint.removeprefix("sha256:")
    return f"gs://marin-{region}/{MANIFEST_PREFIX}/{mix_name}/{digest}.json"
