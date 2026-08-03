# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load and run a pinned FastTransformer embedding bundle."""

import hashlib
import io
import math
from dataclasses import asdict, dataclass
from typing import Literal, Self

import equinox as eqx
import jax.random as jr
import numpy as np
import pyarrow as pa
from luxical.tokenization import ArrowTokenizer
from pydantic import BaseModel, Field, model_validator
from rigging.filesystem import StoragePath

from experiments.datakit.cluster.quality.fast_transformer.embedding import (
    pack_remapped_windows,
    predict_embeddings,
)
from experiments.datakit.cluster.quality.fast_transformer.model import (
    ACCELERATOR_COMPUTE_DTYPE_NAME,
    CPU_COMPUTE_DTYPE_NAME,
    FastEmbeddingTransformer,
    FastTransformerConfig,
)

MANIFEST_FILENAME = "manifest.json"
SHA256_PATTERN = r"^[0-9a-f]{64}$"


class FastEmbeddingRuntimeManifest(BaseModel):
    """Pinned files and inference settings for one embedding model."""

    version: Literal["runtime-v1"] = "runtime-v1"
    model_filename: str = Field(min_length=1)
    model_sha256: str = Field(pattern=SHA256_PATTERN)
    token_remap_filename: str = Field(min_length=1)
    token_remap_sha256: str = Field(pattern=SHA256_PATTERN)
    tokenizer_filename: str = Field(min_length=1)
    tokenizer_sha256: str = Field(pattern=SHA256_PATTERN)
    tokenizer_name: str = Field(min_length=1)
    raw_vocab_size: int = Field(ge=1)
    config: FastTransformerConfig
    output_dimension: int = Field(ge=1)
    characters_per_region: int = Field(ge=1)
    cpu_compute_dtype: str = Field(min_length=1)
    accelerator_compute_dtype: str = Field(min_length=1)
    training_report_url: str = Field(min_length=1)
    training_report_sha256: str = Field(pattern=SHA256_PATTERN)


class FastEmbeddingBundleManifest(FastEmbeddingRuntimeManifest):
    """Pinned runtime files and release evidence for one embedding model."""

    version: Literal["v2"] = "v2"
    evaluation_report_url: str = Field(min_length=1)
    evaluation_report_sha256: str = Field(pattern=SHA256_PATTERN)
    speed_report_url: str = Field(min_length=1)
    speed_report_sha256: str = Field(pattern=SHA256_PATTERN)
    accelerator_speed_report_url: str = Field(min_length=1)
    accelerator_speed_report_sha256: str = Field(pattern=SHA256_PATTERN)
    blind_review_report_url: str = Field(min_length=1)
    blind_review_report_sha256: str = Field(pattern=SHA256_PATTERN)
    blind_review_package_url: str = Field(min_length=1)
    blind_review_package_sha256: str = Field(pattern=SHA256_PATTERN)
    quantization_range: float = Field(gt=0)
    quantization_scale: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_quantization_scale(self) -> Self:
        """Require the 255-level symmetric int8 scale."""
        expected_scale = self.quantization_range / 127
        if not math.isclose(self.quantization_scale, expected_scale, rel_tol=0, abs_tol=1e-12):
            raise ValueError("The quantization scale does not match the quantization range")
        return self


def payload_sha256(payload: bytes) -> str:
    """Return the SHA-256 digest of one payload."""
    return hashlib.sha256(payload).hexdigest()


def document_view(text: str, characters_per_region: int) -> str:
    """Return a bounded head, middle, and tail view of one document."""
    if characters_per_region < 1:
        raise ValueError("The characters-per-region value must be positive")
    if len(text) <= 3 * characters_per_region:
        return text
    middle_start = max(0, len(text) // 2 - characters_per_region // 2)
    return "\n".join(
        (
            text[:characters_per_region],
            text[middle_start : middle_start + characters_per_region],
            text[-characters_per_region:],
        )
    )


def verified_payload(root: StoragePath, filename: str, expected_sha256: str) -> bytes:
    """Read one bundle file and verify its digest."""
    payload = (root / filename).read_bytes()
    actual_sha256 = payload_sha256(payload)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"The digest for {filename} is {actual_sha256}, expected {expected_sha256}")
    return payload


@dataclass(frozen=True)
class FastEmbeddingModel:
    """A FastTransformer, tokenizer, and token remap for text embeddings."""

    model: FastEmbeddingTransformer
    raw_to_compact: np.ndarray
    tokenizer: ArrowTokenizer
    manifest: FastEmbeddingRuntimeManifest

    @classmethod
    def load_runtime(cls, bundle_root: str, expected_manifest_sha256: str) -> "FastEmbeddingModel":
        """Load a pinned runtime bundle before its release evidence exists."""
        root = StoragePath(bundle_root)
        manifest_payload = verified_payload(root, MANIFEST_FILENAME, expected_manifest_sha256)
        manifest = FastEmbeddingRuntimeManifest.model_validate_json(manifest_payload)
        return cls._from_manifest(root, manifest)

    @classmethod
    def load(cls, bundle_root: str, expected_manifest_sha256: str) -> "FastEmbeddingModel":
        """Load a bundle only when its manifest and file digests match."""
        root = StoragePath(bundle_root)
        manifest_payload = verified_payload(root, MANIFEST_FILENAME, expected_manifest_sha256)
        manifest = FastEmbeddingBundleManifest.model_validate_json(manifest_payload)
        return cls._from_manifest(root, manifest)

    @classmethod
    def _from_manifest(
        cls,
        root: StoragePath,
        manifest: FastEmbeddingRuntimeManifest,
    ) -> "FastEmbeddingModel":
        """Load one model from an already validated runtime manifest."""
        if manifest.cpu_compute_dtype != CPU_COMPUTE_DTYPE_NAME:
            raise ValueError("The CPU compute data type does not match the bundle loader")
        if manifest.accelerator_compute_dtype != ACCELERATOR_COMPUTE_DTYPE_NAME:
            raise ValueError("The accelerator compute data type does not match the bundle loader")
        model_payload = verified_payload(root, manifest.model_filename, manifest.model_sha256)
        remap_payload = verified_payload(root, manifest.token_remap_filename, manifest.token_remap_sha256)
        tokenizer_payload = verified_payload(root, manifest.tokenizer_filename, manifest.tokenizer_sha256)

        raw_to_compact = np.load(io.BytesIO(remap_payload), allow_pickle=False)
        if raw_to_compact.dtype != np.int32 or raw_to_compact.shape != (manifest.raw_vocab_size,):
            raise ValueError("The token remap does not match the manifest")
        if int(raw_to_compact.min()) < 1 or int(raw_to_compact.max()) >= manifest.config.vocab_size:
            raise ValueError("The token remap contains an ID outside the model vocabulary")
        template = FastEmbeddingTransformer(
            manifest.config,
            output_dim=manifest.output_dimension,
            key=jr.PRNGKey(0),
        )
        model = eqx.tree_deserialise_leaves(io.BytesIO(model_payload), template)
        tokenizer = ArrowTokenizer(tokenizer_payload.decode())
        return cls(model=model, raw_to_compact=raw_to_compact, tokenizer=tokenizer, manifest=manifest)

    def __call__(self, texts: list[str], batch_size: int = 4_096) -> np.ndarray:
        """Return one normalized embedding for each input document."""
        if batch_size < 1:
            raise ValueError("The batch size must be positive")
        if not texts:
            return np.empty((0, self.manifest.output_dimension), dtype=np.float32)
        outputs = []
        for start in range(0, len(texts), batch_size):
            views = [
                document_view(text, self.manifest.characters_per_region) for text in texts[start : start + batch_size]
            ]
            token_lists = self.tokenizer.tokenize(pa.array(views), add_special_tokens=False).to_pylist()
            raw_windows = [[row] for row in token_lists]
            ids = pack_remapped_windows(
                raw_windows,
                self.raw_to_compact,
                self.manifest.config.max_tokens,
                self.manifest.config.max_tokens,
            )
            outputs.append(predict_embeddings(self.model, ids, batch_size=batch_size))
        vectors = np.concatenate(outputs)
        if not np.isfinite(vectors).all():
            raise ValueError("The FastTransformer returned a non-finite embedding")
        return vectors

    def metadata(self) -> dict[str, object]:
        """Return the pinned runtime identity and input settings."""
        return {
            "tokenizer": self.manifest.tokenizer_name,
            "output_dimension": self.manifest.output_dimension,
            "tokens_per_document_window": self.manifest.config.max_tokens,
            "windows_per_document": 1,
            "source_windows_per_document": 3,
            "characters_per_source_window": self.manifest.characters_per_region,
            "cpu_compute_dtype": self.manifest.cpu_compute_dtype,
            "accelerator_compute_dtype": self.manifest.accelerator_compute_dtype,
            "config": asdict(self.manifest.config),
            "final_model_sha256": self.manifest.model_sha256,
        }
