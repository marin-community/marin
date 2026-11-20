# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned, executable-code-free inference for low-value classifier heads."""

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from filelock import FileLock
from levanter.tokenizers import MarinTokenizer, load_tokenizer
from rigging.filesystem.factory import open_url
from rigging.filesystem.storage_path import StoragePath, prefix_join

from marin.datakit.low_value.features import FEATURE_NAMES, FEATURES_VERSION


@dataclass(frozen=True)
class RegressionTree:
    left: np.ndarray
    right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    value: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        nodes = np.zeros(len(features), dtype=np.int64)
        active = self.left[nodes] >= 0
        while active.any():
            rows = np.flatnonzero(active)
            current = nodes[rows]
            goes_left = features[rows, self.feature[current]] <= self.threshold[current]
            nodes[rows] = np.where(goes_left, self.left[current], self.right[current])
            active = self.left[nodes] >= 0
        return self.value[nodes]


@dataclass(frozen=True)
class BoostedTreeHead:
    trees: tuple[RegressionTree, ...]
    intercept: float
    learning_rate: float

    def logits(self, features: np.ndarray) -> np.ndarray:
        # sklearn's tree boundary casts to float32 before comparing stored thresholds.
        values = features.astype(np.float32)
        result = np.full(len(features), self.intercept, dtype=np.float64)
        for tree in self.trees:
            result += self.learning_rate * tree.predict(values)
        return result


@dataclass(frozen=True)
class LowValueModel:
    """A structural feature contract and a small binary prediction head."""

    head: BoostedTreeHead
    window_tokens: int
    tokenizer_path: str
    tokenizer_files_sha256: tuple[tuple[str, str], ...]
    sha256: str

    @property
    def tokenizer_sha256(self) -> str:
        return dict(self.tokenizer_files_sha256)["tokenizer.json"]

    def predict(self, structural: np.ndarray) -> np.ndarray:
        """Return higher-is-low-value scores, preserving input row order."""
        if structural.ndim != 2 or structural.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"Expected structural rows with {len(FEATURE_NAMES)} features")
        if not np.isfinite(structural).all():
            raise ValueError("Non-finite classifier features")
        logits = self.head.logits(structural)
        return 1 / (1 + np.exp(-np.clip(logits, -700, 700)))


def read_pinned_bytes(path: str, sha256: str) -> bytes:
    """Return artifact bytes only when their SHA-256 matches the supplied pin."""
    with open_url(path, "rb") as stream:
        content = stream.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != sha256:
        raise ValueError(f"Artifact {path} SHA256 {digest} differs from pinned {sha256}")
    return content


def read_pinned_json(path: str, sha256: str) -> dict:
    """Read JSON only after verifying the caller's immutable artifact digest."""
    return json.loads(read_pinned_bytes(path, sha256))


def load_model(path: str, sha256: str) -> LowValueModel:
    """Load a model without pickle, Python execution, or training dependencies."""
    payload = read_pinned_json(path, sha256)
    if payload["features_version"] != FEATURES_VERSION:
        raise ValueError("Model uses a different structural feature version")
    if tuple(payload["feature_names"]) != FEATURE_NAMES or payload["embedding_dimensions"] != 0:
        raise ValueError("Model does not use the supported structural feature schema")
    if payload["kind"] != "gradient_boosting":
        raise ValueError(f"Unsupported model head {payload['kind']}")
    trees = tuple(
        RegressionTree(
            left=np.asarray(tree["left"], dtype=np.int64),
            right=np.asarray(tree["right"], dtype=np.int64),
            feature=np.asarray(tree["feature"], dtype=np.int64),
            threshold=np.asarray(tree["threshold"], dtype=np.float64),
            value=np.asarray(tree["value"], dtype=np.float64),
        )
        for tree in payload["trees"]
    )
    head = BoostedTreeHead(trees, payload["intercept"], payload["learning_rate"])
    return LowValueModel(
        head,
        payload["window_tokens"],
        prefix_join(str(StoragePath(path).parent), payload["tokenizer"]),
        tuple(sorted(payload["tokenizer_files_sha256"].items())),
        sha256,
    )


def load_model_tokenizer(model: LowValueModel) -> MarinTokenizer:
    """Verify the bundled tokenizer files before staging the tokenizer backend."""
    manifest = json.dumps(model.tokenizer_files_sha256, separators=(",", ":")).encode()
    digest = hashlib.sha256(manifest).hexdigest()
    directory = Path(tempfile.gettempdir()) / "marin-low-value-tokenizers" / digest
    contents = {}
    for name, sha256 in model.tokenizer_files_sha256:
        if Path(name).name != name:
            raise ValueError("Tokenizer file names must be basenames")
        contents[name] = read_pinned_bytes(prefix_join(model.tokenizer_path, name), sha256)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory) + ".lock"):
        directory.mkdir(exist_ok=True)
        for name, content in contents.items():
            destination = directory / name
            if (
                not destination.exists()
                or hashlib.sha256(destination.read_bytes()).hexdigest() != hashlib.sha256(content).hexdigest()
            ):
                destination.write_bytes(content)
        return load_tokenizer(str(directory))


@dataclass(frozen=True)
class OperatingPoint:
    cutoff: float
    target_false_drop: float
    calibration_documents: int
    calibration_false_drop: float

    def drop(self, scores: np.ndarray) -> np.ndarray:
        return scores >= self.cutoff


def load_operating_point(path: str, sha256: str, model_sha256: str, target_false_drop: float) -> OperatingPoint:
    """Require calibration for these model bytes and this explicit operating point."""
    payload = read_pinned_json(path, sha256)
    if payload["model_sha256"] != model_sha256:
        raise ValueError("Calibration was fitted for different model bytes")
    setting = payload["thresholds"][str(target_false_drop)]
    return OperatingPoint(
        setting["cutoff"],
        target_false_drop,
        setting["calibration"]["documents"],
        setting["calibration"]["false_drop_rate"],
    )
