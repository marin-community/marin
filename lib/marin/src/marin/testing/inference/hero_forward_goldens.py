# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral loading and comparison for Hero forward golden bundles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

FORMAT_NAME = "marin-hero-forward-goldens"
FORMAT_VERSION = 1
ARRAYS_FILENAME = "arrays.npz"
MANIFEST_FILENAME = "manifest.json"

INPUT_ARRAYS = (
    "tokens",
    "token_validity",
    "segment_ids",
    "positions",
    "valid_lengths",
    "score_mask",
)
ALIGNMENT_ARRAYS = (
    "score_case_indices",
    "prediction_positions",
    "target_token_ids",
    "full_logit_case_indices",
    "full_logit_prediction_positions",
)
VALUE_ARRAYS = (
    "target_logprobs",
    "top_token_ids",
    "top_logprobs",
    "full_logits",
    "route_expert_ids",
    "route_combine_weights",
    "route_cutoff_gaps",
)
REQUIRED_ARRAYS = (*INPUT_ARRAYS, *ALIGNMENT_ARRAYS, *VALUE_ARRAYS)
REQUIRED_OBSERVATIONS = (*ALIGNMENT_ARRAYS, *VALUE_ARRAYS)


@dataclass(frozen=True)
class ComparisonTolerances:
    """Explicit absolute bounds for values whose equality is numerical."""

    target_logprob: float
    top_logprob: float
    full_logit: float
    route_combine_weight: float
    route_cutoff_gap: float

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} tolerance must be finite and non-negative")


@dataclass(frozen=True)
class ComparisonIssue:
    kind: Literal["missing", "alignment", "numerical", "routing"]
    field: str
    detail: str
    count: int | None = None
    well_separated_count: int | None = None
    max_abs_error: float | None = None


@dataclass(frozen=True)
class ComparisonReport:
    issues: tuple[ComparisonIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    def raise_for_errors(self) -> None:
        if self.issues:
            details = "\n".join(f"[{issue.kind}] {issue.field}: {issue.detail}" for issue in self.issues)
            raise AssertionError(f"Hero forward observations differ from the golden bundle:\n{details}")


@dataclass(frozen=True)
class GoldenBundle:
    root: Path
    manifest: dict
    arrays: Mapping[str, np.ndarray]

    @classmethod
    def load(cls, root: str | Path) -> GoldenBundle:
        root = Path(root)
        manifest = json.loads((root / MANIFEST_FILENAME).read_text())
        _validate_manifest(manifest)
        arrays_path = root / ARRAYS_FILENAME
        expected = manifest["files"][ARRAYS_FILENAME]
        actual_size = arrays_path.stat().st_size
        if actual_size != expected["bytes"]:
            raise ValueError(f"{ARRAYS_FILENAME} size is {actual_size}, expected {expected['bytes']}")
        actual_digest = _sha256(arrays_path)
        if actual_digest != expected["sha256"]:
            raise ValueError(f"{ARRAYS_FILENAME} checksum is {actual_digest}, expected {expected['sha256']}")
        with np.load(arrays_path, allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
        _validate_arrays(manifest, arrays)
        return cls(root=root, manifest=manifest, arrays=arrays)


def write_bundle(root: str | Path, manifest: Mapping[str, object], arrays: Mapping[str, np.ndarray]) -> None:
    """Write a new local bundle. The caller must publish it to a never-overwritten path."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    for name, value in normalized.items():
        if value.dtype.hasobject:
            raise ValueError(f"{name} uses an object dtype, which is not portable")
    arrays_path = root / ARRAYS_FILENAME
    np.savez_compressed(arrays_path, **normalized)
    complete_manifest = dict(manifest)
    complete_manifest.update(
        {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "files": {
                ARRAYS_FILENAME: {
                    "bytes": arrays_path.stat().st_size,
                    "sha256": _sha256(arrays_path),
                }
            },
        }
    )
    _validate_manifest(complete_manifest)
    _validate_arrays(complete_manifest, normalized)
    (root / MANIFEST_FILENAME).write_text(json.dumps(complete_manifest, indent=2, sort_keys=True) + "\n")


def load_observations(path: str | Path) -> dict[str, np.ndarray]:
    """Load backend observations from one pickle-free NumPy archive."""
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def compare_observations(
    golden: GoldenBundle,
    observations: Mapping[str, np.ndarray],
    tolerances: ComparisonTolerances,
) -> ComparisonReport:
    """Compare aligned scores, logits, and routes without importing a model backend."""
    observed = {name: np.asarray(value) for name, value in observations.items()}
    issues: list[ComparisonIssue] = []
    for name in REQUIRED_OBSERVATIONS:
        if name not in observed:
            issues.append(ComparisonIssue("missing", name, "required observation is absent"))

    for name in ALIGNMENT_ARRAYS:
        if name not in observed:
            continue
        expected = golden.arrays[name]
        actual = observed[name]
        if actual.shape != expected.shape:
            issues.append(ComparisonIssue("alignment", name, f"shape {actual.shape} != {expected.shape}"))
        elif not np.array_equal(actual, expected):
            changed = int(np.count_nonzero(actual != expected))
            issues.append(ComparisonIssue("alignment", name, f"{changed} aligned entries differ"))

    _compare_numeric(issues, golden, observed, "target_logprobs", tolerances.target_logprob)
    _compare_exact_values(issues, golden, observed, "top_token_ids")
    _compare_numeric(issues, golden, observed, "top_logprobs", tolerances.top_logprob)
    _compare_numeric(issues, golden, observed, "full_logits", tolerances.full_logit)

    if "route_expert_ids" in observed:
        expected_ids = golden.arrays["route_expert_ids"]
        actual_ids = observed["route_expert_ids"]
        if actual_ids.shape != expected_ids.shape:
            issues.append(
                ComparisonIssue("routing", "route_expert_ids", f"shape {actual_ids.shape} != {expected_ids.shape}")
            )
        else:
            changed = actual_ids != expected_ids
            if np.any(changed):
                changed_tokens = np.any(changed, axis=-1)
                valid = golden.arrays["token_validity"][None, :, :]
                changed_valid = changed_tokens & valid
                gaps = golden.arrays["route_cutoff_gaps"]
                well_separated = changed_valid & (gaps > tolerances.route_cutoff_gap)
                issues.append(
                    ComparisonIssue(
                        "routing",
                        "route_expert_ids",
                        f"{int(changed_valid.sum())} valid layer-token routes changed; "
                        f"{int(well_separated.sum())} have a golden cutoff gap above "
                        f"{tolerances.route_cutoff_gap:g}. Small gaps are reported, not excused.",
                        count=int(changed_valid.sum()),
                        well_separated_count=int(well_separated.sum()),
                    )
                )
    _compare_numeric(
        issues,
        golden,
        observed,
        "route_combine_weights",
        tolerances.route_combine_weight,
        kind="routing",
    )
    _compare_numeric(
        issues,
        golden,
        observed,
        "route_cutoff_gaps",
        tolerances.route_cutoff_gap,
        kind="routing",
    )
    return ComparisonReport(tuple(issues))


def _compare_exact_values(
    issues: list[ComparisonIssue],
    golden: GoldenBundle,
    observed: Mapping[str, np.ndarray],
    name: str,
) -> None:
    if name not in observed:
        return
    expected = golden.arrays[name]
    actual = observed[name]
    if actual.shape != expected.shape:
        issues.append(ComparisonIssue("numerical", name, f"shape {actual.shape} != {expected.shape}"))
    elif not np.array_equal(actual, expected):
        count = int(np.count_nonzero(actual != expected))
        issues.append(ComparisonIssue("numerical", name, f"{count} values differ", count=count))


def _compare_numeric(
    issues: list[ComparisonIssue],
    golden: GoldenBundle,
    observed: Mapping[str, np.ndarray],
    name: str,
    tolerance: float,
    *,
    kind: Literal["numerical", "routing"] = "numerical",
) -> None:
    if name not in observed:
        return
    expected = golden.arrays[name]
    actual = observed[name]
    if actual.shape != expected.shape:
        issues.append(ComparisonIssue(kind, name, f"shape {actual.shape} != {expected.shape}"))
        return
    if not np.isfinite(actual).all():
        issues.append(ComparisonIssue(kind, name, "observation contains a non-finite value"))
        return
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    if error.size and float(error.max()) > tolerance:
        max_abs_error = float(error.max())
        count = int(np.count_nonzero(error > tolerance))
        issues.append(
            ComparisonIssue(
                kind,
                name,
                f"max absolute error {max_abs_error:.8g} exceeds {tolerance:g}; {count} values exceed the bound",
                count=count,
                max_abs_error=max_abs_error,
            )
        )


def _validate_manifest(manifest: Mapping[str, object]) -> None:
    if manifest.get("format") != FORMAT_NAME or manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported Hero golden format")
    for field in ("bundle_id", "checkpoint", "producer", "model", "tokenizer", "cases", "indexing", "files"):
        if field not in manifest:
            raise ValueError(f"Manifest is missing {field}")
    files = manifest["files"]
    if not isinstance(files, dict) or ARRAYS_FILENAME not in files:
        raise ValueError(f"Manifest does not describe {ARRAYS_FILENAME}")


def _validate_arrays(manifest: Mapping[str, object], arrays: Mapping[str, np.ndarray]) -> None:
    missing = [name for name in REQUIRED_ARRAYS if name not in arrays]
    if missing:
        raise ValueError(f"Bundle is missing arrays: {', '.join(missing)}")
    tokens = arrays["tokens"]
    if tokens.ndim != 2 or tokens.dtype != np.int32:
        raise ValueError("tokens must be int32 [case, sequence]")
    batch, sequence = tokens.shape
    validity = arrays["token_validity"]
    segment_ids = arrays["segment_ids"]
    positions = arrays["positions"]
    score_mask = arrays["score_mask"]
    if validity.shape != tokens.shape or validity.dtype != np.bool_:
        raise ValueError("token_validity must be bool with the token shape")
    if segment_ids.shape != tokens.shape or segment_ids.dtype != np.int32:
        raise ValueError("segment_ids must be int32 with the token shape")
    if positions.shape != tokens.shape or positions.dtype != np.int32:
        raise ValueError("positions must be int32 with the token shape")
    if not np.array_equal(positions, np.broadcast_to(np.arange(sequence, dtype=np.int32), tokens.shape)):
        raise ValueError("positions must use zero-based absolute sequence indexing")
    if score_mask.shape != tokens.shape or score_mask.dtype != np.bool_:
        raise ValueError("score_mask must be bool with the token shape")
    if np.any(score_mask & ~validity) or np.any(score_mask[:, 0]):
        raise ValueError("score_mask must select valid target tokens after the first token")
    valid_lengths = arrays["valid_lengths"]
    if valid_lengths.shape != (batch,) or valid_lengths.dtype != np.int32:
        raise ValueError("valid_lengths must be int32 [case]")
    if not np.array_equal(valid_lengths, validity.sum(axis=1)):
        raise ValueError("valid_lengths disagree with token_validity")
    if not np.array_equal(validity, segment_ids >= 0):
        raise ValueError("non-negative segment IDs must exactly identify valid tokens")

    target_indices = np.argwhere(score_mask)
    expected_cases = target_indices[:, 0].astype(np.int32)
    expected_predictions = (target_indices[:, 1] - 1).astype(np.int32)
    expected_targets = tokens[target_indices[:, 0], target_indices[:, 1]]
    if not np.array_equal(arrays["score_case_indices"], expected_cases):
        raise ValueError("score_case_indices disagree with score_mask")
    if not np.array_equal(arrays["prediction_positions"], expected_predictions):
        raise ValueError("prediction_positions disagree with score_mask")
    if not np.array_equal(arrays["target_token_ids"], expected_targets):
        raise ValueError("target_token_ids disagree with tokens and prediction positions")
    score_count = len(target_indices)
    if arrays["target_logprobs"].shape != (score_count,):
        raise ValueError("target_logprobs must contain one value per scored token")
    if arrays["top_token_ids"].ndim != 2 or arrays["top_token_ids"].shape[0] != score_count:
        raise ValueError("top_token_ids must be [score, rank]")
    if arrays["top_logprobs"].shape != arrays["top_token_ids"].shape:
        raise ValueError("top_logprobs must match top_token_ids")

    full_cases = arrays["full_logit_case_indices"]
    full_positions = arrays["full_logit_prediction_positions"]
    full_logits = arrays["full_logits"]
    if full_cases.ndim != 1 or full_positions.shape != full_cases.shape or full_logits.ndim != 2:
        raise ValueError("full-logit arrays have incompatible ranks")
    if full_logits.shape[0] != len(full_cases):
        raise ValueError("full_logits must contain one row per requested case and position")
    if np.any(full_cases < 0) or np.any(full_cases >= batch) or np.any(full_positions < 0):
        raise ValueError("full-logit indices are outside the input batch")
    if np.any(full_positions >= valid_lengths[full_cases]):
        raise ValueError("full-logit positions must be valid prediction positions")

    route_ids = arrays["route_expert_ids"]
    route_weights = arrays["route_combine_weights"]
    route_gaps = arrays["route_cutoff_gaps"]
    model = manifest["model"]
    if not isinstance(model, dict):
        raise ValueError("model manifest entry must be an object")
    layers = int(model["num_layers"])
    experts = int(model["num_experts"])
    routes = int(model["experts_per_token"])
    if route_ids.shape != (layers, batch, sequence, routes):
        raise ValueError("route_expert_ids must be [layer, case, token, route_slot]")
    if route_ids.dtype != np.int32 or route_weights.dtype != np.float32 or route_gaps.dtype != np.float32:
        raise ValueError("route IDs must be int32; route weights and cutoff gaps must be float32")
    if route_weights.shape != route_ids.shape or route_gaps.shape != (layers, batch, sequence):
        raise ValueError("route weights or cutoff gaps do not match route indexing")
    route_validity = np.broadcast_to(validity[None, :, :, None], route_ids.shape)
    if (
        np.any(route_ids[route_validity] < 0)
        or np.any(route_ids[route_validity] >= experts)
        or np.any(route_ids[~route_validity] != -1)
    ):
        raise ValueError("valid route expert IDs must be in range and padding must use -1")
    if np.any(route_weights[~route_validity] != 0) or np.any(route_gaps[:, ~validity] != 0):
        raise ValueError("route padding values must be zero")
    for name in ("target_logprobs", "top_logprobs", "full_logits", "route_combine_weights", "route_cutoff_gaps"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} contains a non-finite value")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
