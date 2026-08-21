# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze the H4 multi-target rollout rerun (ATOM-031).

The v6 rollouts recorded only `paloma_programming_languages`, which is monotone in the StarCoder weight,
so their argmin is always the boundary and any monotone utility recovers it for free. This release reruns
exactly the same rollout rows with every configured target recorded, so that a trade-off -- and therefore
an interior optimum a utility can be right or wrong about -- can appear at all.

Nothing about the design changes. The rollout manifests are copied verbatim from the v6 release, so the
same parents, q grid, update horizon, readout steps, and frozen source streams are used. What changes is
the runtime that measures them and the root the results are written to, which is why this is a separate
release rather than an edit to v6: the v6 release is still the parent of the in-flight mechanism-repair
work and must keep hashing to the runtime it was frozen against.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    freeze_starcoder_wsd80_gradient_probe_20260816 as parent,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = Path(__file__).with_name("reference_outputs") / "starcoder_wsd80_h4_macro_rollout_v1_20260819"
RUNTIME_PATH = REPO_ROOT / "experiments/domain_phase_mix/starcoder_wsd80_h4_macro_rollout.py"
ANALYZER_PATH = (
    REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/"
    "analyze_starcoder_wsd80_h4_macro_rollout_20260819.py"
)
PREREGISTRATION_PATH = (
    REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/"
    "reference_outputs/atom031_h4_macro_preregistration_20260819.md"
)
PARENT_RELEASE_PATH = parent.OUTPUT_DIR / "release.json"
CANARY_MANIFEST_PATH = OUTPUT_DIR / "canary_rollout_manifest.csv"
FULL_MANIFEST_PATH = OUTPUT_DIR / "full_rollout_manifest.csv"
RELEASE_PATH = OUTPUT_DIR / "release.json"

MARIN_PREFIX = parent.MARIN_PREFIX
RESULT_ROOT = (
    "gs://marin-us-central1/analysis/pinlin_calvin_xu/data_mixture/starcoder_wsd80_h4_macro_rollout_v1_20260819"
)
RELEASE_VERSION = "2026-08-19-h4-macro-rollout-v1"
# The preregistered primary objective. Equal weights, matching how every macro in this programme is
# defined; recorded here so the release hash covers the objective and it cannot be chosen after the fact.
PRIMARY_OBJECTIVE = {"weighting": "equal", "targets": sorted(parent.TARGET_COMPONENTS)}
CANARY_MAX_CONCURRENT = 2
FULL_MAX_CONCURRENT = 14

canonical_json = parent.canonical_json
canonical_sha256 = parent.canonical_sha256
file_sha256 = parent.file_sha256


def _write_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        if path.read_bytes() != payload:
            raise RuntimeError(f"Frozen release artifact already exists with different content: {path}") from error


def _copy_parent_manifest(name: str, destination: Path) -> str:
    """Copy a v6 rollout manifest verbatim. Identical rows are the point: only the measurement changes."""
    payload = (parent.OUTPUT_DIR / name).read_bytes()
    _write_create_only(destination, payload)
    return file_sha256(destination)


def freeze() -> dict[str, Any]:
    parent_release = json.loads(PARENT_RELEASE_PATH.read_text())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifests = {
        "canary_rollout": {
            "path": str(CANARY_MANIFEST_PATH.relative_to(REPO_ROOT)),
            "sha256": _copy_parent_manifest("canary_rollout_manifest.csv", CANARY_MANIFEST_PATH),
        },
        "full_rollout": {
            "path": str(FULL_MANIFEST_PATH.relative_to(REPO_ROOT)),
            "sha256": _copy_parent_manifest("full_rollout_manifest.csv", FULL_MANIFEST_PATH),
        },
    }
    implementation = (Path(__file__).resolve(), RUNTIME_PATH, ANALYZER_PATH)
    release = {
        "release_version": RELEASE_VERSION,
        "release_sha256": "",
        "result_root": RESULT_ROOT,
        "marin_prefix": MARIN_PREFIX,
        "primary_objective": PRIMARY_OBJECTIVE,
        "preregistration_sha256": file_sha256(PREREGISTRATION_PATH),
        "parent_release_version": parent_release["release_version"],
        "parent_release_sha256": parent_release["release_sha256"],
        "parent_release_file_sha256": file_sha256(PARENT_RELEASE_PATH),
        # Carried forward unchanged: the rerun measures the same targets on the same frozen sequence sets.
        "target_sampling_contract": parent_release["target_sampling_contract"],
        "manifests": manifests,
        "implementation_files": {str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in implementation},
        "parent_implementation_files": parent_release["implementation_files"],
        "endpoint_metrics_read": False,
        "scientific_status": "preregistered_multi_target_rollout_rerun",
    }
    release["release_sha256"] = canonical_sha256({**release, "release_sha256": ""})
    _write_create_only(RELEASE_PATH, (json.dumps(release, indent=2, sort_keys=True) + "\n").encode())
    return release


def main() -> None:
    print(json.dumps(freeze(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
