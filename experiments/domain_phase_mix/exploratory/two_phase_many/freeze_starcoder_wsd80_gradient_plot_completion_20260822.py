# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze saved-checkpoint probes needed to complete the gradient-mechanism plots."""

import csv
import hashlib
import json
import subprocess
import tarfile
import tomllib
from collections import Counter
from io import BytesIO, StringIO
from itertools import pairwise
from pathlib import Path
from typing import Any

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    freeze_starcoder_wsd80_gradient_mechanism_repair_20260818 as repair,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    freeze_starcoder_wsd80_gradient_probe_20260816 as parent,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = Path(__file__).with_name("reference_outputs") / "starcoder_wsd80_gradient_plot_completion_v8_20260822"
RUNTIME_PATH = REPO_ROOT / "experiments/domain_phase_mix/starcoder_wsd80_gradient_plot_completion.py"
MATERIALIZER_PATH = (
    REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/"
    "materialize_starcoder_wsd80_gradient_plot_completion_20260822.py"
)
PLOTTER_PATH = (
    REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/"
    "plot_starcoder_wsd80_gradient_mechanism_repair_20260820.py"
)
ALL_SOURCE_MATERIALIZER_PATH = (
    REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/"
    "materialize_starcoder_wsd80_gradient_all_source_geometry_20260822.py"
)
CANARY_CONFIG_PATH = REPO_ROOT / "experiments/domain_phase_mix/launch_starcoder_wsd80_gradient_conflict.py"
FULL_CONFIG_PATH = REPO_ROOT / "experiments/domain_phase_mix/launch_starcoder_wsd80_gradient_conflict_full.py"
FULL_SOURCE_DESIGN_PATH = (
    REPO_ROOT / "experiments/domain_phase_mix/starcoder_wsd80_dense_support_surface_design_20260808.json"
)
EXECUTION_IMPORT_PATHS = tuple(
    REPO_ROOT / relative_path
    for relative_path in (
        "experiments/datasets/dolma.py",
        "experiments/datasets/nemotron.py",
        "experiments/datasets/paloma.py",
        "experiments/datasets/uncheatable.py",
        "experiments/llama.py",
        "experiments/simple_train_config.py",
        "experiments/scaling_law_sweeps/completed_adamh.py",
        *parent.full.RUNTIME_SOURCE_PATHS,
    )
)
V10_RELEASE_PATH = repair.RELEASE_PATH
V10_RELEASE_FILE_SHA256 = "9a09ea7eb49fc2880e6fab8aa38128790b3604e32fbea3b0661334718ddad129"
V10_RELEASE_SHA256 = "051dc75c4ee6baa67b3df7f4ff305e4da8f83cadb5a1b3f18edf889176b00d3b"
PARENT_RELEASE_PATH = parent.OUTPUT_DIR / "release.json"
PARENT_FULL_MANIFEST_PATH = parent.OUTPUT_DIR / "full_probe_manifest.csv"
TRAJECTORY_MANIFEST_PATH = parent.DESIGN_DIR / "trajectory_manifest.csv"
ANALYSIS_CONTRACT_PATH = OUTPUT_DIR / "analysis_contract.json"
FULL_MANIFEST_PATH = OUTPUT_DIR / "full_plot_completion_manifest.csv"
COVERAGE_AUDIT_PATH = OUTPUT_DIR / "plot_coverage_audit.csv"
REPORT_PATH = OUTPUT_DIR / "report.md"
CHECKPOINT_PROVENANCE_PATH = OUTPUT_DIR / "checkpoint_object_provenance.csv"
PARENT_RESULT_PROVENANCE_PATH = OUTPUT_DIR / "parent_result_object_provenance.csv"
RELEASE_PATH = OUTPUT_DIR / "release.json"
FULL_LAUNCH_AUTHORIZATION_PATH = OUTPUT_DIR / "full_launch_authorization.json"
CC_REVIEW_PATH = REPO_ROOT / ".agents/handoffs/starcoder_wsd80_gradient_plot_completion_cc_review_v8_20260822.md"
RUNTIME_STACK_MANIFEST_PATH = OUTPUT_DIR / "historical_runtime_stack_manifest.csv"
RUNTIME_ENVIRONMENT_BASELINE_PATH = OUTPUT_DIR / "stage1_runtime_environment_baseline.json"

REFERENCE_ROOT = Path(__file__).parent / "reference_outputs"
BASE_RESULTS_DIR = REFERENCE_ROOT / "starcoder_wsd80_gradient_mechanism_repair_results_v10_20260821"
BASE_SOURCE_GEOMETRY_PATH = (
    REFERENCE_ROOT
    / "starcoder_wsd80_gradient_mechanism_all_source_geometry_20260822/source_source_geometry_all_states.csv"
)
MULTIPLICITY_AUDIT_PATH = (
    REFERENCE_ROOT / "starcoder_wsd80_gradient_mechanism_repair_review_sensitivity_20260821/"
    "recomputed_frozen_tests_two_sided.csv"
)
PLOT_OUTPUT_DIR = REFERENCE_ROOT / "starcoder_wsd80_gradient_mechanism_complete_plots_v8_20260822"
COMPLETE_TABLES_DIR = REFERENCE_ROOT / "starcoder_wsd80_gradient_plot_complete_tables_v8_20260822"
BASE_RESULT_FILES = (
    "source_source_geometry.csv",
    "target_source_utilities.csv",
    "target_source_choice_alignment.csv",
    "h2_h3_summary.csv",
    "h3_repetition_mechanism_summary.csv",
    "h5_profile_summary.csv",
)

MARIN_PREFIX = "gs://marin-us-central1"
RESULT_ROOT = (
    "gs://marin-us-central1/analysis/pinlin_calvin_xu/data_mixture/"
    "starcoder_wsd80_gradient_plot_completion_v8_20260822"
)
REMOTE_ADAPTER_CANARY_PATH = f"{RESULT_ROOT}/runtime_adapter_canary/passed.json"
RELEASE_VERSION = "2026-08-22-gradient-plot-completion-v8"
PARENT_ARTIFACT_VERSION = repair.PARENT_ARTIFACT_VERSION
SCIENTIFIC_STATUS = "post_outcome_descriptive_plot_completion_not_confirmation"

RECORDED_CLEAN_COMMIT = "7efb96842624a2e8cbab36c9a9aa6b1cb68c4922"
HISTORICAL_RUNTIME_COMMIT = "377ad16d816a1726cc97396355607594910e9f0a"
HISTORICAL_RUNTIME_PATHS = (
    ".python-version",
    "pyproject.toml",
    "uv.lock",
    "infra/pulumi/pyproject.toml",
    "infra/pulumi/src",
    "lib/ducky/pyproject.toml",
    "lib/ducky/src",
    "lib/dupekit/pyproject.toml",
    "lib/dupekit/src",
    "lib/finelog/pyproject.toml",
    "lib/finelog/src",
    "lib/finestore/pyproject.toml",
    "lib/finestore/src",
    "lib/fray/pyproject.toml",
    "lib/fray/src",
    "lib/haliax/pyproject.toml",
    "lib/haliax/src",
    "lib/iris/pyproject.toml",
    "lib/iris/src",
    "lib/levanter/pyproject.toml",
    "lib/levanter/src",
    "lib/marin/pyproject.toml",
    "lib/marin/src",
    "lib/rigging/pyproject.toml",
    "lib/rigging/src",
    "lib/zephyr/pyproject.toml",
    "lib/zephyr/src",
)
HISTORICAL_RUNTIME_EXCLUDED_PATHS = frozenset(
    {
        "lib/iris/src/iris/_build_info.py",
        "lib/marin/src/marin/inference/dashboard/src/template.html",
        "lib/marin/src/marin/inference/serve_dashboard.html",
    }
)
TASK_IMAGE = (
    "ghcr.io/marin-community/iris-task@" "sha256:c646ef8b571571edfc96c75fd9c8cc712ad286b61b33781070bdc29ab9f9a6ab"
)
EXPECTED_RUNTIME_VERSIONS = {"jax": "0.10.1", "jaxlib": "0.10.1", "libtpu": "0.0.41"}
EXPECTED_PYTHON_VERSION = "3.12.13"
EXPECTED_DEVICE_COUNT = 4
EXPECTED_HISTORICAL_RUNTIME_ROWS = 1_035
EXPECTED_LOCAL_EDITABLE_PATHS = frozenset(
    {
        ".",
        "infra/pulumi",
        "lib/ducky",
        "lib/dupekit",
        "lib/finelog",
        "lib/finestore",
        "lib/fray",
        "lib/haliax",
        "lib/iris",
        "lib/levanter",
        "lib/marin",
        "lib/rigging",
        "lib/zephyr",
    }
)
LOCAL_LOCK_SOURCE_KEYS = frozenset({"editable", "directory", "path", "virtual"})
EXPECTED_LOCK_SOURCE_KIND_COUNTS = {"editable": 13, "git": 1, "registry": 502}
V1_RELEASE_PATH = REFERENCE_ROOT / "starcoder_wsd80_gradient_plot_completion_v1_20260822/release.json"
V1_RELEASE_FILE_SHA256 = "6d150e35f344f8cd59d961ed89a7b3741828ca571e32fcfc76a2fa10fbb17d0d"
V1_RELEASE_SHA256 = "7e21254a3d41126487ea737a5a1566202d5d692f13b44352769bce16d5e6e292"
V1_FAILURE_PATH = REFERENCE_ROOT / "starcoder_wsd80_gradient_plot_completion_v1_20260822/PRELAUNCH_SUPERSEDED.md"
V2_RELEASE_PATH = REFERENCE_ROOT / "starcoder_wsd80_gradient_plot_completion_v2_20260822/release.json"
V2_RELEASE_FILE_SHA256 = "68f93a624a6aa98d0772eb80b0df591542f7650957c7d3895b83ee729a68ccb1"
V2_RELEASE_SHA256 = "c41eb31750e505dc3d66eabda8b3ce155c20d11259b0f88647b4099d854a69c0"
V2_FAILURE_PATH = REFERENCE_ROOT / "starcoder_wsd80_gradient_plot_completion_v2_20260822/PRELAUNCH_SUPERSEDED.md"
V3_RELEASE_PATH = REFERENCE_ROOT / "starcoder_wsd80_gradient_plot_completion_v3_20260822/release.json"
V3_RELEASE_FILE_SHA256 = "c3936ce207c1065088286bfac917bc4b66fddb273a7df2ac3453037ee59196fb"
V3_RELEASE_SHA256 = "192919e6a761596c81c71c969d829823a477ba8ae4e7692a55dfaa67f48b6ce4"
V3_FAILURE_PATH = REFERENCE_ROOT / "starcoder_wsd80_gradient_plot_completion_v3_20260822/RUNTIME_PROVENANCE_FAILURE.md"
V4_RELEASE_PATH = REFERENCE_ROOT / "starcoder_wsd80_gradient_plot_completion_v4_20260822/release.json"
V4_RELEASE_FILE_SHA256 = "170353ba572c943e043cc7217dd46eeb484c1e7fb39ddcf8cbe513ed6fde9820"
V4_RELEASE_SHA256 = "46bae7e7b6d33dc1bc02219d224dc017a4269cb77930cf981f8ac9b03e538b24"
V4_FAILURE_PATH = REFERENCE_ROOT / "starcoder_wsd80_gradient_plot_completion_v4_20260822/PRELAUNCH_PROVENANCE_FAILURE.md"
V5_RELEASE_PATH = REFERENCE_ROOT / "starcoder_wsd80_gradient_plot_completion_v5_20260822/release.json"
V5_RELEASE_FILE_SHA256 = "670d5295b9d925554f4901178672777f5de075329cfc374b5725ca12b163bc14"
V5_RELEASE_SHA256 = "e229181c9048a62cf439f2e5b92fde63b0bb70160894e8141b1a2401554a4364"
V5_FAILURE_PATH = REFERENCE_ROOT / "starcoder_wsd80_gradient_plot_completion_v5_20260822/RUNTIME_BUNDLE_FAILURE.md"
V6_RELEASE_PATH = REFERENCE_ROOT / "starcoder_wsd80_gradient_plot_completion_v6_20260822/release.json"
V6_RELEASE_FILE_SHA256 = "7df3f4f35f892c4b58c9b0175e6da4d03be15c626a9cbd36deda6b8c92c2c9bb"
V6_RELEASE_SHA256 = "0665eba2ea20ccf01447bbd9cbf0e6e2895876bfc05187a78a6cdfa4f97a3a7c"
V6_FAILURE_PATH = REFERENCE_ROOT / "starcoder_wsd80_gradient_plot_completion_v6_20260822/FROZEN_LOCK_BUNDLE_FAILURE.md"
V7_RELEASE_PATH = REFERENCE_ROOT / "starcoder_wsd80_gradient_plot_completion_v7_20260822/release.json"
V7_RELEASE_FILE_SHA256 = "613584f1cef5fedd5e1184994f3d902eb1db8610af6166d70859d4d0cfecc32e"
V7_RELEASE_SHA256 = "7451260aec29960c7405335e18dc784be31e93d6daaaef6ea73daab0dccbfde6"
V7_FAILURE_PATH = REFERENCE_ROOT / "starcoder_wsd80_gradient_plot_completion_v7_20260822/WORKER_ADAPTER_FAILURE.md"

TARGET_DISTRIBUTIONS = repair.TARGET_DISTRIBUTIONS
GLOBAL_STARCODER = repair.GLOBAL_STARCODER
SUPPORT_STARCODER = repair.SUPPORT_STARCODER
NEMOTRON = repair.NEMOTRON

COMMON_CELL = "r3_increase_d_h0640_s28260"
H5_CELL = "h5_fixed_aggregate_h0640_s28160"
COMMON_POLICY = "common_tied_035"
H5_POLICIES = frozenset({"boundary_beta_0p60", "boundary_beta_0p85"})
COMMON_TARGET_STATES = frozenset({"fraction_0p10", "fraction_0p25", "fraction_0p70", "decay_onset"})
H5_TARGET_STATES = frozenset({"fraction_0p90"})
H5_SOURCE_ONLY_STATES = frozenset({"final"})

EXPECTED_COMMON_TRAJECTORIES = {"m100a": 24, "full": 24, "m100b": 8}
EXPECTED_H5_TRAJECTORIES = {"boundary_beta_0p60": 16, "boundary_beta_0p85": 16}
EXPECTED_FULL_ROWS = 288
EXPECTED_TRAJECTORIES = 88
EXPECTED_CHECKPOINT_LABEL_COUNTS = {
    "decay_onset": 56,
    "final": 32,
    "fraction_0p10": 56,
    "fraction_0p25": 56,
    "fraction_0p70": 56,
    "fraction_0p90": 32,
}
STAGE_ROW_COUNTS = {1: 8, 2: 16, 3: 32, 4: 232}
STAGE_MAX_CONCURRENT = {1: 8, 2: 16, 3: 32, 4: 64}

EXECUTION_ACCEPTANCE = {
    key: value for key, value in repair.EXECUTION_ACCEPTANCE.items() if not key.startswith("canary_")
}

ANALYSIS_CONTRACT: dict[str, Any] = {
    "contract_version": RELEASE_VERSION,
    "scientific_status": SCIENTIFIC_STATUS,
    "outcomes_inspected_before_contract": True,
    "endpoint_metrics_read_by_runner": False,
    "historical_runtime_reproduction": (
        "The v3 provenance canary reproduced every loss exactly but failed the immutable parent-statistic gate after "
        "the checkout moved from JAX 0.10.1/libtpu 0.0.41 and the v10 Levanter gradient accumulator to JAX "
        "0.11.0/libtpu 0.0.44 and a changed accumulator. The v4 preflight then showed that the v10 jobs' recorded "
        "commit had a dirty worktree: clean 7efb could not reconstruct the frozen v6 configs, while the dirty "
        "library state later committed as 377ad reproduces all 256 configs and 5,888 fields exactly. This release "
        "therefore uses a hybrid execution tree: root lockfiles and numerical lib/* sources come from 377ad, while "
        "the recovery runner and probe kernels are a separately hash-pinned overlay. Workers verify both source sets "
        "and record the complete installed distribution inventory. They request an immutable task-image digest and "
        "retain the original 5e-6 parent-reproduction tolerance. The source manifest omits only three deterministic "
        "workspace-packaging artifacts: Iris stamps its build-info date, and its bundle excludes two dashboard HTML "
        "files. The historical uv.lock remains included byte-for-byte and submission runs under UV_FROZEN=1. "
        "Every local editable package named by that lock, including infra/pulumi, is source-pinned and bundled."
    ),
    "recorded_clean_commit": RECORDED_CLEAN_COMMIT,
    "historical_library_source_commit": HISTORICAL_RUNTIME_COMMIT,
    "execution_tree_construction": (
        "Create a detached worktree at historical_library_source_commit, overlay the release-pinned experiment "
        "implementation files and frozen artifacts, then submit from that tree. The task-image digest is a requested "
        "scheduler input; numerical equivalence is established by worker-side source hashes, exact historical "
        "versions for the required numerical packages, and parent-statistic reproduction gates. The complete "
        "installed-package inventory is a cross-stage consistency check, not a v10 baseline comparison."
    ),
    "superseded_prelaunch_draft_release_sha256": V1_RELEASE_SHA256,
    "superseded_oversize_workspace_release_sha256": V2_RELEASE_SHA256,
    "superseded_runtime_canary_release_sha256": V3_RELEASE_SHA256,
    "superseded_prelaunch_release_sha256": V4_RELEASE_SHA256,
    "superseded_runtime_bundle_release_sha256": V5_RELEASE_SHA256,
    "superseded_frozen_lock_bundle_release_sha256": V6_RELEASE_SHA256,
    "superseded_worker_adapter_release_sha256": V7_RELEASE_SHA256,
    "worker_adapter_recovery": (
        "V7 reached all eight Stage-1 workers but produced no result rows because a process-local configured flag "
        "survived remote serialization while the patched v10 module binding did not. V8 removes that early return, "
        "reapplies every binding at each process entry, and checks the resulting identities. A local cloudpickle "
        "round-trip smoke-tests binding repair; the required one-worker remote canary reproduces the actual Iris "
        "process boundary and verifies worker sources, packages, configuration, root binding, and output creation "
        "before authorization."
    ),
    "execution_materialization_separation": (
        "Local readiness and authorization verify every frozen plot-only input. The remote execution parent omits the "
        "eight large visualization CSVs and relies on the release-bound authorization sidecar; it still verifies the "
        "small v10 execution-reference manifest. Workers read only frozen execution manifests, saved checkpoints, and "
        "parent probe results. Materialization re-verifies every plot input before merging outputs."
    ),
    "purpose": (
        "Complete descriptive temporal plots using checkpoints already saved by the frozen v6 trajectory panel. "
        "No training trajectory is rerun, and no frozen inferential summary or hypothesis test is recomputed."
    ),
    "display_scope": {
        "common_tied_035": {
            "cell_id": COMMON_CELL,
            "target_update_states_added": sorted(COMMON_TARGET_STATES),
            "source_states_already_recovered_locally": [
                "fraction_0p10",
                "fraction_0p25",
                "fraction_0p40",
                "fraction_0p55",
                "fraction_0p70",
                "decay_minus_256",
                "decay_minus_64",
                "decay_onset",
                "decay_plus_64",
                "decay_plus_256",
                "fraction_0p90",
                "final",
            ],
        },
        "h5_fixed_aggregate": {
            "cell_id": H5_CELL,
            "policies": sorted(H5_POLICIES),
            "target_update_states_added": sorted(H5_TARGET_STATES),
            "source_only_states_added": sorted(H5_SOURCE_ONLY_STATES),
        },
    },
    "structural_missingness": (
        "Target-source optimizer-update alignment at final is undefined, not missing: the learning rate is zero, "
        "so every corrected optimizer update is the zero vector and has no cosine direction. Final rows therefore "
        "request source gradients only."
    ),
    "scope_exclusion": (
        "Saved checkpoints from other N-D cells and policy cohorts are not silently pooled into these fixed-cell "
        "plots. They remain available for separate cross-cell or policy-specific analyses."
    ),
    "inference_exclusion": (
        "These rows were selected after outcomes and figure inspection. They are descriptive completion data and "
        "must not be used to revise the frozen H1-H5 p-values, confidence intervals, or multiplicity family."
    ),
    "execution_design": (
        "The 256 target-bearing completion rows use exactly two workload shapes and the 32 source-only final rows "
        "use a third shape. Stage 1 executes eight rows spanning all six missing checkpoint labels, all three "
        "workload shapes, all three support cohorts, both H5 policies, and the final zero-learning-rate path. Later "
        "stages increase maximum concurrency by at most 2x and are blocked until every exact prior-stage output "
        "audit passes."
    ),
    "materialization_contract": (
        "Merged target/source tables are visualization-only. Every row retains its original analysis role and an "
        "explicit evidence_role, and merged targets use `_visualization_only.csv` filenames; they must not be passed "
        "to the frozen v10 inferential analyzer."
    ),
}


def canonical_json(value: Any) -> str:
    return repair.canonical_json(value)


def canonical_sha256(value: Any) -> str:
    return repair.canonical_sha256(value)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _review_verdict(text: str) -> str:
    for line in text.splitlines():
        normalized = line.strip().strip("#*` ")
        if not normalized:
            continue
        if normalized not in {"PASS_AFTER_BLOCKERS_RESOLVED", "BLOCKED"}:
            raise ValueError(f"Final CC review must begin with the exact verdict token, found: {normalized}")
        return normalized
    raise ValueError("Final CC review is empty")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        if path.read_bytes() != payload:
            raise RuntimeError(f"Frozen completion artifact already exists with different content: {path}") from error


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty manifest: {path}")
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _write_create_only(path, buffer.getvalue().encode())


def _write_json(path: Path, value: Any) -> None:
    _write_create_only(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def _historical_lock_inventory() -> dict[str, Any]:
    lock = subprocess.run(
        ("git", "show", f"{HISTORICAL_RUNTIME_COMMIT}:uv.lock"),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    source_kind_counts: Counter[str] = Counter()
    local_sources: list[dict[str, str]] = []
    for package in tomllib.loads(lock)["package"]:
        source = package.get("source", {})
        if len(source) != 1:
            raise RuntimeError(f"Historical lock source shape is ambiguous for {package['name']}: {source}")
        source_kind, source_value = next(iter(source.items()))
        if source_kind not in {"registry", "git", *LOCAL_LOCK_SOURCE_KEYS}:
            raise RuntimeError(f"Historical lock contains an unknown source kind: {source_kind}")
        source_kind_counts[source_kind] += 1
        if source_kind in LOCAL_LOCK_SOURCE_KEYS:
            local_sources.append(
                {
                    "package": str(package["name"]),
                    "source_kind": source_kind,
                    "path": str(source_value),
                }
            )
    if dict(sorted(source_kind_counts.items())) != EXPECTED_LOCK_SOURCE_KIND_COUNTS:
        raise RuntimeError(
            "Historical lock source-kind inventory drifted: "
            f"{dict(sorted(source_kind_counts.items()))} != {EXPECTED_LOCK_SOURCE_KIND_COUNTS}"
        )
    local_paths = frozenset(item["path"] for item in local_sources)
    if local_paths != EXPECTED_LOCAL_EDITABLE_PATHS:
        raise RuntimeError(
            "Historical local dependency set drifted: "
            f"{sorted(local_paths)} != {sorted(EXPECTED_LOCAL_EDITABLE_PATHS)}"
        )
    if {item["source_kind"] for item in local_sources} != {"editable"}:
        raise RuntimeError(f"Historical local dependency source kinds drifted: {local_sources}")
    return {
        "local_sources": sorted(local_sources, key=lambda item: (item["path"], item["package"])),
        "source_kind_counts": dict(sorted(source_kind_counts.items())),
    }


def _historical_runtime_rows() -> list[dict[str, str]]:
    """Hash the numerical source tree exactly as it existed for the v10 outputs."""
    archive = subprocess.run(
        ("git", "archive", "--format=tar", HISTORICAL_RUNTIME_COMMIT, *HISTORICAL_RUNTIME_PATHS),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    rows: list[dict[str, str]] = []
    consumed_exclusions: set[str] = set()
    with tarfile.open(fileobj=BytesIO(archive), mode="r:") as handle:
        for member in sorted(handle.getmembers(), key=lambda item: item.name):
            if not member.isfile():
                continue
            if member.name in HISTORICAL_RUNTIME_EXCLUDED_PATHS:
                consumed_exclusions.add(member.name)
                continue
            extracted = handle.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Historical runtime archive omitted file payload: {member.name}")
            payload = extracted.read()
            rows.append(
                {
                    "path": member.name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size": str(len(payload)),
                }
            )
    if not rows:
        raise RuntimeError("Historical runtime source manifest is empty")
    if consumed_exclusions != HISTORICAL_RUNTIME_EXCLUDED_PATHS:
        raise RuntimeError(
            "Historical runtime packaging exclusions drifted: "
            f"{sorted(consumed_exclusions)} != {sorted(HISTORICAL_RUNTIME_EXCLUDED_PATHS)}"
        )
    if len(rows) != EXPECTED_HISTORICAL_RUNTIME_ROWS:
        raise RuntimeError(
            f"Historical runtime source row count drifted: {len(rows)} != {EXPECTED_HISTORICAL_RUNTIME_ROWS}"
        )
    lock_inventory = _historical_lock_inventory()
    editable_paths = frozenset(item["path"] for item in lock_inventory["local_sources"])
    manifested_paths = {row["path"] for row in rows}
    for editable_path in sorted(editable_paths):
        pyproject_path = "pyproject.toml" if editable_path == "." else f"{editable_path}/pyproject.toml"
        if pyproject_path not in manifested_paths:
            raise RuntimeError(f"Historical local editable package is missing its pyproject: {editable_path}")
        if editable_path != "." and not any(path.startswith(f"{editable_path}/src/") for path in manifested_paths):
            raise RuntimeError(f"Historical local editable package is missing its source tree: {editable_path}")
    return rows


def _full_training_design_reference_paths() -> dict[str, Path]:
    manifest_path = parent.full.DESIGN_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    references = {"full_training_design_manifest": manifest_path}
    artifact_hashes = {
        **manifest["input_artifact_sha256"],
        **manifest["artifact_sha256"],
    }
    for relative_path, expected_hash in sorted(artifact_hashes.items()):
        path = (
            REPO_ROOT / relative_path if relative_path.startswith("experiments/") else parent.DESIGN_DIR / relative_path
        )
        if file_sha256(path) != expected_hash:
            raise ValueError(f"Full training design artifact drifted: {relative_path}")
        references[f"full_training_design/{relative_path}"] = path
    return references


def _required_stage1_workspace_paths(
    *,
    runtime_stack: list[dict[str, str]],
    implementation_files: dict[str, str],
    parent_implementation_files: dict[str, str],
    parent_release: dict[str, Any],
    execution_reference_paths: dict[str, Path],
) -> list[str]:
    paths = {
        str(RELEASE_PATH.relative_to(REPO_ROOT)),
        str(ANALYSIS_CONTRACT_PATH.relative_to(REPO_ROOT)),
        str(FULL_MANIFEST_PATH.relative_to(REPO_ROOT)),
        str(COVERAGE_AUDIT_PATH.relative_to(REPO_ROOT)),
        str(REPORT_PATH.relative_to(REPO_ROOT)),
        str(CHECKPOINT_PROVENANCE_PATH.relative_to(REPO_ROOT)),
        str(PARENT_RESULT_PROVENANCE_PATH.relative_to(REPO_ROOT)),
        str(RUNTIME_STACK_MANIFEST_PATH.relative_to(REPO_ROOT)),
        str(FULL_LAUNCH_AUTHORIZATION_PATH.relative_to(REPO_ROOT)),
        str(CC_REVIEW_PATH.relative_to(REPO_ROOT)),
        str(V10_RELEASE_PATH.relative_to(REPO_ROOT)),
        str(PARENT_RELEASE_PATH.relative_to(REPO_ROOT)),
        str(PARENT_FULL_MANIFEST_PATH.relative_to(REPO_ROOT)),
        str(TRAJECTORY_MANIFEST_PATH.relative_to(REPO_ROOT)),
        *(str(path.relative_to(REPO_ROOT)) for path in execution_reference_paths.values()),
        *(str(path.relative_to(REPO_ROOT)) for path in EXECUTION_IMPORT_PATHS),
        *implementation_files,
        *parent_implementation_files,
        *(str(row["path"]) for row in runtime_stack),
    }
    for release_path, failure_path in (
        (V1_RELEASE_PATH, V1_FAILURE_PATH),
        (V2_RELEASE_PATH, V2_FAILURE_PATH),
        (V3_RELEASE_PATH, V3_FAILURE_PATH),
        (V4_RELEASE_PATH, V4_FAILURE_PATH),
        (V5_RELEASE_PATH, V5_FAILURE_PATH),
        (V6_RELEASE_PATH, V6_FAILURE_PATH),
        (V7_RELEASE_PATH, V7_FAILURE_PATH),
    ):
        paths.add(str(release_path.relative_to(REPO_ROOT)))
        paths.add(str(failure_path.relative_to(REPO_ROOT)))
    paths.update(str(summary["path"]) for summary in parent_release["manifests"].values())
    paths.update(
        str((parent.DESIGN_DIR / name).relative_to(REPO_ROOT)) for name in parent_release["source_design_files"]
    )
    return sorted(paths)


def _parent_release() -> dict[str, Any]:
    return repair._parent_release()


def _completion_identity(payload: dict[str, Any]) -> tuple[str, str]:
    identity = {"scope": "full", **payload, "parent_release_sha256": _parent_release()["release_sha256"]}
    digest = canonical_sha256(identity)[:24]
    return f"plot_completion_{digest}", f"plot_completion_group_{digest}"


def _completion_row(
    group: list[dict[str, str]],
    trajectory: dict[str, str],
    *,
    source_ids: tuple[str, ...],
    target_ids: tuple[str, ...],
    completion_role: str,
    display_analysis_role: str,
) -> dict[str, Any]:
    representative = group[0]
    block_counts, update_draws, sequence_sets, probe_row_ids = repair._distribution_contract(
        group,
        source_ids=source_ids,
        target_ids=target_ids,
    )
    payload = {
        "trajectory_id": representative["trajectory_id"],
        "checkpoint_label": representative["checkpoint_label"],
        "checkpoint_step": int(representative["checkpoint_step"]),
        "expected_restored_state_step": int(representative["expected_restored_state_step"]),
        "checkpoint_uri": representative["checkpoint_uri"],
        "parent_probe_group_id": representative["group_id"],
        "train_config_sha256": representative["train_config_sha256"],
        "analysis_role": completion_role,
        "completion_role": completion_role,
        "display_analysis_role": display_analysis_role,
        "source_distribution_ids_json": canonical_json(source_ids),
        "target_distribution_ids_json": canonical_json(target_ids),
        "distribution_block_counts_json": canonical_json(block_counts),
        "distribution_update_draw_counts_json": canonical_json(update_draws),
        "distribution_sequence_set_ids_json": canonical_json(sequence_sets),
        "distribution_probe_row_ids_json": canonical_json(probe_row_ids),
        "scientific_status": SCIENTIFIC_STATUS,
        "endpoint_metrics_read_by_runner": False,
        "arm": trajectory["arm"],
        "cell_id": trajectory["cell_id"],
        "support_id": trajectory["support_id"],
        "training_seed": int(trajectory["training_seed"]),
        "policy_role": trajectory["policy_role"],
    }
    row_id, group_id = _completion_identity(payload)
    return {**payload, "scope": "full", "row_id": row_id, "group_id": group_id, "launch_stage": 0}


def _display_role(support_id: str, policy_role: str) -> str:
    if policy_role in H5_POLICIES:
        return "h5_preregistered_profile"
    return {
        "m100a": "h2_primary",
        "full": "h3_full_support_pair",
        "m100b": "h3_second_pool_sensitivity",
    }[support_id]


def _assign_launch_stages(rows: list[dict[str, Any]]) -> None:
    """Select a complete runtime gate, then ramp fleet concurrency by at most 2x."""

    def pick_one(
        *,
        role: str,
        checkpoint: str,
        support: str | None = None,
        policy: str | None = None,
    ) -> dict[str, Any]:
        candidates = [
            row
            for row in rows
            if row["completion_role"] == role
            and row["checkpoint_label"] == checkpoint
            and (support is None or row["support_id"] == support)
            and (policy is None or row["policy_role"] == policy)
            and row["group_id"] not in selected_ids
        ]
        if not candidates:
            raise ValueError(
                "No completion row matches stage-1 selector: "
                f"role={role}, checkpoint={checkpoint}, support={support}, policy={policy}"
            )
        return min(candidates, key=lambda row: (row["trajectory_id"], row["checkpoint_label"], row["group_id"]))

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for support, checkpoint in (
        ("m100a", "fraction_0p10"),
        ("full", "fraction_0p25"),
        ("m100b", "fraction_0p70"),
        ("m100a", "decay_onset"),
    ):
        row = pick_one(
            role="common_target_trajectory_completion",
            checkpoint=checkpoint,
            support=support,
        )
        selected.append(row)
        selected_ids.add(row["group_id"])
    for policy in sorted(H5_POLICIES):
        for role, checkpoint in (
            ("h5_target_tail_completion", "fraction_0p90"),
            ("h5_source_tail_completion", "final"),
        ):
            row = pick_one(role=role, checkpoint=checkpoint, policy=policy)
            selected.append(row)
            selected_ids.add(row["group_id"])

    if len(selected_ids) != STAGE_ROW_COUNTS[1]:
        raise ValueError(f"Stage-1 selector drifted: {len(selected_ids)} != {STAGE_ROW_COUNTS[1]}")
    stage_by_id = {group_id: 1 for group_id in selected_ids}
    remaining = sorted(
        (row for row in rows if row["group_id"] not in selected_ids),
        key=lambda row: (row["completion_role"], row["support_id"], row["policy_role"], row["group_id"]),
    )
    offset = 0
    for stage in (2, 3, 4):
        count = STAGE_ROW_COUNTS[stage]
        for row in remaining[offset : offset + count]:
            stage_by_id[row["group_id"]] = stage
        offset += count
    if offset != len(remaining):
        raise ValueError(f"Staged completion inventory drifted: {offset} != {len(remaining)}")
    for row in rows:
        row["launch_stage"] = stage_by_id[row["group_id"]]


def _full_rows() -> list[dict[str, Any]]:
    groups = repair._manifest_groups(PARENT_FULL_MANIFEST_PATH)
    trajectories = {row["trajectory_id"]: row for row in _read_csv(TRAJECTORY_MANIFEST_PATH)}
    rows: list[dict[str, Any]] = []
    for group in groups.values():
        representative = group[0]
        trajectory = trajectories[representative["trajectory_id"]]
        cell_id = trajectory["cell_id"]
        policy_role = trajectory["policy_role"]
        checkpoint_label = representative["checkpoint_label"]
        if cell_id == COMMON_CELL and policy_role == COMMON_POLICY and checkpoint_label in COMMON_TARGET_STATES:
            rows.append(
                _completion_row(
                    group,
                    trajectory,
                    source_ids=(GLOBAL_STARCODER, SUPPORT_STARCODER, NEMOTRON),
                    target_ids=TARGET_DISTRIBUTIONS,
                    completion_role="common_target_trajectory_completion",
                    display_analysis_role=_display_role(trajectory["support_id"], policy_role),
                )
            )
        elif cell_id == H5_CELL and policy_role in H5_POLICIES and checkpoint_label in H5_TARGET_STATES:
            rows.append(
                _completion_row(
                    group,
                    trajectory,
                    source_ids=(GLOBAL_STARCODER, NEMOTRON),
                    target_ids=TARGET_DISTRIBUTIONS,
                    completion_role="h5_target_tail_completion",
                    display_analysis_role=_display_role(trajectory["support_id"], policy_role),
                )
            )
        elif cell_id == H5_CELL and policy_role in H5_POLICIES and checkpoint_label in H5_SOURCE_ONLY_STATES:
            rows.append(
                _completion_row(
                    group,
                    trajectory,
                    source_ids=(GLOBAL_STARCODER, NEMOTRON),
                    target_ids=(),
                    completion_role="h5_source_tail_completion",
                    display_analysis_role=_display_role(trajectory["support_id"], policy_role),
                )
            )
    rows = sorted(rows, key=lambda row: row["group_id"])
    _assign_launch_stages(rows)
    return rows


def _coverage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(row["completion_role"] for row in rows)
    return [
        {
            "plot_family": "source-source alignment matrix and trajectory",
            "cohort": "common tied 0.35, r3",
            "missing_states_before": "none after local all-state source materialization",
            "saved_checkpoint_groups_to_probe": 0,
            "repair_action": "reuse sealed v10 source statistics",
            "status_after_release": "complete: 12/12 saved states",
        },
        {
            "plot_family": "source-source alignment matrix and trajectory",
            "cohort": "H5 beta 0.60 and 0.85",
            "missing_states_before": "fraction_0p90; final",
            "saved_checkpoint_groups_to_probe": (
                counts["h5_target_tail_completion"] + counts["h5_source_tail_completion"]
            ),
            "repair_action": "recompute source gradients and corrected updates from sealed checkpoints",
            "status_after_release": (
                "complete: 11/11 saved states; final optimizer update remains structurally undefined"
            ),
        },
        {
            "plot_family": "target-source alignment matrices and trajectories",
            "cohort": "common tied 0.35, r3",
            "missing_states_before": "; ".join(sorted(COMMON_TARGET_STATES)),
            "saved_checkpoint_groups_to_probe": counts["common_target_trajectory_completion"],
            "repair_action": "recompute four target gradients and three source updates from sealed checkpoints",
            "status_after_release": "complete: 11/11 states with nonzero learning rate; final undefined by construction",
        },
        {
            "plot_family": "target-source alignment matrices and trajectories",
            "cohort": "H5 beta 0.60 and 0.85",
            "missing_states_before": "fraction_0p90",
            "saved_checkpoint_groups_to_probe": counts["h5_target_tail_completion"],
            "repair_action": "recompute four target gradients and two source updates from sealed checkpoints",
            "status_after_release": "complete: 10/10 states with nonzero learning rate; final undefined by construction",
        },
        {
            "plot_family": "frozen effect forest and preregistered summaries",
            "cohort": "H1-H5 frozen estimands",
            "missing_states_before": "none",
            "saved_checkpoint_groups_to_probe": 0,
            "repair_action": "none; do not alter frozen inference",
            "status_after_release": "unchanged and complete",
        },
        {
            "plot_family": "endpoint interventions and H4 rollout validation",
            "cohort": "linked frozen endpoint artifact",
            "missing_states_before": "none in their registered panels",
            "saved_checkpoint_groups_to_probe": 0,
            "repair_action": "none",
            "status_after_release": "unchanged and complete",
        },
    ]


def _validate_design(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != EXPECTED_FULL_ROWS or len({row["group_id"] for row in rows}) != EXPECTED_FULL_ROWS:
        raise ValueError(f"Plot-completion row count drifted: {len(rows)} != {EXPECTED_FULL_ROWS}")
    if len({row["row_id"] for row in rows}) != EXPECTED_FULL_ROWS:
        raise ValueError("Plot-completion row identities are not unique")
    trajectory_count = len({row["trajectory_id"] for row in rows})
    if trajectory_count != EXPECTED_TRAJECTORIES:
        raise ValueError(f"Plot-completion trajectory inventory drifted: {trajectory_count} != {EXPECTED_TRAJECTORIES}")
    checkpoint_label_counts = Counter(row["checkpoint_label"] for row in rows)
    if checkpoint_label_counts != EXPECTED_CHECKPOINT_LABEL_COUNTS:
        raise ValueError(
            "Plot-completion checkpoint inventory drifted: "
            f"{checkpoint_label_counts} != {EXPECTED_CHECKPOINT_LABEL_COUNTS}"
        )
    stage_counts = Counter(int(row["launch_stage"]) for row in rows)
    if stage_counts != STAGE_ROW_COUNTS:
        raise ValueError(f"Plot-completion stage inventory drifted: {stage_counts} != {STAGE_ROW_COUNTS}")
    common = [row for row in rows if row["completion_role"] == "common_target_trajectory_completion"]
    common_inventory = Counter(row["support_id"] for row in common)
    expected_common = {
        support: count * len(COMMON_TARGET_STATES) for support, count in EXPECTED_COMMON_TRAJECTORIES.items()
    }
    if common_inventory != expected_common:
        raise ValueError(f"Common target-completion inventory drifted: {common_inventory} != {expected_common}")
    h5 = [row for row in rows if row["policy_role"] in H5_POLICIES]
    h5_inventory = Counter(row["policy_role"] for row in h5)
    expected_h5 = {
        policy: count * (len(H5_TARGET_STATES) + len(H5_SOURCE_ONLY_STATES))
        for policy, count in EXPECTED_H5_TRAJECTORIES.items()
    }
    if h5_inventory != expected_h5:
        raise ValueError(f"H5 tail-completion inventory drifted: {h5_inventory} != {expected_h5}")
    if any(json.loads(row["target_distribution_ids_json"]) for row in rows if row["checkpoint_label"] == "final"):
        raise ValueError("Final rows must not fabricate target-update alignment when LR=0")
    parent_group_ids = set(repair._manifest_groups(PARENT_FULL_MANIFEST_PATH))
    v10_group_ids = {row["parent_probe_group_id"] for row in _read_csv(repair.FULL_MANIFEST_PATH)}
    completion_group_ids = {row["parent_probe_group_id"] for row in rows}
    if not completion_group_ids <= parent_group_ids:
        raise ValueError("Plot-completion rows reference an unknown parent probe group")
    outside_plot_scope = parent_group_ids - v10_group_ids - completion_group_ids
    stage1_rows = [row for row in rows if int(row["launch_stage"]) == 1]
    target_rows = [row for row in rows if row["completion_role"] != "h5_source_tail_completion"]
    stage1_shapes = {repair._workload_shape_sha256(row) for row in stage1_rows}
    stage1_labels = {row["checkpoint_label"] for row in stage1_rows}
    expected_stage1_labels = COMMON_TARGET_STATES | H5_TARGET_STATES | H5_SOURCE_ONLY_STATES
    if stage1_labels != expected_stage1_labels:
        raise ValueError(f"Stage 1 checkpoint coverage drifted: {stage1_labels} != {expected_stage1_labels}")
    stage1_supports = {row["support_id"] for row in stage1_rows}
    if stage1_supports != set(EXPECTED_COMMON_TRAJECTORIES):
        raise ValueError(f"Stage 1 support coverage drifted: {stage1_supports}")
    stage1_h5_policies = {row["policy_role"] for row in stage1_rows if row["policy_role"] in H5_POLICIES}
    if stage1_h5_policies != H5_POLICIES:
        raise ValueError(f"Stage 1 H5 policy coverage drifted: {stage1_h5_policies}")
    new_target_shapes = {repair._workload_shape_sha256(row) for row in target_rows}
    if len(new_target_shapes) != 2:
        raise ValueError(f"Expected exactly two new target-bearing workload shapes, found {len(new_target_shapes)}")
    source_only_shapes = {
        repair._workload_shape_sha256(row) for row in rows if row["completion_role"] == "h5_source_tail_completion"
    }
    expected_stage1_shapes = new_target_shapes | source_only_shapes
    if stage1_shapes != expected_stage1_shapes:
        raise ValueError("Stage 1 must exercise both target-bearing shapes and the final source-only shape")
    v10_shapes = {repair._workload_shape_sha256(row) for row in _read_csv(repair.FULL_MANIFEST_PATH)}
    if not source_only_shapes <= v10_shapes:
        raise ValueError("Source-only final workload shape was not previously executed by v10")
    ordered_stages = sorted(STAGE_MAX_CONCURRENT)
    for previous, current in pairwise(ordered_stages):
        if STAGE_MAX_CONCURRENT[current] > 2 * STAGE_MAX_CONCURRENT[previous]:
            raise ValueError(f"Stage concurrency jumps by more than 2x: {previous} -> {current}")
    return {
        "full_rows": len(rows),
        "trajectory_count": trajectory_count,
        "completion_role_counts": dict(sorted(Counter(row["completion_role"] for row in rows).items())),
        "checkpoint_label_counts": dict(sorted(checkpoint_label_counts.items())),
        "launch_stage_counts": dict(sorted(stage_counts.items())),
        "stage1_checkpoint_labels": sorted(stage1_labels),
        "stage1_support_ids": sorted(stage1_supports),
        "stage1_h5_policies": sorted(stage1_h5_policies),
        "stage1_new_workload_shape_sha256": sorted(stage1_shapes),
        "previously_executed_source_only_shape_sha256": sorted(source_only_shapes),
        "support_counts": dict(sorted(Counter(row["support_id"] for row in rows).items())),
        "parent_groups_outside_plot_scope": len(outside_plot_scope),
    }


def _report(rows: list[dict[str, Any]], coverage: list[dict[str, Any]], validation: dict[str, Any]) -> str:
    other_parent_groups = validation["parent_groups_outside_plot_scope"]
    role_counts = validation["completion_role_counts"]
    concurrency_ramp = " -> ".join(str(STAGE_MAX_CONCURRENT[stage]) for stage in sorted(STAGE_MAX_CONCURRENT))
    lines = [
        "# StarCoder WSD80 gradient-plot coverage audit",
        "",
        "This audit compares every page in the gradient-mechanism artifact with the sealed v6 checkpoint/probe "
        "manifest and the immutable v10 repair release. It does not inspect endpoint outcomes.",
        "",
        "## Result",
        "",
        f"- **{len(rows)} saved-checkpoint groups are recoverable without retraining.**",
        (
            f"- {role_counts['common_target_trajectory_completion']} fill four missing target-source states on "
            "the r3 common-tied trajectories."
        ),
        (
            f"- {role_counts['h5_target_tail_completion']} add the saved 0.90T state to both H5 policies, "
            "including target-source statistics."
        ),
        (
            f"- {role_counts['h5_source_tail_completion']} add the saved final state to both H5 policies for "
            "source-gradient geometry only."
        ),
        (
            "- Final target-source optimizer-update alignment is not a recoverable number: LR=0 makes every "
            "corrected update the zero vector, so its cosine is undefined."
        ),
        (
            f"- {other_parent_groups:,} other saved parent groups belong to different N-D cells or policy "
            "cohorts and are intentionally not pooled into these fixed-cell plots."
        ),
        "- Frozen effect estimates, p-values, intervals, and multiplicity decisions remain unchanged.",
        (
            f"- Execution is gated: {STAGE_ROW_COUNTS[1]} rows exercise both target-bearing shapes, then later stages "
            f"ramp maximum concurrency {concurrency_ramp} only after exact prior-stage audits."
        ),
        "",
        "## Page audit",
        "",
        "| Plot family | Cohort | Missing before | Probe groups | Action | State after |",
        "|---|---|---|---:|---|---|",
    ]
    for row in coverage:
        lines.append(
            f"| {row['plot_family']} | {row['cohort']} | {row['missing_states_before']} | "
            f"{row['saved_checkpoint_groups_to_probe']} | {row['repair_action']} | {row['status_after_release']} |"
        )
    lines.extend(
        [
            "",
            "The Probe groups column is per plot family and is not additive: each H5 0.90T probe contributes to both "
            "the source-source and target-source pages.",
            "",
            f"The complete plot artifact is rendered to `{PLOT_OUTPUT_DIR.relative_to(REPO_ROOT)}`. It supersedes the "
            "older v10 plot directory for temporal coverage without changing the frozen v10 inferential tables.",
            "",
            "## Interpretation boundary",
            "",
            "The completion rows were selected after inspecting the figures. They improve descriptive temporal "
            "coverage only. They are not untouched confirmation and cannot revise the preregistered mechanism verdicts.",
            "",
        ]
    )
    return "\n".join(lines)


def freeze() -> dict[str, Any]:
    """Materialize a hash-pinned, endpoint-blind plot-completion release."""
    if not CC_REVIEW_PATH.exists():
        raise FileNotFoundError(f"Final CC review is missing: {CC_REVIEW_PATH}")
    review_verdict = _review_verdict(CC_REVIEW_PATH.read_text())
    if review_verdict != "PASS_AFTER_BLOCKERS_RESOLVED":
        raise ValueError(f"Final CC review did not pass: {review_verdict}")
    parent_release = _parent_release()
    v10_release = json.loads(V10_RELEASE_PATH.read_text())
    if file_sha256(V10_RELEASE_PATH) != V10_RELEASE_FILE_SHA256:
        raise ValueError("Immutable v10 release file drifted")
    if v10_release["release_sha256"] != V10_RELEASE_SHA256:
        raise ValueError("Immutable v10 release identity drifted")
    for label, release_path, release_file_sha256, release_sha256, failure_path in (
        ("v1", V1_RELEASE_PATH, V1_RELEASE_FILE_SHA256, V1_RELEASE_SHA256, V1_FAILURE_PATH),
        ("v2", V2_RELEASE_PATH, V2_RELEASE_FILE_SHA256, V2_RELEASE_SHA256, V2_FAILURE_PATH),
    ):
        superseded_release = json.loads(release_path.read_text())
        if file_sha256(release_path) != release_file_sha256:
            raise ValueError(f"Superseded {label} release file drifted")
        if superseded_release["release_sha256"] != release_sha256:
            raise ValueError(f"Superseded {label} release identity drifted")
        if not failure_path.exists():
            raise FileNotFoundError(f"Superseded {label} failure marker is missing: {failure_path}")
    v3_release = json.loads(V3_RELEASE_PATH.read_text())
    if file_sha256(V3_RELEASE_PATH) != V3_RELEASE_FILE_SHA256:
        raise ValueError("Superseded v3 runtime-canary release file drifted")
    if v3_release["release_sha256"] != V3_RELEASE_SHA256:
        raise ValueError("Superseded v3 runtime-canary release identity drifted")
    if not V3_FAILURE_PATH.exists():
        raise FileNotFoundError(f"Superseded v3 failure marker is missing: {V3_FAILURE_PATH}")
    v4_release = json.loads(V4_RELEASE_PATH.read_text())
    if file_sha256(V4_RELEASE_PATH) != V4_RELEASE_FILE_SHA256:
        raise ValueError("Superseded v4 prelaunch release file drifted")
    if v4_release["release_sha256"] != V4_RELEASE_SHA256:
        raise ValueError("Superseded v4 prelaunch release identity drifted")
    if not V4_FAILURE_PATH.exists():
        raise FileNotFoundError(f"Superseded v4 prelaunch failure marker is missing: {V4_FAILURE_PATH}")
    v5_release = json.loads(V5_RELEASE_PATH.read_text())
    if file_sha256(V5_RELEASE_PATH) != V5_RELEASE_FILE_SHA256:
        raise ValueError("Superseded v5 runtime-bundle release file drifted")
    if v5_release["release_sha256"] != V5_RELEASE_SHA256:
        raise ValueError("Superseded v5 runtime-bundle release identity drifted")
    if not V5_FAILURE_PATH.exists():
        raise FileNotFoundError(f"Superseded v5 runtime-bundle failure marker is missing: {V5_FAILURE_PATH}")
    v6_release = json.loads(V6_RELEASE_PATH.read_text())
    if file_sha256(V6_RELEASE_PATH) != V6_RELEASE_FILE_SHA256:
        raise ValueError("Superseded v6 frozen-lock bundle release file drifted")
    if v6_release["release_sha256"] != V6_RELEASE_SHA256:
        raise ValueError("Superseded v6 frozen-lock bundle release identity drifted")
    if not V6_FAILURE_PATH.exists():
        raise FileNotFoundError(f"Superseded v6 frozen-lock bundle failure marker is missing: {V6_FAILURE_PATH}")
    v7_release = json.loads(V7_RELEASE_PATH.read_text())
    if file_sha256(V7_RELEASE_PATH) != V7_RELEASE_FILE_SHA256:
        raise ValueError("Superseded v7 worker-adapter release file drifted")
    if v7_release["release_sha256"] != V7_RELEASE_SHA256:
        raise ValueError("Superseded v7 worker-adapter release identity drifted")
    if not V7_FAILURE_PATH.exists():
        raise FileNotFoundError(f"Superseded v7 worker-adapter failure marker is missing: {V7_FAILURE_PATH}")
    v10_manifest_summary = v10_release["manifests"]["full"]
    if file_sha256(repair.FULL_MANIFEST_PATH) != v10_manifest_summary["sha256"]:
        raise ValueError("Immutable v10 mechanism manifest drifted")
    implementation_paths = (
        Path(__file__).resolve(),
        RUNTIME_PATH,
        MATERIALIZER_PATH,
        PLOTTER_PATH,
        ALL_SOURCE_MATERIALIZER_PATH,
        CANARY_CONFIG_PATH,
        FULL_CONFIG_PATH,
        Path(repair.__file__).resolve(),
        repair.RUNTIME_PATH,
        repair.ANALYZER_PATH,
    )
    for path in implementation_paths:
        if not path.exists():
            raise ValueError(f"Plot-completion implementation is missing: {path}")
    implementation_files = {str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in implementation_paths}
    parent_implementation_files = parent_release["implementation_files"]
    recovery_implementation_manifest_sha256 = canonical_sha256(
        {
            "implementation_files": implementation_files,
            "parent_implementation_files": parent_implementation_files,
        }
    )
    plot_input_paths = {f"v10_results/{name}": BASE_RESULTS_DIR / name for name in BASE_RESULT_FILES} | {
        "all_state_source_geometry": BASE_SOURCE_GEOMETRY_PATH,
        "multiplicity_audit": MULTIPLICITY_AUDIT_PATH,
    }
    execution_reference_paths = {
        **_full_training_design_reference_paths(),
        "full_training_source_design": FULL_SOURCE_DESIGN_PATH,
        "v10_full_mechanism_manifest": repair.FULL_MANIFEST_PATH,
    }
    for name, path in plot_input_paths.items():
        if not path.exists():
            raise ValueError(f"Plot-completion input is missing: {name}: {path}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _full_rows()
    validation = _validate_design(rows)
    coverage = _coverage_rows(rows)
    runtime_stack = _historical_runtime_rows()
    lock_inventory = _historical_lock_inventory()
    _write_json(ANALYSIS_CONTRACT_PATH, ANALYSIS_CONTRACT)
    _write_csv(FULL_MANIFEST_PATH, rows)
    _write_csv(COVERAGE_AUDIT_PATH, coverage)
    _write_csv(RUNTIME_STACK_MANIFEST_PATH, runtime_stack)
    _write_create_only(REPORT_PATH, (_report(rows, coverage, validation) + "\n").encode())
    rows_by_scope = {"full": rows}
    checkpoint_inventory = repair._checkpoint_inventory(rows_by_scope)
    parent_result_inventory = repair._parent_result_inventory(rows_by_scope)
    _write_csv(CHECKPOINT_PROVENANCE_PATH, checkpoint_inventory)
    _write_csv(PARENT_RESULT_PROVENANCE_PATH, parent_result_inventory)
    required_stage1_workspace_paths = _required_stage1_workspace_paths(
        runtime_stack=runtime_stack,
        implementation_files=implementation_files,
        parent_implementation_files=parent_implementation_files,
        parent_release=parent_release,
        execution_reference_paths=execution_reference_paths,
    )

    materialize_command = (
        "uv run experiments/domain_phase_mix/exploratory/two_phase_many/"
        "materialize_starcoder_wsd80_gradient_plot_completion_20260822.py "
        f"--output-dir {COMPLETE_TABLES_DIR.relative_to(REPO_ROOT)} "
        "--release-sha256 <release_sha256>"
    )
    render_command = (
        "uv run experiments/domain_phase_mix/exploratory/two_phase_many/"
        "plot_starcoder_wsd80_gradient_mechanism_repair_20260820.py "
        f"--input-dir {COMPLETE_TABLES_DIR.relative_to(REPO_ROOT)} "
        f"--output-dir {PLOT_OUTPUT_DIR.relative_to(REPO_ROOT)} "
        f"--multiplicity-audit {MULTIPLICITY_AUDIT_PATH.relative_to(REPO_ROOT)} "
        f"--release-path {RELEASE_PATH.relative_to(REPO_ROOT)} "
        f"--source-geometry-all-states "
        f"{(COMPLETE_TABLES_DIR / 'source_source_geometry_all_states.csv').relative_to(REPO_ROOT)}"
    )
    release = {
        "release_version": RELEASE_VERSION,
        "release_sha256": "",
        "scientific_status": SCIENTIFIC_STATUS,
        "outcomes_inspected_before_repair": True,
        "endpoint_metrics_read_by_runner": False,
        "parent_release_path": str(PARENT_RELEASE_PATH.relative_to(REPO_ROOT)),
        "parent_release_file_sha256": file_sha256(PARENT_RELEASE_PATH),
        "parent_release_sha256": parent_release["release_sha256"],
        "result_root": RESULT_ROOT,
        "required_region": "us-central1",
        "required_zone": "us-central1-a",
        "required_bucket_prefix": MARIN_PREFIX,
        "analysis_contract": {
            "path": str(ANALYSIS_CONTRACT_PATH.relative_to(REPO_ROOT)),
            "sha256": file_sha256(ANALYSIS_CONTRACT_PATH),
        },
        "coverage_audit": {
            "path": str(COVERAGE_AUDIT_PATH.relative_to(REPO_ROOT)),
            "sha256": file_sha256(COVERAGE_AUDIT_PATH),
            "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
            "report_sha256": file_sha256(REPORT_PATH),
        },
        "v10_release": {
            "path": str(V10_RELEASE_PATH.relative_to(REPO_ROOT)),
            "file_sha256": V10_RELEASE_FILE_SHA256,
            "release_sha256": V10_RELEASE_SHA256,
        },
        "superseded_prelaunch_draft": {
            "path": str(V1_RELEASE_PATH.relative_to(REPO_ROOT)),
            "file_sha256": V1_RELEASE_FILE_SHA256,
            "release_sha256": V1_RELEASE_SHA256,
            "failure_marker_path": str(V1_FAILURE_PATH.relative_to(REPO_ROOT)),
            "failure_marker_sha256": file_sha256(V1_FAILURE_PATH),
            "status": "non_consumable_never_authorized_or_launched",
        },
        "superseded_oversize_workspace_release": {
            "path": str(V2_RELEASE_PATH.relative_to(REPO_ROOT)),
            "file_sha256": V2_RELEASE_FILE_SHA256,
            "release_sha256": V2_RELEASE_SHA256,
            "failure_marker_path": str(V2_FAILURE_PATH.relative_to(REPO_ROOT)),
            "failure_marker_sha256": file_sha256(V2_FAILURE_PATH),
            "status": "non_consumable_local_submission_rejected",
        },
        "superseded_runtime_canary": {
            "path": str(V3_RELEASE_PATH.relative_to(REPO_ROOT)),
            "file_sha256": V3_RELEASE_FILE_SHA256,
            "release_sha256": V3_RELEASE_SHA256,
            "failure_marker_path": str(V3_FAILURE_PATH.relative_to(REPO_ROOT)),
            "failure_marker_sha256": file_sha256(V3_FAILURE_PATH),
            "status": "non_consumable_runtime_provenance_failure",
        },
        "superseded_prelaunch_release": {
            "path": str(V4_RELEASE_PATH.relative_to(REPO_ROOT)),
            "file_sha256": V4_RELEASE_FILE_SHA256,
            "release_sha256": V4_RELEASE_SHA256,
            "failure_marker_path": str(V4_FAILURE_PATH.relative_to(REPO_ROOT)),
            "failure_marker_sha256": file_sha256(V4_FAILURE_PATH),
            "status": "non_consumable_never_launched_provenance_failure",
        },
        "superseded_runtime_bundle_release": {
            "path": str(V5_RELEASE_PATH.relative_to(REPO_ROOT)),
            "file_sha256": V5_RELEASE_FILE_SHA256,
            "release_sha256": V5_RELEASE_SHA256,
            "failure_marker_path": str(V5_FAILURE_PATH.relative_to(REPO_ROOT)),
            "failure_marker_sha256": file_sha256(V5_FAILURE_PATH),
            "status": "non_consumable_parent_failed_before_probe_submission",
        },
        "superseded_frozen_lock_bundle_release": {
            "path": str(V6_RELEASE_PATH.relative_to(REPO_ROOT)),
            "file_sha256": V6_RELEASE_FILE_SHA256,
            "release_sha256": V6_RELEASE_SHA256,
            "failure_marker_path": str(V6_FAILURE_PATH.relative_to(REPO_ROOT)),
            "failure_marker_sha256": file_sha256(V6_FAILURE_PATH),
            "status": "non_consumable_parent_build_failed_missing_local_lock_dependency",
        },
        "superseded_worker_adapter_release": {
            "path": str(V7_RELEASE_PATH.relative_to(REPO_ROOT)),
            "file_sha256": V7_RELEASE_FILE_SHA256,
            "release_sha256": V7_RELEASE_SHA256,
            "failure_marker_path": str(V7_FAILURE_PATH.relative_to(REPO_ROOT)),
            "failure_marker_sha256": file_sha256(V7_FAILURE_PATH),
            "status": "non_consumable_workers_failed_before_result_materialization",
        },
        "historical_runtime": {
            "recorded_clean_commit": RECORDED_CLEAN_COMMIT,
            "historical_library_source_commit": HISTORICAL_RUNTIME_COMMIT,
            "source_manifest": {
                "path": str(RUNTIME_STACK_MANIFEST_PATH.relative_to(REPO_ROOT)),
                "sha256": file_sha256(RUNTIME_STACK_MANIFEST_PATH),
                "row_count": len(runtime_stack),
                "excluded_packaging_paths": sorted(HISTORICAL_RUNTIME_EXCLUDED_PATHS),
                "lock_local_sources": lock_inventory["local_sources"],
                "lock_source_kind_counts": lock_inventory["source_kind_counts"],
            },
            "recovery_implementation_manifest_sha256": recovery_implementation_manifest_sha256,
            "requested_task_image": TASK_IMAGE,
            "python_version": EXPECTED_PYTHON_VERSION,
            "required_package_versions": EXPECTED_RUNTIME_VERSIONS,
            "tpu_type": "v5p-8",
            "device_count": EXPECTED_DEVICE_COUNT,
            "parent_reproduction_relative_tolerance": 5e-6,
            "stage1_environment_baseline_path": str(RUNTIME_ENVIRONMENT_BASELINE_PATH.relative_to(REPO_ROOT)),
        },
        "plot_inputs": {
            name: {"path": str(path.relative_to(REPO_ROOT)), "sha256": file_sha256(path)}
            for name, path in sorted(plot_input_paths.items())
        },
        "execution_reference_inputs": {
            name: {"path": str(path.relative_to(REPO_ROOT)), "sha256": file_sha256(path)}
            for name, path in sorted(execution_reference_paths.items())
        },
        "materialization": {
            "tables_are_visualization_only": True,
            "materialize_command": materialize_command,
            "render_command": render_command,
            "plot_module": str(PLOTTER_PATH.relative_to(REPO_ROOT)),
            "plot_output_dir": str(PLOT_OUTPUT_DIR.relative_to(REPO_ROOT)),
            "render_manifest_path": str((PLOT_OUTPUT_DIR / "render_manifest.json").relative_to(REPO_ROOT)),
        },
        "external_review": {
            "path": str(CC_REVIEW_PATH.relative_to(REPO_ROOT)),
            "sha256": file_sha256(CC_REVIEW_PATH),
            "account": "plambdafour@proton.me",
            "model": "claude-opus-5[1m]",
            "verdict": review_verdict,
        },
        "authorization_contract": {
            "confirmation": "I_AUTHORIZE_THE_SAVED_CHECKPOINT_GRADIENT_PLOT_COMPLETION",
            "remote_adapter_canary_required": True,
            "remote_adapter_canary_path": REMOTE_ADAPTER_CANARY_PATH,
        },
        "submission_contract": {
            "required_environment": {"UV_FROZEN": "1"},
            "command_prefix": "UV_FROZEN=1 uv run python -m marin.run.iris_run",
            "runtime_adapter_preflight_command": (
                "uv run python -m experiments.domain_phase_mix.starcoder_wsd80_gradient_plot_completion "
                "--release-sha256 <release_sha256> --mode runtime-adapter-preflight"
            ),
            "remote_adapter_canary_command": (
                "uv run python -m experiments.domain_phase_mix.starcoder_wsd80_gradient_plot_completion "
                "--release-sha256 <release_sha256> --mode remote-adapter-canary"
            ),
            "required_preauthorization_workspace_paths": sorted(
                set(required_stage1_workspace_paths) - {str(FULL_LAUNCH_AUTHORIZATION_PATH.relative_to(REPO_ROOT))}
            ),
            "required_stage1_workspace_paths": required_stage1_workspace_paths,
            "required_stage2_plus_workspace_paths": sorted(
                {
                    *required_stage1_workspace_paths,
                    str(RUNTIME_ENVIRONMENT_BASELINE_PATH.relative_to(REPO_ROOT)),
                }
            ),
            "reason": (
                "The historical uv.lock is part of the numerical source manifest. Without UV_FROZEN, the nested "
                "uv run iris invocation rewrites the lock before Iris bundles the workspace."
            ),
        },
        "manifests": {
            "full": {
                "path": str(FULL_MANIFEST_PATH.relative_to(REPO_ROOT)),
                "sha256": file_sha256(FULL_MANIFEST_PATH),
                "row_count": len(rows),
            },
            "checkpoint_provenance": {
                "path": str(CHECKPOINT_PROVENANCE_PATH.relative_to(REPO_ROOT)),
                "sha256": file_sha256(CHECKPOINT_PROVENANCE_PATH),
                "row_count": len(checkpoint_inventory),
                "checkpoint_count": len({row["checkpoint_uri"] for row in checkpoint_inventory}),
            },
            "parent_result_provenance": {
                "path": str(PARENT_RESULT_PROVENANCE_PATH.relative_to(REPO_ROOT)),
                "sha256": file_sha256(PARENT_RESULT_PROVENANCE_PATH),
                "row_count": len(parent_result_inventory),
                "object_count": len({row["object_uri"] for row in parent_result_inventory}),
            },
        },
        "implementation_files": implementation_files,
        "parent_implementation_files": parent_implementation_files,
        "parent_result_artifact_version": PARENT_ARTIFACT_VERSION,
        "execution_acceptance": EXECUTION_ACCEPTANCE,
        "design_validation": validation,
        "full_launch_stages": {
            str(stage): {"row_count": STAGE_ROW_COUNTS[stage], "max_concurrent": STAGE_MAX_CONCURRENT[stage]}
            for stage in sorted(STAGE_ROW_COUNTS)
        },
    }
    release["release_sha256"] = canonical_sha256({**release, "release_sha256": ""})
    _write_json(RELEASE_PATH, release)
    return release


def main() -> None:
    print(json.dumps(freeze(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
