# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.2",
# ]
# ///
"""Build the focused ChatGPT Pro follow-up packet for Delphi 3e18 modeling."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
OLD_PACKET = SCRIPT_DIR / "chatgpt_pro_two_phase_surrogate_packet_20260721"
TEMPLATES = SCRIPT_DIR / "chatgpt_pro_two_phase_surrogate_followup_packet_20260723_src"
PACKET = SCRIPT_DIR / "chatgpt_pro_two_phase_surrogate_followup_packet_20260723"
ZIP_PATH = PACKET.with_suffix(".zip")
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
HELDOUT_SOURCE = REFERENCE_OUTPUTS / "delphi_3e18_append_only_heldouts_20260714" / "heldout_current.csv"
SYNTHESIS_CODE = SCRIPT_DIR / "cross_session_phase_transport_20260723"
SESSION_ROOT = Path.home() / "Downloads" / "chatgpt_pro_sessions"

TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".txt"}
NOISE_NAMES = {".DS_Store", "__pycache__"}
PRIVATE_GCS_PREFIX = re.compile(r"gs://marin-[^/]+/[^/]+/")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
SECRET_ASSIGNMENT = re.compile(r"(?i)(token|secret|password|api[_-]?key)\s*=\s*[^\s,;]+")
PRIVATE_USERNAME = re.compile(re.escape(Path.home().name), re.IGNORECASE)

SESSION_ARCHIVES = {
    "session_1": SESSION_ROOT / "1_chatgpt_pro_two_phase_surrogate_review_package.zip",
    "session_2": SESSION_ROOT / "2_chatgpt_pro_two_phase_surrogate_progress_review.zip",
    "session_3": SESSION_ROOT / "3_chatgpt_pro_two_phase_surrogate_solution.zip",
    "session_4": SESSION_ROOT / "4_chatgpt_pro_two_phase_surrogate_solution.zip",
    "session_5": SESSION_ROOT / "5_chatgpt_pro_two_phase_surrogate_all_progress_review.zip",
}

SESSION_FILES = {
    "session_1": {
        "REPORT.md": "REPORT.md",
        "APPROACH_REGISTRY.csv": "APPROACH_REGISTRY.csv",
        "39BUCKET_TERMINAL_VERDICT.json": "TERMINAL_VERDICT.json",
    },
    "session_2": {
        ("chatgpt_pro_two_phase_surrogate_progress_review/PROGRESS_REVIEW_REPORT.md"): "REPORT.md",
        ("chatgpt_pro_two_phase_surrogate_progress_review/REVIEW_STATUS.json"): "REVIEW_STATUS.json",
        (
            "chatgpt_pro_two_phase_surrogate_progress_review/continuation_progress/APPROACH_REGISTRY_ADDENDUM.csv"
        ): "APPROACH_REGISTRY_ADDENDUM.csv",
    },
    "session_3": {
        "chatgpt_pro_two_phase_surrogate_solution/REPORT.md": "REPORT.md",
        ("chatgpt_pro_two_phase_surrogate_solution/APPROACH_REGISTRY.csv"): "APPROACH_REGISTRY.csv",
        (
            "chatgpt_pro_two_phase_surrogate_solution/results/acceptance_gate_evaluation.csv"
        ): "ACCEPTANCE_GATE_EVALUATION.csv",
    },
    "session_4": {
        "chatgpt_pro_two_phase_surrogate_solution/REPORT.md": "REPORT.md",
        ("chatgpt_pro_two_phase_surrogate_solution/APPROACH_REGISTRY.csv"): "APPROACH_REGISTRY.csv",
        (
            "chatgpt_pro_two_phase_surrogate_solution/results/acceptance_gate_decision.json"
        ): "ACCEPTANCE_GATE_DECISION.json",
    },
    "session_5": {
        (
            "chatgpt_pro_two_phase_surrogate_all_progress_review/"
            "authoritative_solution/chatgpt_pro_two_phase_surrogate_solution/"
            "REPORT.md"
        ): "REPORT.md",
        (
            "chatgpt_pro_two_phase_surrogate_all_progress_review/"
            "authoritative_solution/chatgpt_pro_two_phase_surrogate_solution/"
            "APPROACH_REGISTRY.csv"
        ): "APPROACH_REGISTRY.csv",
        (
            "chatgpt_pro_two_phase_surrogate_all_progress_review/"
            "authoritative_solution/chatgpt_pro_two_phase_surrogate_solution/"
            "results/candidate_scorecard.csv"
        ): "CANDIDATE_SCORECARD.csv",
        (
            "chatgpt_pro_two_phase_surrogate_all_progress_review/"
            "authoritative_solution/chatgpt_pro_two_phase_surrogate_solution/"
            "docs/PHASE_AMPLITUDE_NONIDENTIFICATION.md"
        ): "PHASE_AMPLITUDE_NONIDENTIFICATION.md",
        (
            "chatgpt_pro_two_phase_surrogate_all_progress_review/"
            "authoritative_solution/chatgpt_pro_two_phase_surrogate_solution/"
            "docs/ALL_SCALE_DIRECTION_NONIDENTIFICATION.md"
        ): "ALL_SCALE_DIRECTION_NONIDENTIFICATION.md",
    },
}

SYNTHESIS_OUTPUT_SELECTION = {
    "cross_session_phase_transport_20260723": (
        "FINAL_SYNTHESIS.md",
        "STRUCTURAL_AUDIT.md",
        "CROSS_SESSION_PACKET_PLAN.md",
        "consolidated_approach_registry.csv",
        "paired_cv_metrics.csv",
        "paired_cv_coefficients.csv",
        "full_fit_parameters.csv",
        "heldout_metrics.csv",
        "exact_fiber_metrics.csv",
        "raw_optimization_audit.csv",
        "raw_optimization_summary.json",
        "raw_optimum_weights.csv",
    ),
    "cross_session_compact_transport_20260723": (
        "report.md",
        "aggregate_shape_search.csv",
        "paired_cv_metrics.csv",
        "full_fit_parameters.csv",
        "heldout_metrics.csv",
        "exact_fiber_metrics.csv",
    ),
    "cross_session_shared_recency_20260723": (
        "report.md",
        "full_fits.csv",
        "paired_cv_metrics.csv",
        "heldout_metrics.csv",
        "exact_fiber_metrics.csv",
        "paired_bootstrap_vs_separate_fpt.csv",
    ),
}

PORTABLE_CODE = (
    "run_phase_transport_synthesis.py",
    "run_compact_transport_batch.py",
    "run_fpt_optimization_audit.py",
    "run_shared_recency_batch.py",
    "compare_transport_batches.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ignored(path: Path) -> bool:
    return any(part in NOISE_NAMES for part in path.parts) or path.suffix == ".pyc"


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        if ignored(path):
            continue
        copy_file(path, destination / path.relative_to(source))


def phase_weights(
    frame: pd.DataFrame,
    column: str,
    domains: list[str],
) -> np.ndarray:
    return np.asarray(
        [[float(json.loads(value)[domain]) for domain in domains] for value in frame[column]],
        dtype=float,
    )


def canonicalize_heldouts(
    source: pd.DataFrame,
    domains: list[str],
) -> pd.DataFrame:
    complete = source.loc[
        source["training_state"].eq("finished") & source["checkpoint_declared_complete"].eq(1)
    ].reset_index(drop=True)
    if len(complete) != len(source):
        raise ValueError("The append-only source contains incomplete observations")
    if not (complete["uncheatable_bpb"].notna() & complete["table9_macro_bpb"].notna()).all():
        raise ValueError("Every exported heldout must have both primary targets")

    phase0 = phase_weights(complete, "phase_0_weights_json", domains)
    phase1 = phase_weights(complete, "phase_1_weights_json", domains)
    phase0 /= phase0.sum(axis=1, keepdims=True)
    phase1 /= phase1.sum(axis=1, keepdims=True)

    output = pd.DataFrame(
        {
            "row_id": complete["heldout_id"].astype(str),
            "uncheatable_bpb": complete["uncheatable_bpb"],
            "table9_macro_bpb": complete["table9_macro_bpb"],
            "policy_class": complete["policy_class"],
            "split": "heldout",
            "training_series": complete["training_series"],
            "proposal_target": complete["proposal_target"],
            "candidate_kind": complete["candidate_kind"],
            "group_id": complete["mixture_sha256"],
            "fit_panel_overlap": complete["fit_panel_overlap"],
            "panel_tag": complete["panel_tag"],
            "candidate_id": complete["candidate_id"],
            "fit_source": complete["fit_source"],
            "aggregate_kl_coefficient": complete["aggregate_kl_coefficient"],
            "phase_information_budget": complete["phase_information_budget"],
            "anchor_id": complete["anchor_id"],
            "direction_id": complete["direction_id"],
            "radius_fraction": complete["radius_fraction"],
            "seed_block": complete["seed_block"],
            "observation_fingerprint": complete["observation_fingerprint"],
            "phase_0_weights_json": [
                json.dumps(
                    dict(zip(domains, row, strict=True)),
                    separators=(",", ":"),
                    sort_keys=True,
                )
                for row in phase0
            ],
            "phase_1_weights_json": [
                json.dumps(
                    dict(zip(domains, row, strict=True)),
                    separators=(",", ":"),
                    sort_keys=True,
                )
                for row in phase1
            ],
        }
    )
    for index, domain in enumerate(domains):
        output[f"phase_0_weight::{domain}"] = phase0[:, index]
        output[f"phase_1_weight::{domain}"] = phase1[:, index]
    return output


def weight_tensor(frame: pd.DataFrame, domains: list[str]) -> np.ndarray:
    phase0 = frame[[f"phase_0_weight::{domain}" for domain in domains]].to_numpy(float)
    phase1 = frame[[f"phase_1_weight::{domain}" for domain in domains]].to_numpy(float)
    return np.stack([phase0, phase1], axis=1)


def build_data() -> dict[str, Any]:
    old_catalog = json.loads((OLD_PACKET / "data/catalog.json").read_text())
    two_spec = dict(old_catalog["datasets"]["delphi_3e18_two_phase_fit"])
    domains = list(two_spec["domains"])
    for dataset in ("delphi_3e18_one_phase_fit", "delphi_3e18_two_phase_fit"):
        copy_file(
            OLD_PACKET / f"data/canonical/{dataset}.csv",
            PACKET / f"data/canonical/{dataset}.csv",
        )

    one = pd.read_csv(PACKET / "data/canonical/delphi_3e18_one_phase_fit.csv")
    two = pd.read_csv(PACKET / "data/canonical/delphi_3e18_two_phase_fit.csv")
    if len(one) != 280 or len(two) != 280:
        raise ValueError("Expected exactly 280 paired one-phase/two-phase rows")
    if not np.array_equal(one["group_id"].astype(str), two["group_id"].astype(str)):
        raise ValueError("Paired fit rows are not aligned by group_id")

    c0 = np.asarray(two_spec["c0"], dtype=float)
    c1 = np.asarray(two_spec["c1"], dtype=float)
    alpha0_by_bucket = c0 / (c0 + c1)
    alpha0 = float(np.median(alpha0_by_bucket))
    if np.max(np.abs(alpha0_by_bucket - alpha0)) > 1e-8:
        raise ValueError("Delphi phase fraction is not constant across buckets")
    alpha1 = 1.0 - alpha0
    one_weights = weight_tensor(one, domains)
    two_weights = weight_tensor(two, domains)
    if not np.allclose(one_weights[:, 0], one_weights[:, 1], atol=5e-10):
        raise ValueError("One-phase fit rows are not tied")
    aggregate = alpha0 * two_weights[:, 0] + alpha1 * two_weights[:, 1]
    aggregate_error = np.max(np.abs(aggregate - one_weights[:, 0]), axis=1)
    if float(np.max(aggregate_error)) > 5e-10:
        raise ValueError("Fit rows are not exact aggregate-matched pairs")

    pair_index = pd.DataFrame(
        {
            "pair_id": one["group_id"].astype(str),
            "one_phase_row_id": one["row_id"].astype(str),
            "two_phase_row_id": two["row_id"].astype(str),
            "aggregate_max_abs_error": aggregate_error,
            "phase_tv": 0.5 * np.sum(np.abs(two_weights[:, 1] - two_weights[:, 0]), axis=1),
            "uncheatable_one_phase_bpb": one["uncheatable_bpb"],
            "uncheatable_two_phase_bpb": two["uncheatable_bpb"],
            "uncheatable_phase_delta": (two["uncheatable_bpb"] - one["uncheatable_bpb"]),
            "table9_one_phase_bpb": one["table9_macro_bpb"],
            "table9_two_phase_bpb": two["table9_macro_bpb"],
            "table9_phase_delta": (two["table9_macro_bpb"] - one["table9_macro_bpb"]),
        }
    )
    pair_index.to_csv(PACKET / "data/delphi_3e18_pair_index.csv", index=False)

    heldout_source = pd.read_csv(HELDOUT_SOURCE, low_memory=False)
    heldout = canonicalize_heldouts(heldout_source, domains)
    heldout.to_csv(
        PACKET / "data/canonical/delphi_3e18_heldouts.csv",
        index=False,
    )

    catalog = {
        "schema_version": 2,
        "scope": "Delphi 3e18 39-bucket one-phase/two-phase surrogate modeling",
        "datasets": {},
    }
    for dataset, frame, policy_class in (
        ("delphi_3e18_one_phase_fit", one, "one_phase"),
        ("delphi_3e18_two_phase_fit", two, "two_phase"),
        ("delphi_3e18_heldouts", heldout, "mixed"),
    ):
        spec = dict(old_catalog["datasets"][dataset])
        spec["path"] = f"data/canonical/{dataset}.csv"
        spec["row_count"] = len(frame)
        spec["policy_class"] = policy_class
        spec["targets"] = ["uncheatable_bpb", "table9_macro_bpb"]
        spec["target_coverage"] = {target: int(frame[target].notna().sum()) for target in spec["targets"]}
        catalog["datasets"][dataset] = spec
    (PACKET / "data/catalog.json").write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    return {
        "domains": domains,
        "alpha0": alpha0,
        "alpha1": alpha1,
        "heldout_rows": len(heldout),
        "heldout_unique_coordinates": int(heldout["group_id"].nunique()),
        "heldout_two_phase_rows": int(heldout["policy_class"].eq("two_phase").sum()),
        "heldout_tied_rows": int(heldout["policy_class"].eq("single_phase_tied").sum()),
        "heldout_series": int(heldout["training_series"].nunique()),
        "pair_max_aggregate_error": float(np.max(aggregate_error)),
    }


def copy_synthesis_evidence() -> None:
    for directory, filenames in SYNTHESIS_OUTPUT_SELECTION.items():
        source = REFERENCE_OUTPUTS / directory
        destination = PACKET / "evidence" / directory
        for filename in filenames:
            copy_file(source / filename, destination / filename)

    prior = REFERENCE_OUTPUTS / "mechanistic_surrogate_discovery_20260719"
    copy_file(
        prior / "approach_registry.csv",
        PACKET / "evidence/prior_local_search/approach_registry.csv",
    )
    for filename in ("final_report.md", "acceptance_gate_evaluation.csv"):
        copy_file(
            prior / "final_synthesis" / filename,
            PACKET / "evidence/prior_local_search" / filename,
        )


def extract_session_evidence() -> dict[str, Any]:
    sources: dict[str, Any] = {}
    for session, archive_path in SESSION_ARCHIVES.items():
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        sources[session] = {
            "archive_name": archive_path.name,
            "archive_sha256": sha256(archive_path),
            "extracted_files": [],
        }
        with zipfile.ZipFile(archive_path) as archive:
            available = set(archive.namelist())
            for source_name, destination_name in SESSION_FILES[session].items():
                if source_name not in available:
                    raise FileNotFoundError(f"{source_name} is absent from {archive_path}")
                destination = PACKET / "evidence" / "prior_sessions" / session / destination_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(source_name))
                sources[session]["extracted_files"].append(
                    str(destination.relative_to(PACKET / "evidence" / "prior_sessions"))
                )
    (PACKET / "evidence/prior_sessions/SOURCE_ARCHIVES.json").write_text(
        json.dumps(sources, indent=2, sort_keys=True) + "\n"
    )
    return sources


def portable_source(filename: str) -> str:
    text = (SYNTHESIS_CODE / filename).read_text()
    if filename == "run_phase_transport_synthesis.py":
        old = """HERE = Path(__file__).resolve().parent
TWO_PHASE_MANY = HERE.parent
PACKET = TWO_PHASE_MANY / "chatgpt_pro_two_phase_surrogate_packet_20260721"
HELDOUT_PATH = (
    TWO_PHASE_MANY
    / "reference_outputs"
    / "delphi_3e18_append_only_heldouts_20260714"
    / "heldout_current.csv"
)
OUTPUT = (
    TWO_PHASE_MANY
    / "reference_outputs"
    / "cross_session_phase_transport_20260723"
)
"""
        new = """HERE = Path(__file__).resolve().parent
PACKET = HERE.parent
HELDOUT_PATH = PACKET / "data" / "canonical" / "delphi_3e18_heldouts.csv"
OUTPUT = PACKET / "reproduced" / "cross_session_phase_transport"
"""
        if old not in text:
            raise ValueError(f"Portable path block drifted in {filename}")
        return text.replace(old, new)
    replacements = {
        "run_compact_transport_batch.py": (
            """OUTPUT = (
    HERE.parent
    / "reference_outputs"
    / "cross_session_compact_transport_20260723"
)
""",
            """OUTPUT = PACKET_ROOT / "reproduced" / "cross_session_compact_transport"
""",
            "HERE = Path(__file__).resolve().parent\n",
            ("HERE = Path(__file__).resolve().parent\nPACKET_ROOT = HERE.parent\n"),
        ),
        "run_fpt_optimization_audit.py": (
            """OUTPUT = (
    HERE.parent
    / "reference_outputs"
    / "cross_session_phase_transport_20260723"
)
""",
            """OUTPUT = HERE.parent / "reproduced" / "cross_session_phase_transport"
""",
            "",
            "",
        ),
        "run_shared_recency_batch.py": (
            """OUTPUT = (
    HERE.parent
    / "reference_outputs"
    / "cross_session_shared_recency_20260723"
)
""",
            """OUTPUT = HERE.parent / "reproduced" / "cross_session_shared_recency"
""",
            "",
            "",
        ),
        "compare_transport_batches.py": (
            'REFERENCE_OUTPUTS = HERE.parent / "reference_outputs"\n',
            'REFERENCE_OUTPUTS = HERE.parent / "reproduced"\n',
            "",
            "",
        ),
    }
    old, new, anchor, replacement = replacements[filename]
    if old not in text:
        raise ValueError(f"Portable output block drifted in {filename}")
    text = text.replace(old, new)
    if anchor:
        if anchor not in text:
            raise ValueError(f"Portable anchor drifted in {filename}")
        text = text.replace(anchor, replacement, 1)
    text = text.replace(
        "cross_session_phase_transport_20260723",
        "cross_session_phase_transport",
    )
    text = text.replace(
        "cross_session_shared_recency_20260723",
        "cross_session_shared_recency",
    )
    return text


def copy_standalone_code() -> None:
    old_code = OLD_PACKET / "standalone_code"
    for filename in (
        "reference_models.py",
        "reproduce_fit.py",
        "fit_olmix.py",
        "inspect_packet.py",
    ):
        copy_file(old_code / filename, PACKET / "standalone_code" / filename)
    for filename in PORTABLE_CODE:
        destination = PACKET / "standalone_code" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(portable_source(filename))
    for filename in ("PROTOCOL.json", "BATCH_2_PROTOCOL.json", "BATCH_3_PROTOCOL.json"):
        copy_file(
            SYNTHESIS_CODE / filename,
            PACKET / "protocol" / filename,
        )


def compose_prompts() -> None:
    common = (PACKET / "prompts/PROMPT_SHARED_CORE.md").read_text().strip()
    assignment_root = PACKET / "prompts/assignments"
    ready_root = PACKET / "prompts/ready_to_send"
    ready_root.mkdir(parents=True, exist_ok=True)
    for index in range(1, 6):
        assignment = (assignment_root / f"SESSION_{index}_ASSIGNMENT.md").read_text().strip()
        prompt = (
            f"{common}\n\n"
            "## Your Independent Assignment\n\n"
            f"{assignment}\n\n"
            "## Required Return\n\n"
            "Return one complete self-contained ZIP containing the report, executable "
            "code, frozen protocols, metric tables, row-level predictions for promoted "
            "candidates, approach-registry update, data-use ledger update, raw and "
            "regularized optimum policies, and a short continuation handoff. State "
            "plainly if no candidate survives."
        )
        (ready_root / f"SESSION_{index}_FOLLOWUP_PROMPT.md").write_text(prompt + "\n")


def sanitize_text(text: str) -> str:
    redacted = text.replace(str(REPO_ROOT), "<MARIN_REPO>")
    redacted = redacted.replace(str(Path.home()), "<USER_HOME>")
    redacted = PRIVATE_GCS_PREFIX.sub("gs://<redacted-private-prefix>/", redacted)
    redacted = EMAIL.sub("<REDACTED_EMAIL>", redacted)
    redacted = PRIVATE_USERNAME.sub("<REDACTED_USER>", redacted)
    return SECRET_ASSIGNMENT.sub(
        lambda match: match.group(1) + "=<REDACTED>",
        redacted,
    )


def sanitize_packet() -> None:
    for path in PACKET.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(errors="ignore")
        redacted = sanitize_text(text)
        if redacted != text:
            path.write_text(redacted)


def audit_packet() -> None:
    failures: list[str] = []
    forbidden = (
        "/Users/",
        "pinlin_calvin_xu",
        "plambdafour",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "WANDB_API_KEY",
        "HF_TOKEN",
    )
    for path in PACKET.rglob("*"):
        if ignored(path):
            failures.append(f"noise file: {path.relative_to(PACKET)}")
            continue
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(errors="ignore")
        for value in forbidden:
            if value in text:
                failures.append(f"forbidden text {value!r}: {path.relative_to(PACKET)}")
        if EMAIL.search(text):
            failures.append(f"email address: {path.relative_to(PACKET)}")
        if PRIVATE_GCS_PREFIX.search(text):
            failures.append(f"private GCS prefix: {path.relative_to(PACKET)}")
    if failures:
        raise RuntimeError("Packet privacy audit failed:\n" + "\n".join(failures[:100]))


def write_metadata(
    data_summary: dict[str, Any],
    session_sources: dict[str, Any],
) -> None:
    metadata = {
        "packet": PACKET.name,
        "schema_version": 1,
        "primary_scope": "39-bucket Delphi 3e18 one-/two-phase surrogate modeling",
        "fit_pair_count": 280,
        **data_summary,
        "session_archives": {
            session: {
                "archive_name": record["archive_name"],
                "archive_sha256": record["archive_sha256"],
            }
            for session, record in session_sources.items()
        },
        "evidence_status": (
            "All target outcomes in this packet are exposed development evidence. "
            "No untouched confirmation outcome is included."
        ),
        "privacy": (
            "Local paths, private storage prefixes, account identifiers, email "
            "addresses, and credential references are removed."
        ),
    }
    (PACKET / "PACKET_METADATA.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def write_manifest() -> None:
    excluded = {"MANIFEST.json", "CHECKSUMS.sha256"}
    files = []
    for path in sorted(item for item in PACKET.rglob("*") if item.is_file()):
        relative = str(path.relative_to(PACKET))
        if relative in excluded:
            continue
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "packet": PACKET.name,
        "schema_version": 1,
        "build_script": str(Path(__file__).relative_to(REPO_ROOT)),
        "file_count": len(files),
        "files": files,
    }
    (PACKET / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    checksums = [f"{record['sha256']}  {record['path']}" for record in files]
    (PACKET / "CHECKSUMS.sha256").write_text("\n".join(checksums) + "\n")


def build_zip() -> None:
    ZIP_PATH.unlink(missing_ok=True)
    with zipfile.ZipFile(
        ZIP_PATH,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(item for item in PACKET.rglob("*") if item.is_file()):
            archive.write(
                path,
                Path(PACKET.name) / path.relative_to(PACKET),
            )


def main() -> None:
    for required in (OLD_PACKET, TEMPLATES, HELDOUT_SOURCE, SYNTHESIS_CODE):
        if not required.exists():
            raise FileNotFoundError(required)
    if PACKET.exists():
        shutil.rmtree(PACKET)
    copy_tree(TEMPLATES, PACKET)
    data_summary = build_data()
    copy_synthesis_evidence()
    session_sources = extract_session_evidence()
    copy_standalone_code()
    compose_prompts()
    sanitize_packet()
    write_metadata(data_summary, session_sources)
    audit_packet()
    write_manifest()
    build_zip()
    print(f"Built {PACKET}")
    print(f"Archive {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1024 / 1024:.1f} MiB)")
    print(json.dumps(data_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
