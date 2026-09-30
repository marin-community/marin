# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble the reviewed mixed alpha dataset beside its regional TaskTrove inputs.

The command writes a public-only, publication-ready artifact to regional object
storage. Upload to Hugging Face is a separate operation after inspecting its
manifest and card.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.importers.nemo_workplace import (
    DATASET,
    DATASET_REVISION,
    DATASET_SPLIT_MAX_BYTES,
    DATASET_SPLIT_SHA256,
    PROVIDER,
    PROVIDER_GIT_REVISION,
    workplace_environment_config,
)
from taskcompendium.importers.nemo_workplace import (
    IMPORTER_REVISION as WORKPLACE_IMPORTER_REVISION,
)
from taskcompendium.mixed_release import (
    CohortInput,
    HarborSample,
    ProviderPin,
    ReleaseReview,
    SourceAsset,
    SourceRights,
    assemble_mixed_candidate,
    finalize_mixed_candidate,
)
from taskcompendium.models import SCHEMA_VERSION
from taskcompendium.provider_sources import stage_git_provider
from taskcompendium.public_release import build_workplace_candidate
from taskcompendium.release_common import sha256_file

PROJECTION_MANIFEST_SHA256 = "fd1035c795393bf7b43977eee7bf48bba61bf126791b44fe37303a80b58ba6e8"
PROJECTION_MANIFEST_URI = (
    "s3://marin-us-east-02a/marin/taskcompendium/tasktrove/2026.09.18.3/"
    "rights-cleared-projection-ee389a0-2026-09-30/manifest.json"
)
RELEASE_PREFIX = "s3://marin-us-east-02a/marin/taskcompendium/releases/"
TASKTROVE_SOURCE = "s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.18.3"
TASKTROVE_IMPORTER_REVISION = "taskcompendium-tasktrove-v0.3"
TASKTROVE_ACCEPTED_SCHEMA = "0.13"
PROJECTION_BUILDER_REVISION = "ee389a0e645fa086cdbd49a5c95e30c1516c014e"
WORKPLACE_HARBOR_EVIDENCE = "https://github.com/marin-community/marin/pull/9523#issuecomment-5906493281"
TASKTROVE_HARBOR_EVIDENCE = "https://github.com/marin-community/marin/pull/9593"
TASKCOMPENDIUM_WORKPLACE_TRIAL_REVISION = "6a67b2666cd219b55c9d08db8659a7e91b23f08a"
TASKCOMPENDIUM_TASKTROVE_TRIAL_REVISION = "ac621edd1d7a148f5962e450315bcd252a6e4c1d"
HARBOR_REVISION = "ef1eaf207f41f84eb53d17f9c8bd84c073a9aba5"
WORKPLACE_CARD_REVISION = DATASET_REVISION
MCQA_CARD_REVISION = "5d35ead3ba07abda719b3d24f6f395fee8108efd"
PRISM_CARD_REVISION = "8a35a0602167ad1f1ec9d6db5e72281e486738a7"


def _copy_pinned(source: str, destination: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    with StoragePath(source).open("rb") as input_stream, destination.open("wb") as output_stream:
        for chunk in iter(lambda: input_stream.read(1024 * 1024), b""):
            digest.update(chunk)
            output_stream.write(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"Source digest mismatch for {source}")


def _download_workplace(split: str, destination: Path) -> None:
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{DATASET_REVISION}/{split}.jsonl"
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(DATASET_SPLIT_MAX_BYTES + 1)
    if len(data) > DATASET_SPLIT_MAX_BYTES or hashlib.sha256(data).hexdigest() != DATASET_SPLIT_SHA256[split]:
        raise ValueError(f"Pinned Workplace {split} file changed")
    destination.write_bytes(data)


def _workplace_cohorts(source_dir: Path, builder_revision: str, provider_source: Path) -> tuple[CohortInput, ...]:
    candidate = source_dir / "workplace"
    build_workplace_candidate(
        source_dir / "train.jsonl",
        source_dir / "validation.jsonl",
        candidate,
        provider_source=provider_source,
        builder_revision=builder_revision,
    )
    provider = workplace_environment_config(provider_source).tool_providers["workplace"]
    provider_pin = ProviderPin(
        name="workplace",
        locator=provider.provider,
        provider_revision=PROVIDER_GIT_REVISION,
        action_interface=provider.action_interface,
        seed_sha256=provider.seed_sha256,
        tools_sha256=provider.tools_sha256,
        tools=tuple(provider.tools),
    )
    sample = HarborSample(
        evidence_url=WORKPLACE_HARBOR_EVIDENCE,
        taskcompendium_revision=TASKCOMPENDIUM_WORKPLACE_TRIAL_REVISION,
        harbor_revision=HARBOR_REVISION,
        trials=12,
        coverage="both pinned splits, five categories, no-op cases, and deliberate wrong state",
    )
    rights = SourceRights(
        license="cc-by-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        attribution="NVIDIA Corporation, Nemotron-RL-agent-workplace_assistant",
        source_card_url=f"https://huggingface.co/datasets/{DATASET}/blob/{WORKPLACE_CARD_REVISION}/README.md",
        source_card_revision=WORKPLACE_CARD_REVISION,
        change_notice="Converted source rows to agent-visible tool tasks; private gold actions and state were excluded.",
    )
    rows = {"train": 1255, "validation": 545}
    return tuple(
        CohortInput(
            config="workplace",
            cohort=split,
            split=split,
            record_format="public_task",
            input_path=candidate / "data" / f"{split}.jsonl",
            input_sha256=sha256_file(candidate / "data" / f"{split}.jsonl"),
            accepted_rows=rows[split],
            source_records=rows[split],
            parsed_rows=rows[split],
            source_dataset=DATASET,
            source_revision=DATASET_REVISION,
            source_assets=(SourceAsset(path=f"{split}.jsonl", pin=f"sha256:{DATASET_SPLIT_SHA256[split]}"),),
            task_spec_schema=SCHEMA_VERSION,
            importer_revision=WORKPLACE_IMPORTER_REVISION,
            projection_builder_revision=builder_revision,
            rights=rights,
            harbor_samples=(sample,),
            provider_pins=(provider_pin,),
        )
        for split in ("train", "validation")
    )


def _tasktrove_cohorts(source_dir: Path, manifest: dict[str, Any]) -> tuple[CohortInput, ...]:
    cohorts = (
        (
            "laion__nemotron-gym-knowledge-mcqa-v2",
            "qa-short-answer",
            MCQA_CARD_REVISION,
            "nvidia/Nemotron-RL-knowledge-mcqa",
        ),
        ("laion__nemo-prism-math-v3", "math-answer", PRISM_CARD_REVISION, "nvidia/Nemotron-PrismMath"),
    )
    result = []
    for subset, category, card_revision, card_dataset in cohorts:
        key = f"{subset}/{category}/tasks"
        output = manifest["outputs"][key]
        details = manifest["cohort_counts"][key]
        local = source_dir / f"{subset}.jsonl"
        _copy_pinned(output["uri"], local, output["sha256"])
        if output["rows"] != details["accepted_public_records"]:
            raise ValueError(f"Projection row counts disagree for {key}")
        assets = tuple(
            SourceAsset(path=asset["path"], pin=asset["input_object_pin"]) for asset in details["source_assets"]
        )
        change = "Converted source rows to agent-visible answer tasks; answers and verifier material were excluded."
        if category == "qa-short-answer":
            change += (
                " The source card describes synthetic questions informed by books, articles, and OpenScienceReasoning-2."
            )
        else:
            change += " The source card describes synthetic Qwen-derived math and notes Qwen redistribution conditions."
        result.append(
            CohortInput(
                config="tasktrove_clean",
                cohort="mcqa" if category == "qa-short-answer" else "prism_math",
                split="train",
                record_format="accepted_public_record",
                input_path=local,
                input_sha256=output["sha256"],
                accepted_rows=output["rows"],
                source_records=details["eligible_mode_rows_in_tasks_split"],
                parsed_rows=details["converter_completed_rows"],
                source_dataset=TASKTROVE_SOURCE,
                source_subset=subset,
                source_category=category,
                projection_manifest_sha256=PROJECTION_MANIFEST_SHA256,
                source_assets=assets,
                task_spec_schema=TASKTROVE_ACCEPTED_SCHEMA,
                importer_revision=TASKTROVE_IMPORTER_REVISION,
                projection_builder_revision=PROJECTION_BUILDER_REVISION,
                rights=SourceRights(
                    license="cc-by-4.0",
                    license_url="https://creativecommons.org/licenses/by/4.0/",
                    attribution=f"NVIDIA Corporation, {card_dataset}",
                    source_card_url=f"https://huggingface.co/datasets/{card_dataset}/blob/{card_revision}/README.md",
                    source_card_revision=card_revision,
                    change_notice=change,
                ),
                harbor_samples=(
                    HarborSample(
                        evidence_url=TASKTROVE_HARBOR_EVIDENCE,
                        taskcompendium_revision=TASKCOMPENDIUM_TASKTROVE_TRIAL_REVISION,
                        harbor_revision=HARBOR_REVISION,
                        trials=2,
                        coverage="one accepted-output specimen with correct and deliberate wrong replies",
                    ),
                ),
            )
        )
    return tuple(result)


def _upload_regional(source: Path, destination: str) -> dict[str, str]:
    uploaded = {}
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        uri = f"{destination.rstrip('/')}/{relative}"
        with path.open("rb") as input_stream, StoragePath(uri).open("wb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
        uploaded[relative] = sha256_file(path)
    return uploaded


def _stage_cohorts(root: Path, builder_revision: str, provider_source: Path) -> tuple[CohortInput, ...]:
    projection_bytes = StoragePath(PROJECTION_MANIFEST_URI).read_bytes()
    if hashlib.sha256(projection_bytes).hexdigest() != PROJECTION_MANIFEST_SHA256:
        raise ValueError("Projection manifest digest mismatch")
    projection = json.loads(projection_bytes)
    for split in ("train", "validation"):
        _download_workplace(split, root / f"{split}.jsonl")
    return _workplace_cohorts(root, builder_revision, provider_source) + _tasktrove_cohorts(root, projection)


def _build_ready_artifact(
    root: Path, cohorts: tuple[CohortInput, ...], builder_revision: str, rights_url: str
) -> tuple[Path, str]:
    candidate = assemble_mixed_candidate(cohorts, root / "candidate", builder_revision=builder_revision)
    review = ReleaseReview(
        candidate_manifest_sha256=sha256_file(candidate / "manifest.json"),
        rights_review_url=rights_url,
        harbor_evidence_urls=(WORKPLACE_HARBOR_EVIDENCE, TASKTROVE_HARBOR_EVIDENCE),
    )
    ready = finalize_mixed_candidate(candidate, root / "ready", review)
    return ready, review.candidate_manifest_sha256


def _write_assembly_report(ready: Path, prefix: str, candidate_sha256: str, uploaded: dict[str, str]) -> None:
    summary = {
        "output_prefix": prefix,
        "candidate_manifest_sha256": candidate_sha256,
        "files": uploaded,
        "data_files": json.loads((ready / "manifest.json").read_text())["data_files"],
    }
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "assembly-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    shutil.copy2(ready / "manifest.json", output / "manifest.json")
    shutil.copy2(ready / "README.md", output / "README.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the regional TaskCompendium alpha-1 public artifact")
    parser.add_argument("--output-prefix", required=True, help="New regional S3 prefix for public-only release files")
    parser.add_argument("--builder-revision", required=True, help="Full commit of the executed package bundle")
    parser.add_argument("--workplace-provider-checkout", type=Path, required=True)
    parser.add_argument("--rights-review-url", required=True)
    args = parser.parse_args()
    if not args.output_prefix.startswith(RELEASE_PREFIX) or args.output_prefix.rstrip("/") == RELEASE_PREFIX.rstrip("/"):
        raise ValueError("Output must be a new regional TaskCompendium release prefix")
    if StoragePath(f"{args.output_prefix.rstrip('/')}/manifest.json").exists():
        raise FileExistsError(f"Regional release already exists: {args.output_prefix}")
    with tempfile.TemporaryDirectory(prefix="taskcompendium-alpha-") as directory:
        root = Path(directory)
        provider_source = root / "provider"
        stage_git_provider(PROVIDER, args.workplace_provider_checkout, provider_source)
        cohorts = _stage_cohorts(root, args.builder_revision, provider_source)
        ready, candidate_sha256 = _build_ready_artifact(root, cohorts, args.builder_revision, args.rights_review_url)
        uploaded = _upload_regional(ready, args.output_prefix)
        _write_assembly_report(ready, args.output_prefix, candidate_sha256, uploaded)


if __name__ == "__main__":
    main()
