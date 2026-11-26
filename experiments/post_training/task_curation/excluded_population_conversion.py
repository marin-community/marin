# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert the mechanically-excluded compendium populations into executable TaskSpecs.

The 2026.11.14 mechanical waves retained zero tasks for these sources: every task's
docker build referenced an official Docker Hub base by mutable tag, the contract
derivation had no pin to grant, and the fail-closed oracle filtration excluded the
whole population (``unpinned_environment``, no oracle trial ever ran — the runtime
record is the empty ``{}``). This module repairs exactly that class: digest-pin every
external FROM base through the landed native-conversion resolver, restore the oracle
adapter the laion v4 patch dropped (the gold solution ships as ``solution/solution.txt``
while the static verification requires the executable ``solution/solve.sh``), build the
executable ScriptGrader from the package's own verifier, and publish the pinned
conversion release the admission chain consumes.

The input population is the archived filtration input (the staged compendium
tasks.parquet whose sha the mechanical receipt's evidence records), so the conversion
covers exactly the rows the catalog's exclusion evidence measured.
"""

import argparse
import hashlib
import io
import json
import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
from finestore.rl.licenses import LicenseDecision, LicenseEvidence, LicenseReceipt
from huggingface_hub import HfApi
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import task_spec_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation import native_source_conversion
from experiments.post_training.task_curation.harbor_cohort_conversion import (
    CONVERSION_PROTOCOL,
    CohortConversionRejected,
)
from experiments.post_training.task_curation.harbor_cohort_release import (
    converter_revision,
    release_repo,
)
from experiments.post_training.task_curation.license_binding import (
    LICENSE_RECEIPT_INDEX,
    AppendWouldRewriteSharedShard,
    append_license_receipt,
    read_license_receipts,
)
from experiments.post_training.task_curation.license_publication import (
    check_publication,
    licensed_dataset_card,
    prepare_license_publication,
)
from experiments.post_training.task_curation.license_resolution import LicenseTarget, resolve_license
from experiments.post_training.task_curation.native_source_conversion import (
    BasePinMissing,
    convert_native_source_row,
    load_digest_cache,
    save_digest_cache,
    write_native_conversion_artifacts,
)
from infra.marina.apps.rl_data_catalog.server.licensing import require_unbanned

LOGGER = logging.getLogger(__name__)

ORACLE_ADAPTER = b"""#!/bin/bash
# Oracle adapter: write the archived gold completion to the declared answer path.
# The laion v4 patch ships the gold as solution/solution.txt beside this script and
# grades /app/solution.txt against the tests fixture; the adapter restores the
# executable oracle the static verification requires without inventing content.
set -euo pipefail
mkdir -p /app
cp "$(dirname "$0")/solution.txt" /app/solution.txt
"""


def add_oracle_adapter(task: TaskSpec) -> TaskSpec:
    """Restore the executable oracle adapter the laion v4 patch dropped.

    The v4-patched packages carry the gold completion as ``solution/solution.txt`` and
    grade the agent's ``/app/solution.txt`` against the tests fixture, but ship no
    ``solution/solve.sh`` for the oracle control to execute. The adapter copies the
    packaged gold verbatim; the oracle replay therefore exercises the same grader the
    agent faces with the reference submission.
    """
    paths = {resource.path for resource in task.resources.oracle}
    if "solution/solve.sh" in paths:
        return task
    if "solution/solution.txt" not in paths:
        raise CohortConversionRejected(
            f"{task.id}: the package carries neither an executable oracle nor a solution.txt gold to adapt"
        )
    adapter = inline_resource("solution/solve.sh", ORACLE_ADAPTER).model_copy(update={"mode": "0755"})
    updated = (*task.resources.oracle, adapter)
    return task.model_copy(update={"resources": task.resources.model_copy(update={"oracle": updated})})


def load_population(uri: str, sha256: str, *, limit: int | None = None) -> list[dict]:
    """Download the archived filtration input and verify it byte-exact against the receipt."""
    raw = StoragePath(uri).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != sha256:
        raise ValueError(f"The archived population changed: {uri} sha256 {digest} != {sha256}")
    rows = pq.read_table(io.BytesIO(raw), columns=["task_spec"]).to_pylist()
    if limit is not None:
        rows = rows[:limit]
    return rows


def convert_population_rows(rows: list[dict], resolve: Callable[[str], str]) -> tuple[list[dict], list[dict], dict]:
    """Convert the archived population; strict — any rejection other than a missing base fails."""
    converted: list[dict] = []
    receipts: list[dict] = []
    exclusions: list[dict] = []
    digest_cache = os.environ.get("NATIVE_CONVERSION_DIGEST_CACHE")
    resolutions_at_last_save = len(native_source_conversion._DIGEST_CACHE)
    for row in rows:
        task = TaskSpec.model_validate_json(row["task_spec"])
        adapted = add_oracle_adapter(task)
        try:
            outcome = convert_native_source_row(adapted, input_task_sha256=task_spec_sha256(task), resolve=resolve)
        except BasePinMissing as error:
            exclusions.append(
                {
                    "task_id": task.id,
                    "reason": "base_image_missing",
                    "detail": str(error),
                    "input_task_sha256": task_spec_sha256(adapted),
                }
            )
            continue
        if digest_cache and len(native_source_conversion._DIGEST_CACHE) - resolutions_at_last_save >= 250:
            save_digest_cache(digest_cache)
            resolutions_at_last_save = len(native_source_conversion._DIGEST_CACHE)
        serialized = outcome.task.model_dump_json()
        converted.append(
            {
                "path": task.source.row,
                "task_spec": serialized,
                "task_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
            }
        )
        receipts.append(
            {
                "task_id": task.id,
                "protocol": CONVERSION_PROTOCOL,
                "input_task_sha256": outcome.input_task_sha256,
                "output_task_sha256": outcome.output_task_sha256,
                "changed_fields": ["environment_requirements", "execution_requirements", "grader", "tags", "labels"],
                "base_pins": [{"base": pin.base, "pinned": pin.pinned} for pin in outcome.base_pins],
            }
        )
    counts = {
        "input_rows": len(rows),
        "converted_rows": len(converted),
        "excluded_rows": len(exclusions),
    }
    if counts["converted_rows"] + counts["excluded_rows"] != counts["input_rows"]:
        raise ValueError("The conversion lost rows: converted plus excluded must cover every input row")
    return converted, receipts, {**counts, "exclusions": exclusions}


def _card_evidence(release: str, revision: str, card_bytes: bytes) -> LicenseEvidence:
    """The release card's own wrapper declaration, quoting its license_source lines."""
    quoted = [
        line
        for line in card_bytes.decode("utf-8").splitlines()
        if line.startswith(("license_source:", "  dataset_id:", "  revision:", "  url:"))
    ][:6]
    if not quoted:
        raise CohortConversionRejected("The release card declares no license_source to bind")
    return LicenseEvidence(
        kind="wrapper_source",
        url=f"https://huggingface.co/datasets/{release}/resolve/{revision}/README.md",
        sha256=hashlib.sha256(card_bytes).hexdigest(),
        license_ids=[],
        quoted_lines=quoted,
        restrictive_terms=False,
        unidentified_third_party_terms=False,
    )


def repin_license_receipt(previous: LicenseReceipt, release: str, revision: str, card_bytes: bytes) -> LicenseReceipt:
    """Re-pin the source's existing license receipt onto the pinned conversion release.

    The conversion derives from the archived filtration input of the receipt's own
    dataset, so the resolved upstream must not move: it is re-resolved through the
    checker and compared against the recorded pin (fail-closed drift). Recorded
    decisions carry their original lineage per the repackager rule.
    """
    upstream = previous.resolved_source
    if upstream is None:
        raise CohortConversionRejected(f"{previous.source_id}: the existing receipt resolves no upstream")
    fresh = resolve_license(LicenseTarget(previous.source_id, upstream.dataset_id, upstream.revision, upstream.url))
    if fresh.resolved_source is None or fresh.resolved_source.revision != upstream.revision:
        raise CohortConversionRejected(
            f"{previous.source_id}: the recorded upstream {upstream.dataset_id}@{upstream.revision[:12]} "
            "no longer resolves to the recorded source pin"
        )
    carried = []
    for decision in previous.decisions:
        carried.append(
            LicenseDecision(
                kind=decision.kind,
                approved=decision.approved,
                who=decision.who,
                when=datetime.now(UTC).isoformat(),
                source_id=previous.source_id,
                dataset_id=release,
                revision=revision,
                scope=list(decision.scope),
                rationale=(
                    f"{decision.rationale} Lineage: recorded at {decision.dataset_id}@{decision.revision} "
                    f"({decision.when}); carried to this release pin by the repackager rule."
                ),
                evidence_urls=list(decision.evidence_urls),
                issue_url=decision.issue_url,
            )
        )
    return LicenseReceipt(
        source_id=previous.source_id,
        dataset_id=release,
        revision=revision,
        resolved_at=fresh.resolved_at,
        resolver_version=fresh.resolver_version,
        evidence=[*fresh.evidence, _card_evidence(release, revision, card_bytes)],
        bundled_third_party=previous.bundled_third_party,
        model_generated_outputs=previous.model_generated_outputs,
        decisions=carried,
        resolved_source=upstream,
        wrapper_chain=previous.wrapper_chain,
    )


def publish_pinned_population(
    *,
    source_id: str,
    intended_use: str,
    population_uri: str,
    population_sha256: str,
    declared_release: dict,
    output_root: Path,
    artifact_version: str,
    converter_revision: str,
    converted: list[dict],
    receipts: list[dict],
    counts: dict,
) -> dict:
    """Publish the pinned conversion release and bind its license receipt."""
    manifest = write_native_conversion_artifacts(
        converted, receipts, counts, output_root=str(output_root), source_id=source_id
    )
    manifest["converter"] = "experiments.post_training.task_curation.excluded_population_conversion"
    manifest["converter_revision"] = converter_revision
    manifest["mode"] = "publish"
    manifest["declared_release"] = declared_release
    manifest["input_population"] = {"uri": population_uri, "sha256": population_sha256}
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    receipts_by_source = read_license_receipts(LICENSE_RECEIPT_INDEX)
    previous = receipts_by_source.get(source_id)
    if previous is None:
        raise CohortConversionRejected(f"{source_id}: no license receipt to re-pin for the conversion release")
    require_unbanned(source_id, previous.dataset_id, previous)

    api = HfApi()
    repo = release_repo(source_id, artifact_version)
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(output_root),
        commit_message="Publish the base-pinned conversion release",
    )
    staged_revision = api.repo_info(repo, repo_type="dataset").sha
    card_path = Path(api.hf_hub_download(repo, "README.md", repo_type="dataset", revision=staged_revision))
    bound = repin_license_receipt(previous, repo, staged_revision, card_path.read_bytes())
    try:
        index_commit: str | None = append_license_receipt(
            LICENSE_RECEIPT_INDEX,
            bound,
            message=f"Bind {source_id}'s base-pinned conversion release to its original upstream license",
        )
    except AppendWouldRewriteSharedShard as error:
        LOGGER.warning("receipt append refused for %s: %s", source_id, error)
        index_commit = None
    rebind_root = output_root.parent / f"{output_root.name}-rebind"
    if rebind_root.exists():
        import shutil  # noqa: PLC0415

        shutil.rmtree(rebind_root)
    publication = prepare_license_publication(bound, rebind_root, use=intended_use)
    body = card_path.read_text(encoding="utf-8").split("---", 2)[-1].strip()
    (rebind_root / "README.md").write_text(
        licensed_dataset_card(body, publication, pretty_name="Base-pinned conversion release")
    )
    card_commit = api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(rebind_root),
        allow_patterns=["README.md"],
        commit_message="Bind the release card to its original upstream license",
    )
    license_check = check_publication(source_id, repo, card_commit.oid, intended_use)
    result = {
        **manifest,
        "release_repo": repo,
        "release_revision": card_commit.oid,
        "license_index_commit": index_commit,
        "license_check_verdict": license_check["verdict"],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True, help="The excluded source id to convert.")
    parser.add_argument(
        "--population-uri",
        required=True,
        help="The archived filtration input tasks.parquet (an s3:// URI or an HF dataset path).",
    )
    parser.add_argument(
        "--population-sha256", required=True, help="The archived population's sha256 from the mechanical receipt."
    )
    parser.add_argument(
        "--declared-release",
        required=True,
        help="JSON object {repo, revision} — the declared release pin the conversion replaces.",
    )
    parser.add_argument("--intended-use", required=True, help="The declared intended use (train or eval).")
    parser.add_argument("--output-root", required=True, help="The conversion output directory (local scratch).")
    parser.add_argument("--artifact-version", required=True, help="The release repo version suffix.")
    parser.add_argument("--limit", type=int, help="Convert only the first N rows (local validation slices).")
    parser.add_argument(
        "--digest-cache",
        help="The durable in-region digest cache (an s3:// object): resolved FROM-base digests "
        "persist here so a rerun skips tags it has already resolved.",
    )
    parser.add_argument(
        "--converter-revision",
        help="The landed revision the conversion runs from; the bootstrap passes the archive revision.",
    )
    parser.add_argument(
        "--run", action="store_true", help="Publish the private release; omit to validate and plan only."
    )
    parser.add_argument("--report", type=Path, help="Write the result JSON here.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.digest_cache:
        seeded = load_digest_cache(args.digest_cache)
        LOGGER.info("digest cache seeded with %d resolutions from %s", seeded, args.digest_cache)
        os.environ["NATIVE_CONVERSION_DIGEST_CACHE"] = args.digest_cache

    rows = load_population(args.population_uri, args.population_sha256, limit=args.limit)
    converted, receipts, counts = convert_population_rows(rows, native_source_conversion.resolve_base_digest)
    if args.digest_cache:
        saved = save_digest_cache(args.digest_cache)
        LOGGER.info("digest cache persisted %d resolutions to %s", saved, args.digest_cache)
    declared_release = json.loads(args.declared_release)
    if not args.run:
        manifest = write_native_conversion_artifacts(
            converted, receipts, counts, output_root=args.output_root, source_id=args.source_id
        )
        manifest["converter"] = "experiments.post_training.task_curation.excluded_population_conversion"
        manifest["converter_revision"] = converter_revision(args.converter_revision)
        manifest["mode"] = "plan"
        manifest["declared_release"] = declared_release
        manifest["input_population"] = {"uri": args.population_uri, "sha256": args.population_sha256}
        (Path(args.output_root) / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps({"plan": manifest, "release_repo": release_repo(args.source_id, args.artifact_version)}, indent=2)
        )
        return
    result = publish_pinned_population(
        source_id=args.source_id,
        intended_use=args.intended_use,
        population_uri=args.population_uri,
        population_sha256=args.population_sha256,
        declared_release=declared_release,
        output_root=Path(args.output_root),
        artifact_version=args.artifact_version,
        converter_revision=converter_revision(args.converter_revision),
        converted=converted,
        receipts=receipts,
        counts=counts,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()
