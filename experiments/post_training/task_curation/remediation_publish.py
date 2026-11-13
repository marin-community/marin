# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish frozen combined-retention plans without converting or rewriting TaskSpecs."""

import argparse
import hashlib
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.curation import LexicalUse
from finestore.rl.source_identity import canonical_source_id
from huggingface_hub import HfApi, hf_hub_download
from pydantic import ConfigDict
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.filesystem.transfer import copy
from taskcompendium.evaluation.corpus import file_sha256

from experiments.post_training.task_curation.calibforge_publish import publish
from experiments.post_training.task_curation.datasets.lexical_measurements import LEXICAL_MEASUREMENTS
from experiments.post_training.task_curation.native_quick import source_slug
from experiments.post_training.task_curation.retention_plan import RetentionPlan

LOGGER = logging.getLogger(__name__)
TASK_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("path", pa.string()),
        ("task_spec", pa.large_string()),
        ("task_sha256", pa.string()),
        ("runtime", pa.large_string()),
    ]
)


class ReleasePlan(RetentionPlan):
    model_config = ConfigDict(frozen=True, extra="forbid")

    unique_count: int
    within_source_duplicates: int
    raw_lexical_count: int
    unique_lexical_count: int
    multiplicities: dict[str, int]
    drop_manifest_uri: str
    evidence_url: str
    lexical_use: LexicalUse
    census_evidence_url: str


def retention_repo(plan: ReleasePlan, version: str) -> str:
    slug = source_slug(plan.source_id)[:55].rstrip("-")
    key = hashlib.sha256(plan.source_id.encode()).hexdigest()[:8]
    return f"open-athena/{slug}-{key}-retained-{version}"


def read_drop_ids(plan: ReleasePlan) -> set[str]:
    with StoragePath(plan.combined_drop_manifest_uri).open("rb") as stream:
        table = pq.read_table(stream, columns=["task_id"])
    identifiers = table.column(0).to_pylist()
    if len(identifiers) != len(set(identifiers)) or len(identifiers) != plan.combined_drop_count:
        raise ValueError(f"Frozen combined manifest count differs: {plan.source_id}")
    return set(identifiers)


def filter_population(plan: ReleasePlan, specs_dir: str, output: Path, original: Path | None) -> dict:
    """Select the manifest's exact complement, keeping serialized task and runtime bytes."""
    drops = read_drop_ids(plan)
    seen_drops = set()
    hashes = {}
    raw_count = 0
    shards = []
    with pq.ParquetWriter(output, TASK_SCHEMA, compression="zstd") as writer:
        original_writer = None if original is None else pq.ParquetWriter(original, TASK_SCHEMA, compression="zstd")
        try:
            for path in sorted((StoragePath(specs_dir) / "*.parquet").glob(), key=str):
                shard_count = 0
                with path.open("rb") as stream:
                    for batch in pq.ParquetFile(stream).iter_batches(batch_size=16):
                        selected, all_rows = [], []
                        for row in batch.to_pylist():
                            identifier, encoded = row["id"], row["task_spec"]
                            if json.loads(encoded)["id"] != identifier:
                                raise ValueError(f"Frozen task ID differs: {identifier}")
                            runtime = row["runtime"]
                            digest = hashlib.sha256(encoded.encode()).hexdigest()
                            entry = {
                                "id": identifier,
                                "path": identifier,
                                "task_spec": encoded,
                                "task_sha256": digest,
                                "runtime": runtime,
                            }
                            raw_count += 1
                            shard_count += 1
                            if original_writer is not None:
                                all_rows.append(entry)
                            if identifier in drops:
                                if identifier in seen_drops:
                                    raise ValueError(f"Duplicate frozen task ID: {identifier}")
                                seen_drops.add(identifier)
                                continue
                            if identifier in hashes:
                                raise ValueError(f"Duplicate frozen task ID: {identifier}")
                            hashes[identifier] = (digest, hashlib.sha256(runtime.encode()).hexdigest())
                            selected.append(entry)
                        writer.write_table(pa.Table.from_pylist(selected, schema=TASK_SCHEMA))
                        if original_writer is not None:
                            original_writer.write_table(pa.Table.from_pylist(all_rows, schema=TASK_SCHEMA))
                shards.append({"uri": str(path), "rows": shard_count})
                LOGGER.info("%s: read %d/%d tasks", plan.source_id, raw_count, plan.raw_count)
        finally:
            if original_writer is not None:
                original_writer.close()
    if seen_drops != drops or raw_count != plan.raw_count or len(hashes) != plan.retained_count:
        raise ValueError(f"Frozen selection does not match plan: {plan.source_id}")
    return {"hashes": hashes, "input_shards": shards}


def verify_retained(tasks: Path, expected: dict[str, tuple[str, str]]) -> str:
    """Read every retained task and runtime, rejecting changed, missing or repeated rows."""
    seen = set()
    for batch in pq.ParquetFile(tasks).iter_batches(batch_size=16):
        for row in batch.to_pylist():
            identifier = row["id"]
            digest = hashlib.sha256(row["task_spec"].encode()).hexdigest()
            runtime_digest = hashlib.sha256(row["runtime"].encode()).hexdigest()
            if identifier in seen or expected.get(identifier) != (digest, runtime_digest):
                raise ValueError(f"Retained TaskSpec or runtime differs: {identifier}")
            if row["task_sha256"] != digest or json.loads(row["task_spec"])["id"] != identifier:
                raise ValueError(f"Retained task hash or ID differs: {identifier}")
            seen.add(identifier)
    if seen != expected.keys():
        raise ValueError("Retained population differs from its frozen plan")
    digest = hashlib.sha256()
    for identifier in sorted(expected):
        digest.update(json.dumps([identifier, *expected[identifier]], ensure_ascii=False).encode())
    return digest.hexdigest()


def dataset_card(plan: ReleasePlan, *, population: str, count: int, evidence_root: str) -> str:
    return (
        "---\npretty_name: Atlas TaskCompendium retention release\n---\n\n"
        f"# {plan.source_id}\n\n"
        f"Private TaskCompendium {population}; {count:,} tasks. Use: {plan.lexical_use.value}.\n\n"
        f"Source pin: `{plan.source_dataset_id}@{plan.source_revision}`. "
        f"Converter: `{plan.converter_version}@{plan.converter_revision}`.\n\n"
        f"Raw: {plan.raw_count:,}; exact duplicates: {plan.within_source_duplicates:,}; "
        f"unique lexical flags: {plan.unique_lexical_count:,}; combined drops: {plan.combined_drop_count:,}; "
        f"retained: {plan.retained_count:,}. Training removes the union of duplicates and lexical flags. "
        "Evaluation and benchmark populations remove only duplicates; lexical overlap is measured only.\n\n"
        f"Duplicate manifest: `{plan.drop_manifest_uri}`. "
        f"Combined manifest: `{plan.combined_drop_manifest_uri}`.\n\n"
        f"Frozen input: `{plan.input_artifact}`; declared TaskSpec digest: `{plan.task_specs_sha256}`. "
        f"Plan evidence: `{plan.evidence_url}`; publication evidence: `{evidence_root}`. "
        "The accompanying manifest records the lexical measurement and drop manifest.\n\n"
        "Retained TaskSpec and runtime bytes are unchanged. Readback verifies every retained content hash "
        "and count. This mechanical selection supplies no new oracle, execution or quality qualification. "
        "Upstream licenses and third-party asset terms continue to apply.\n"
    )


def publish_plan(plan: ReleasePlan, version: str, evidence_root: str) -> dict:
    """Publish one approved selection and its immutable regional readback receipt."""
    output = StoragePath(evidence_root) / hashlib.sha256(plan.source_id.encode()).hexdigest()[:20]
    result_path = output / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["plan"] != plan.model_dump(mode="json"):
            raise ValueError(f"Existing receipt has another frozen plan: {plan.source_id}")
        downloaded = hf_hub_download(result["repo"], "tasks.parquet", repo_type="dataset", revision=result["revision"])
        if file_sha256(StoragePath(downloaded)) != result["tasks_parquet_sha256"]:
            raise ValueError(f"Completed release readback differs: {plan.source_id}")
        if not HfApi().repo_info(result["repo"], repo_type="dataset", revision=result["revision"]).private:
            raise ValueError(f"Completed release is not private: {plan.source_id}")
        return result
    metadata = json.loads((StoragePath(plan.input_artifact) / ".artifact.json").read_text())["result"]
    for name, expected in (
        ("dataset_id", plan.source_dataset_id),
        ("dataset_revision", plan.source_revision),
        ("converter_revision", plan.converter_revision),
        ("source_version", plan.converter_version),
        ("task_specs_sha256", plan.task_specs_sha256),
        ("task_count", plan.raw_count),
    ):
        if metadata[name] != expected:
            raise ValueError(f"Frozen input {name} differs: {plan.source_id}")
    if canonical_source_id(metadata["source_id"]) != plan.source_id:
        raise ValueError(f"Frozen source identity differs: {plan.source_id}")
    measurement = LEXICAL_MEASUREMENTS[plan.source_id].measurement.model_dump(mode="json")
    if (measurement["measured_tasks"], measurement["contaminated_tasks"]) != (plan.raw_count, plan.raw_lexical_count):
        raise ValueError(f"Frozen lexical measurement differs: {plan.source_id}")
    with TemporaryDirectory(prefix="atlas-retention-") as scratch:
        root = Path(scratch)
        tasks = root / "tasks.parquet"
        snapshot_required = (
            plan.lexical_use is LexicalUse.TRAIN
            and plan.raw_lexical_count > 0
            and (
                not plan.source_dataset_id.startswith("open-athena/")
                or not HfApi()
                .repo_info(plan.source_dataset_id, repo_type="dataset", revision=plan.source_revision)
                .private
            )
        )
        original = root / "original.parquet" if snapshot_required else None
        selected = filter_population(plan, metadata["specs_output_dir"], tasks, original)
        hashes = selected.pop("hashes")
        content_sha256 = verify_retained(tasks, hashes)
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps({"plan": plan.model_dump(mode="json"), "lexical": measurement, **selected}, indent=2) + "\n"
        )
        card = root / "README.md"
        repo = retention_repo(plan, version)
        original_pin = {"repo": plan.source_dataset_id, "revision": plan.source_revision}
        if original is not None:
            card.write_text(
                dataset_card(plan, population="original archive", count=plan.raw_count, evidence_root=str(output))
            )
            original_repo = repo.replace("-retained-", "-original-")
            original_revision = publish(original, card, original_repo, evidence_files=(manifest,))
            original_pin = {"repo": original_repo, "revision": original_revision}
            (output / "original-pin.json").write_text(json.dumps(original_pin) + "\n", auto_mkdir=True)
        card.write_text(
            dataset_card(plan, population="retained release", count=plan.retained_count, evidence_root=str(output))
        )
        duplicate_manifest, combined_manifest = root / "duplicates.parquet", root / "combined-drops.parquet"
        copy(plan.drop_manifest_uri, str(duplicate_manifest))
        copy(plan.combined_drop_manifest_uri, str(combined_manifest))
        revision = publish(tasks, card, repo, evidence_files=(manifest, duplicate_manifest, combined_manifest))
        downloaded = Path(hf_hub_download(repo, "tasks.parquet", repo_type="dataset", revision=revision))
        if verify_retained(downloaded, hashes) != content_sha256:
            raise ValueError(f"Readback content digest differs: {plan.source_id}")
        tasks_sha256 = file_sha256(StoragePath(str(tasks)))
        copy(str(tasks), str(output / "specs/part-000000.parquet"), no_clobber=True)
        result = {
            "source_id": plan.source_id,
            "plan": plan.model_dump(mode="json"),
            "repo": repo,
            "revision": revision,
            "url": f"https://huggingface.co/datasets/{repo}/tree/{revision}",
            "private": True,
            "format": "taskcompendium",
            "original_private_pin": original_pin,
            "tasks_parquet_sha256": tasks_sha256,
            "retained_content_sha256": content_sha256,
            "verified_retained_tasks": len(hashes),
            "specs_output_dir": str(output / "specs"),
            "evidence_url": str(result_path),
            "lexical": measurement,
        }
        result_path.write_text(json.dumps(result, indent=2) + "\n", auto_mkdir=True)
    LOGGER.info("Verified private release %s@%s: %d retained", repo, revision, plan.retained_count)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--exclude-source", action="append", default=[])
    args = parser.parse_args()
    configure_coreweave_s3()
    plans = [ReleasePlan.model_validate(row) for row in json.loads(StoragePath(args.plans).read_text())]
    plans = [p for p in plans if p.combined_drop_count and p.source_id not in args.exclude_source]
    plans.sort(key=lambda plan: -plan.raw_count)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=configure_coreweave_s3) as pool:
        futures = [pool.submit(publish_plan, plan, args.version, args.evidence_root) for plan in plans]
        results = [future.result() for future in as_completed(futures)]
    results.sort(key=lambda row: row["source_id"])
    (StoragePath(args.evidence_root) / "releases.json").write_text(json.dumps(results, indent=2) + "\n", auto_mkdir=True)


if __name__ == "__main__":
    main()
