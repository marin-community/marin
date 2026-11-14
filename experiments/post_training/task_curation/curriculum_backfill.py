# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage frozen retained populations and run the existing direct-text coverage graph."""

import argparse
import hashlib
import json
import logging
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
from multiprocessing import get_context
from typing import Annotated, Literal
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.curation import SourceDeclaration
from finestore.schema import arrow_schema
from fray.types import ResourceConfig
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from marin.execution.lazy import ArtifactStep, run
from marin.processing.embeddings.harrier import HARRIER_DIM, HARRIER_REPO, HARRIER_REVISION, HarrierConfig, load_harrier
from pydantic import AfterValidator, BaseModel, ConfigDict
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.curriculum.alignment import CurriculumAnchor
from taskcompendium.pipeline.curriculum.models import AlignmentRelease, CurriculumCoverage, EmbeddingSpec
from taskcompendium.pipeline.semantics import TaskInputFile, TaskSemanticInput, task_semantic_input
from zephyr.context import ZephyrContext
from zephyr.readers import load_parquet

from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.catalog_identity import canonical_declaration
from experiments.post_training.task_curation.embedding import (
    INPUTS_VERSION,
    DescriptionEndpoint,
    SemanticArtifact,
    curation_embedding_pipeline,
)
from experiments.post_training.task_curation.harrier_parity import accepted_cuda_parity, accepted_parity
from experiments.post_training.task_curation.local import artifact_prefix, local_client
from experiments.post_training.task_curriculum.catalog_artifact import TASK_CURRICULUM
from experiments.post_training.task_curriculum.models import CurriculumCatalog

LOGGER = logging.getLogger(__name__)
BATCH_SIZE = 4096
STAGES = ("descriptions", "task_embeddings", "curriculum_embeddings", "coverage")


class FrozenCoverageSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    declaration: Annotated[SourceDeclaration, AfterValidator(canonical_declaration)]
    specs_dir: str
    task_column: str
    raw_count: int
    task_specs_sha256: str
    task_spec_scan_sha256: str
    input_record_sha256: str
    intended_use: Literal["train", "eval"]
    input_evidence: str
    drop_manifest_uri: str | None = None
    drop_manifest_sha256: str | None = None
    drop_count: int = 0


def verified_task_spec_rows(source: FrozenCoverageSource) -> Iterator[str]:
    """Verify exact stored JSON cells in sorted Parquet-path and physical row order."""
    digest = hashlib.sha256()
    count = 0
    for path in sorted((StoragePath(source.specs_dir) / "*.parquet").glob(), key=str):
        with path.open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(batch_size=16, columns=[source.task_column]):
                for encoded in batch.column(0).to_pylist():
                    if encoded is None:
                        raise ValueError(f"Frozen converted population contains a rejected row: {source.declaration.id}")
                    digest.update(encoded.encode())
                    count += 1
                    yield encoded
    if digest.hexdigest() != source.task_spec_scan_sha256 or count != source.raw_count:
        raise ValueError(f"Frozen TaskSpec scan changed: {source.declaration.id}")


def prepare_source(source: FrozenCoverageSource, output_root: str) -> dict:
    """Project exact frozen task bytes, accounting for every retained or excluded row."""
    key = canonical_sha256(source.model_dump(mode="json"))
    root = StoragePath(output_root) / "sources" / key
    manifest_path = root / "manifest.json"
    if file_sha256(StoragePath(source.input_evidence)) != source.input_record_sha256:
        raise ValueError(f"Frozen population receipt changed: {source.declaration.id}")
    drops = set()
    if source.drop_manifest_uri is not None:
        drop_path = StoragePath(source.drop_manifest_uri)
        if file_sha256(drop_path) != source.drop_manifest_sha256:
            raise ValueError(f"Frozen drop manifest changed: {source.declaration.id}")
        with drop_path.open("rb") as stream:
            identifiers = pq.read_table(stream, columns=["task_id"]).column(0).to_pylist()
        drops = set(identifiers)
        if len(drops) != len(identifiers) or len(drops) != source.drop_count:
            raise ValueError(f"Drop manifest does not reconcile: {source.declaration.id}")
    elif source.drop_count:
        raise ValueError("Exclusions require a frozen manifest")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["input"] != source.model_dump(mode="json"):
            raise ValueError("Completed public input differs from its frozen source")
        for _ in verified_task_spec_rows(source):
            pass
        public = json.loads((root / "public-inputs.json").read_text())
        if public["stored_stream_sha256"] != source.task_spec_scan_sha256 or public["source_key"] != canonical_sha256(
            manifest
        ):
            raise ValueError("Completed public input has a different verified TaskSpec scan")
        return {"source_id": source.declaration.id, "root": str(root), **manifest}
    manifest = {
        "intended_use": source.intended_use,
        "input_rows": source.raw_count,
        "converted_rows": source.declaration.task_count,
        "rejections": {"frozen_retention_exclusions": source.drop_count},
        "input": source.model_dump(mode="json"),
    }
    source_key = canonical_sha256(manifest)
    task_source = TaskInputFile(source.declaration.id, source_key, "", source.intended_use, "converted")
    attempt = root / "attempts" / uuid4().hex
    paths, seen, seen_drops = [], set(), set()
    rows = []
    retained_count = 0

    def flush() -> None:
        path = attempt / f"part-{len(paths):06d}.parquet"
        with path.open("wb", auto_mkdir=True) as stream:
            pq.write_table(
                pa.Table.from_pylist(rows, schema=arrow_schema(TaskSemanticInput)), stream, compression="zstd"
            )
        paths.append(str(path))
        rows.clear()

    for encoded in verified_task_spec_rows(source):
        task = TaskSpec.model_validate_json(encoded)
        if task.id in seen:
            raise ValueError(f"Repeated task ID: {source.declaration.id}/{task.id}")
        seen.add(task.id)
        if task.id in drops:
            seen_drops.add(task.id)
            continue
        rows.append(task_semantic_input(task, task_source).model_dump(mode="json"))
        retained_count += 1
        if len(rows) == BATCH_SIZE:
            flush()
    if rows:
        flush()
    if (
        retained_count != source.declaration.task_count
        or source.raw_count != retained_count + source.drop_count
        or seen_drops != drops
    ):
        raise ValueError(f"Frozen task bytes or population accounting differs: {source.declaration.id}")
    # Paths are execution outputs; the source key binds the complete frozen input.
    result = {
        "source_id": source.declaration.id,
        "root": str(root),
        "source_key": source_key,
        "paths": paths,
        "stored_stream_sha256": source.task_spec_scan_sha256,
    }
    (root / "public-inputs.json").write_text(json.dumps(result, indent=2))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), auto_mkdir=True)
    LOGGER.info("Prepared %s: %d retained, %d excluded", source.declaration.id, retained_count, len(drops))
    return result


def initialize_preparation_worker() -> None:
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


def prepare_sources(sources: list[FrozenCoverageSource], output_root: str, workers: int) -> None:
    """Write a restartable public corpus without publishing or modifying task contracts."""
    selection_key = canonical_sha256([source.model_dump(mode="json") for source in sources])
    root = StoragePath(output_root)
    corpus_path = root / "corpus.json"
    completed = json.loads(corpus_path.read_text()) if corpus_path.exists() else None
    if completed is not None:
        if completed["selection_key"] != selection_key:
            raise ValueError("A changed source selection requires a new output root")
        for source in sources:
            key = canonical_sha256(source.model_dump(mode="json"))
            if not (root / "sources" / key / "manifest.json").exists():
                raise ValueError("Completed corpus is missing a source manifest")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn"), initializer=initialize_preparation_worker
    ) as pool:
        results = list(pool.map(partial(prepare_source, output_root=output_root), sources))
    paths = []
    for result in results:
        # A completed per-source manifest can be reused after a coordinator interruption.
        public = json.loads((StoragePath(result["root"]) / "public-inputs.json").read_text())
        paths.extend(public["paths"])
    artifact = SemanticArtifact(path=str(root / "inputs"), paths=paths)
    corpus = {
        "selection_key": selection_key,
        "population": "converted",
        "sources": [{"source_id": row["source_id"], "root": row["root"]} for row in results],
        "inventory": [source.declaration.model_dump(mode="json") for source in sources],
        "prepared_inputs": artifact.path,
    }
    if completed is not None:
        if completed != corpus or SemanticArtifact.raw_load(artifact.path).paths != paths:
            raise ValueError("Completed corpus differs from its verified public inputs")
        return
    write_record(
        ArtifactRecord(
            output_path=artifact.path, result_type=result_type_name(SemanticArtifact), result=artifact.result_payload()
        )
    )
    corpus_path.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), auto_mkdir=True)


def prepared_semantics(root: str, name: str) -> ArtifactStep[SemanticArtifact]:
    """Bind adopted semantic content to its immutable producer record."""
    record = (StoragePath(root) / ".artifact.json").read_bytes()
    SemanticArtifact.raw_load(root)
    return ArtifactStep.adopt(
        f"rl-semantics/{name}-{hashlib.sha256(record).hexdigest()[:16]}",
        "2026.11.13.1",
        source=root,
        kind=SemanticArtifact,
    )


def prepared_curriculum(root: str, spec: EmbeddingSpec) -> ArtifactStep[SemanticArtifact]:
    """Require adopted anchors to describe the requested curriculum and specification."""
    path = StoragePath(root)
    catalog = CurriculumCatalog.model_validate_json((path / "curriculum.json").read_bytes())
    expected = CurriculumCatalog.model_validate_json(
        (StoragePath(TASK_CURRICULUM.path()) / "curriculum.json").read_bytes()
    )
    anchor_paths = json.loads((path / "anchor_paths.json").read_bytes())
    anchors = [CurriculumAnchor.model_validate(row) for part in anchor_paths for row in load_parquet(part)]
    if (
        catalog != expected
        or anchors != expected.embedding_anchors()
        or EmbeddingSpec.model_validate_json((path / "embedding_spec.json").read_bytes()) != spec
        or json.loads((path / "nodes.json").read_bytes())
        != [node.model_dump(mode="json") for node in expected.atlas_nodes()]
    ):
        raise ValueError("Adopted anchors differ from the requested curriculum or embedding specification")
    return prepared_semantics(root, "regional-anchors")


def run_stage(
    corpus_uri: str,
    artifact_root: str,
    cache_root: str,
    stage: str,
    device: Literal["cpu", "cuda"],
    workers: int,
    worker_memory: str,
    run_graph: bool,
    *,
    parity_receipt_uri: str | None = None,
) -> None:
    if device == "cuda" and parity_receipt_uri is None:
        raise ValueError("CUDA embedding requires an accepted --parity-receipt")
    corpus = json.loads(StoragePath(corpus_uri).read_text())
    inventory = [canonical_declaration(SourceDeclaration.model_validate(row)) for row in corpus["inventory"]]
    curated = {}
    for row in corpus["sources"]:
        manifest = json.loads((StoragePath(row["root"]) / "manifest.json").read_text())
        curated[row["source_id"]] = ArtifactStep.adopt(
            f"rl-semantics/source-{canonical_sha256(manifest)[:16]}", INPUTS_VERSION, source=row["root"]
        )
    spec = EmbeddingSpec(
        model=HARRIER_REPO,
        revision=HARRIER_REVISION,
        tokenizer_revision=HARRIER_REVISION,
        dimensions=HARRIER_DIM,
        max_tokens=16384,
        compute_dtype="bfloat16",
    )
    parity = None
    if parity_receipt_uri is not None:
        parity = (
            accepted_cuda_parity(parity_receipt_uri, spec)
            if device == "cuda" and run_graph
            else accepted_parity(parity_receipt_uri, spec)
        )
    config = HarrierConfig(
        model=spec.model,
        revision=spec.revision,
        tokenizer_revision=spec.tokenizer_revision,
        dimensions=spec.dimensions,
        max_tokens=spec.max_tokens,
        compute_dtype=spec.compute_dtype,
        device=device,
        cpu_threads=4,
    )
    runtime = CampaignRuntime()
    with artifact_prefix(artifact_root):
        stages = curation_embedding_pipeline(
            curated=curated,
            curriculum=TASK_CURRICULUM,
            runtime=runtime,
            description_model=None,
            endpoint=DescriptionEndpoint(None, "unused"),
            spec=spec,
            embedder_factory=partial(load_harrier, config),
            cache_root=cache_root,
            population="converted",
            sample_size=None,
            seed=0,
            inventory=inventory,
            prepared_inputs=ArtifactStep.adopt(
                f"rl-semantics/frozen-public-inputs-{corpus['selection_key'][:16]}",
                INPUTS_VERSION,
                source=corpus["prepared_inputs"],
                kind=SemanticArtifact,
            ),
            prepared_descriptions=(
                prepared_semantics(corpus["prepared_descriptions"], "regional-instructions")
                if "prepared_descriptions" in corpus
                else None
            ),
            prepared_curriculum_embeddings=(
                prepared_curriculum(corpus["prepared_curriculum_embeddings"], spec)
                if "prepared_curriculum_embeddings" in corpus
                else None
            ),
        )
        paths = {name: getattr(stages, name).path() for name in ("inputs", *STAGES, "assignments", "publication")}
        LOGGER.info("Stage plan: %s", json.dumps(paths, sort_keys=True))
        if not run_graph:
            return
        with (
            local_client() as client,
            ZephyrContext(
                client=client,
                max_workers=workers,
                resources=ResourceConfig(cpu=4, ram=worker_memory),
                coordinator_resources=ResourceConfig(cpu=1, ram="8g", preemptible=False),
                chunk_storage_prefix=str(StoragePath(artifact_root) / "chunks"),
                name="atlas-curriculum-backfill",
            ) as context,
            runtime.activate(context),
        ):
            result = run(getattr(stages, stage), max_concurrent=1)[0]
            LOGGER.info("Completed %s at %s", stage, result.path)
            if stage == "coverage":
                release_uri = StoragePath(paths["publication"]) / "release.json"
                coverage_uri = StoragePath(result.path) / "coverage.json"
                release = AlignmentRelease.model_validate_json(release_uri.read_text())
                coverage = CurriculumCoverage.model_validate_json(coverage_uri.read_text())
                if (
                    release.task_count != sum(row.task_count for row in inventory)
                    or len(release.sources) != len(inventory)
                    or any(row.basis != "matched" for row in release.sources)
                    or coverage.alignment_release_id != release.release_id
                ):
                    raise ValueError("Coverage does not reconcile with the frozen inventory")
                receipt = {
                    "phase": "retained",
                    "representation": "direct",
                    "corpus": corpus_uri,
                    "source_count": len(release.sources),
                    "task_count": release.task_count,
                    "ready_count": release.ready_count,
                    "insufficient_context_count": release.insufficient_context_count,
                    "embedding_spec": spec.model_dump(mode="json"),
                    "embedding_spec_sha256": canonical_sha256(spec.model_dump(mode="json")),
                    "release_id": release.release_id,
                    "coverage_id": coverage.coverage_id,
                    "release_uri": str(release_uri),
                    "release_sha256": file_sha256(release_uri),
                    "coverage_uri": str(coverage_uri),
                    "coverage_sha256": file_sha256(coverage_uri),
                    "stages": paths,
                    "parity_receipt": asdict(parity) if parity is not None else None,
                }
                (StoragePath(result.path) / "retained-coverage-receipt.json").write_text(json.dumps(receipt, indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--selection", required=True)
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--workers", type=int, required=True)
    embed = subparsers.add_parser("embed")
    embed.add_argument("--corpus", required=True)
    embed.add_argument("--artifact-root", required=True)
    embed.add_argument("--cache-root", required=True)
    embed.add_argument("--stage", choices=STAGES, required=True)
    embed.add_argument("--device", choices=("cpu", "cuda"), required=True)
    embed.add_argument("--workers", type=int, required=True)
    embed.add_argument("--worker-memory", required=True)
    embed.add_argument("--run", action="store_true")
    embed.add_argument("--parity-receipt", help="Immutable accepted Harrier backend receipt; required for CUDA.")
    args = parser.parse_args()
    configure_coreweave_s3()
    if args.command == "prepare":
        sources = [
            FrozenCoverageSource.model_validate(row) for row in json.loads(StoragePath(args.selection).read_text())
        ]
        identifiers = [row.declaration.id for row in sources]
        if not identifiers or len(identifiers) != len(set(identifiers)):
            raise ValueError("Selection requires distinct frozen sources")
        prepare_sources(sources, args.output_root, args.workers)
        return
    run_stage(
        args.corpus,
        args.artifact_root,
        args.cache_root,
        args.stage,
        args.device,
        args.workers,
        args.worker_memory,
        args.run,
        parity_receipt_uri=args.parity_receipt,
    )


if __name__ == "__main__":
    main()
