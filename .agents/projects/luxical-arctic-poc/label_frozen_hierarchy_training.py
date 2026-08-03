# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply the accepted GLM hierarchy to a disjoint student-training sample."""

import argparse
import hashlib
import json
import logging
import time
from dataclasses import asdict
from functools import partial
from typing import Any

from glm_hierarchical_labels import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_MAX_NUM_SEQS,
    SOURCE_RUN_ROOT,
    VARIANTS,
    Variant,
    assign_with_checkpoints,
    parse_hierarchy,
    summary,
    validate_hierarchy,
)
from glm_semantic_labels import SampleDocument, read_jsonl, write_json, write_jsonl
from iris.rpc import job_pb2
from label_frozen_hierarchy import MANIFEST_URL, document_identity, documents_excluding
from ladder_config import SEED, read_json

from experiments.rollout_data.glm52_vllm import MODEL, MODEL_REVISION, Glm52LaunchConfig, ServerConfig, serve_glm52

DEFAULT_TRAINING_SIZE = 50_000
DEFAULT_TENSOR_PARALLEL_SIZE = 8
RAY_PORT = 6_379
HTTP_PORT = 8_000

logger = logging.getLogger(__name__)


def identity_digest(documents: list[SampleDocument]) -> str:
    """Return a stable digest of document identities."""
    payload = json.dumps(sorted(document_identity(row) for row in documents), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def projection_training_documents(
    manifest: dict[str, Any],
    pilot_documents: list[SampleDocument],
    evaluation_documents: list[SampleDocument],
    training_size: int,
) -> list[SampleDocument]:
    """Select training documents that exclude the pilot and fixed evaluation."""
    pilot_ids = {document_identity(row) for row in pilot_documents}
    evaluation_ids = {document_identity(row) for row in evaluation_documents}
    overlap = pilot_ids & evaluation_ids
    if overlap:
        raise ValueError(f"The pilot and fixed evaluation overlap on {len(overlap)} documents")
    documents = documents_excluding(manifest, [*pilot_documents, *evaluation_documents], training_size)
    training_ids = {document_identity(row) for row in documents}
    if training_ids & (pilot_ids | evaluation_ids):
        raise ValueError("The projection training sample overlaps an excluded sample")
    return documents


def label_projection_training_set(
    vllm_url: str,
    pilot_run_id: str,
    variant: Variant,
    evaluation_run_id: str,
    training_run_id: str,
    training_size: int,
    batch_size: int,
    concurrency: int,
) -> None:
    """Apply one frozen hierarchy to a disjoint student-training sample."""
    pilot_root = SOURCE_RUN_ROOT / "hierarchies-v1" / pilot_run_id
    taxonomy_path = pilot_root / variant.name / "taxonomy.json"
    taxonomy_text = taxonomy_path.read_text()
    hierarchy = parse_hierarchy(json.loads(taxonomy_text))
    validate_hierarchy(hierarchy, variant)

    pilot_documents = [SampleDocument(**row) for row in read_jsonl(SOURCE_RUN_ROOT / "sample-private.jsonl.gz")]
    evaluation_root = pilot_root / variant.name / evaluation_run_id
    evaluation_documents = [SampleDocument(**row) for row in read_jsonl(evaluation_root / "sample-private.jsonl.gz")]
    manifest = read_json(MANIFEST_URL)
    documents = projection_training_documents(manifest, pilot_documents, evaluation_documents, training_size)

    output_root = pilot_root / variant.name / training_run_id
    write_jsonl(output_root / "sample-private.jsonl.gz", (asdict(document) for document in documents))
    write_json(
        str(output_root / "run-config.json"),
        {
            "training_run_id": training_run_id,
            "pilot_run_id": pilot_run_id,
            "excluded_evaluation_run_id": evaluation_run_id,
            "variant": asdict(variant),
            "manifest_url": MANIFEST_URL,
            "manifest_sha256": manifest["sha256"],
            "taxonomy_path": str(taxonomy_path),
            "taxonomy_sha256": hashlib.sha256(taxonomy_text.encode()).hexdigest(),
            "pilot_identity_sha256": identity_digest(pilot_documents),
            "evaluation_identity_sha256": identity_digest(evaluation_documents),
            "training_identity_sha256": identity_digest(documents),
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "seed": SEED,
            "document_count": len(documents),
            "sampling": "nested_source_balanced_then_remove_pilot_and_fixed_evaluation",
            "source_metadata_in_prompts": False,
            "assignment_input": "raw_document_view",
            "purpose": "semantic_projection_training",
        },
    )

    started = time.time()
    assignments = assign_with_checkpoints(
        vllm_url,
        documents,
        hierarchy,
        0,
        output_root,
        batch_size,
        concurrency,
    )
    result = summary(variant, hierarchy, assignments)
    output = {
        **result,
        "pilot_run_id": pilot_run_id,
        "excluded_evaluation_run_id": evaluation_run_id,
        "training_run_id": training_run_id,
        "elapsed_seconds": time.time() - started,
        "complete": True,
    }
    write_json(str(output_root / "summary.json"), output)
    logger.info("GLM_PROJECTION_TRAINING_LABELS=%s", json.dumps(output, sort_keys=True))


def launch_config(
    pilot_run_id: str,
    variant: Variant,
    evaluation_run_id: str,
    training_run_id: str,
    training_size: int,
    batch_size: int,
    concurrency: int,
    tensor_parallel_size: int,
    max_model_len: int,
    max_num_seqs: int,
) -> Glm52LaunchConfig:
    """Return the server config for projection-training labels."""
    return Glm52LaunchConfig(
        vllm_endpoint=f"glm52-projection-training-{training_run_id}",
        ray_endpoint=f"glm52-projection-training-ray-{training_run_id}",
        server=ServerConfig(max_model_len=max_model_len, max_num_seqs=max_num_seqs),
        tensor_parallel_size=tensor_parallel_size,
        priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
        client=partial(
            label_projection_training_set,
            pilot_run_id=pilot_run_id,
            variant=variant,
            evaluation_run_id=evaluation_run_id,
            training_run_id=training_run_id,
            training_size=training_size,
            batch_size=batch_size,
            concurrency=concurrency,
        ),
    )


def main() -> None:
    """Parse arguments and run the projection-training label client."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-run-id", required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--evaluation-run-id", required=True)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--training-size", type=int, default=DEFAULT_TRAINING_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--tensor-parallel-size", type=int, default=DEFAULT_TENSOR_PARALLEL_SIZE)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS)
    args = parser.parse_args()
    numeric = (
        args.training_size,
        args.batch_size,
        args.concurrency,
        args.tensor_parallel_size,
        args.max_model_len,
        args.max_num_seqs,
    )
    if min(numeric) < 1:
        parser.error("All numeric arguments must be positive")
    logging.basicConfig(level=logging.INFO)
    launch = launch_config(
        args.pilot_run_id,
        VARIANTS[args.variant],
        args.evaluation_run_id,
        args.training_run_id,
        args.training_size,
        args.batch_size,
        args.concurrency,
        args.tensor_parallel_size,
        args.max_model_len,
        args.max_num_seqs,
    )
    serve_glm52(launch, RAY_PORT, HTTP_PORT)


if __name__ == "__main__":
    main()
