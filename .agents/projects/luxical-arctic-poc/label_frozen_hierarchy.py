# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply one accepted semantic hierarchy to a new held-out sample."""

import argparse
import dataclasses
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
from glm_semantic_labels import SampleDocument, read_jsonl, select_sample, write_json, write_jsonl
from iris.rpc import job_pb2
from ladder_config import MANIFEST_ROOT, SEED, read_json

from experiments.rollout_data.glm52_vllm import MODEL, MODEL_REVISION, Glm52LaunchConfig, ServerConfig, serve_glm52

logger = logging.getLogger(__name__)

MANIFEST_URL = f"{MANIFEST_ROOT}/manifest.json"
DEFAULT_EVALUATION_SIZE = 10_000
DEFAULT_TENSOR_PARALLEL_SIZE = 8
RAY_PORT = 6_379
HTTP_PORT = 8_000


def document_identity(document: SampleDocument) -> tuple[str, int]:
    """Return the stable identity of one evaluation row."""
    return document.source, document.eval_rank


def held_out_documents(
    manifest: dict[str, Any],
    pilot_documents: list[SampleDocument],
    evaluation_size: int,
) -> list[SampleDocument]:
    """Select new documents after removing all hierarchy-pilot rows."""
    candidates = select_sample(manifest, evaluation_size + len(pilot_documents))
    pilot_ids = {document_identity(row) for row in pilot_documents}
    if len(pilot_ids) != len(pilot_documents):
        raise ValueError("The hierarchy pilot has duplicate evaluation identities")
    candidate_ids = {document_identity(row) for row in candidates}
    missing = pilot_ids - candidate_ids
    if missing:
        raise ValueError(f"The nested evaluation sample is missing {len(missing)} pilot rows")
    selected = [row for row in candidates if document_identity(row) not in pilot_ids]
    if len(selected) != evaluation_size:
        raise ValueError(f"The held-out sample has {len(selected)} rows, expected {evaluation_size}")
    return [dataclasses.replace(row, sample_index=index) for index, row in enumerate(selected)]


def label_frozen_hierarchy(
    vllm_url: str,
    pilot_run_id: str,
    variant: Variant,
    evaluation_run_id: str,
    evaluation_size: int,
    batch_size: int,
    concurrency: int,
) -> None:
    """Apply one frozen hierarchy to a source-blind held-out sample."""
    pilot_root = SOURCE_RUN_ROOT / "hierarchies-v1" / pilot_run_id
    taxonomy_path = pilot_root / variant.name / "taxonomy.json"
    taxonomy_text = taxonomy_path.read_text()
    hierarchy = parse_hierarchy(json.loads(taxonomy_text))
    validate_hierarchy(hierarchy, variant)

    pilot_documents = [SampleDocument(**row) for row in read_jsonl(SOURCE_RUN_ROOT / "sample-private.jsonl.gz")]
    manifest = read_json(MANIFEST_URL)
    documents = held_out_documents(manifest, pilot_documents, evaluation_size)

    output_root = pilot_root / variant.name / evaluation_run_id
    write_jsonl(output_root / "sample-private.jsonl.gz", (asdict(document) for document in documents))
    write_json(
        str(output_root / "run-config.json"),
        {
            "evaluation_run_id": evaluation_run_id,
            "pilot_run_id": pilot_run_id,
            "variant": asdict(variant),
            "manifest_url": MANIFEST_URL,
            "manifest_sha256": manifest["sha256"],
            "taxonomy_path": str(taxonomy_path),
            "taxonomy_sha256": hashlib.sha256(taxonomy_text.encode()).hexdigest(),
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "seed": SEED,
            "document_count": len(documents),
            "sampling": "nested_source_balanced_then_remove_hierarchy_pilot",
            "source_metadata_in_prompts": False,
            "assignment_input": "raw_document_view",
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
        "evaluation_run_id": evaluation_run_id,
        "elapsed_seconds": time.time() - started,
        "complete": True,
    }
    write_json(str(output_root / "summary.json"), output)
    logger.info("GLM_FROZEN_HIERARCHY_LABELS=%s", json.dumps(output, sort_keys=True))


def launch_config(
    pilot_run_id: str,
    variant: Variant,
    evaluation_run_id: str,
    evaluation_size: int,
    batch_size: int,
    concurrency: int,
    tensor_parallel_size: int,
    max_model_len: int,
    max_num_seqs: int,
) -> Glm52LaunchConfig:
    """Return the server config for held-out hierarchy labeling."""
    return Glm52LaunchConfig(
        vllm_endpoint=f"glm52-frozen-hierarchy-{evaluation_run_id}",
        ray_endpoint=f"glm52-frozen-hierarchy-ray-{evaluation_run_id}",
        server=ServerConfig(max_model_len=max_model_len, max_num_seqs=max_num_seqs),
        tensor_parallel_size=tensor_parallel_size,
        priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
        client=partial(
            label_frozen_hierarchy,
            pilot_run_id=pilot_run_id,
            variant=variant,
            evaluation_run_id=evaluation_run_id,
            evaluation_size=evaluation_size,
            batch_size=batch_size,
            concurrency=concurrency,
        ),
    )


def main() -> None:
    """Parse arguments and run the held-out label client in a GLM server gang."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-run-id", required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--evaluation-run-id", required=True)
    parser.add_argument("--evaluation-size", type=int, default=DEFAULT_EVALUATION_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--tensor-parallel-size", type=int, default=DEFAULT_TENSOR_PARALLEL_SIZE)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS)
    args = parser.parse_args()
    numeric = (
        args.evaluation_size,
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
        args.evaluation_size,
        args.batch_size,
        args.concurrency,
        args.tensor_parallel_size,
        args.max_model_len,
        args.max_num_seqs,
    )
    serve_glm52(launch, RAY_PORT, HTTP_PORT)


if __name__ == "__main__":
    main()
