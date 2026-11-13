# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cross-device Harrier acceptance against the frozen CPU BF16 reference."""

import argparse
import hashlib
import json
import logging
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from marin.processing.embeddings.harrier import HarrierConfig, load_harrier
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.curriculum.alignment import CurriculumAnchor, CurriculumVectors, rank_task_batch
from taskcompendium.pipeline.curriculum.embedding import normalized_vectors
from taskcompendium.pipeline.curriculum.models import EmbeddingSpec

MIN_VECTOR_COSINE = 0.999
MIN_AREA_AGREEMENT = 0.999
PARITY_SAMPLE_SIZE = 2048


@dataclass(frozen=True)
class EmbeddingParity:
    rows: int
    min_cosine: float
    p01_cosine: float
    mean_cosine: float
    area_agreement: float
    facet_agreement: dict[str, float]

    def require_accepted(self) -> None:
        if self.min_cosine < MIN_VECTOR_COSINE or self.area_agreement < MIN_AREA_AGREEMENT:
            raise ValueError(f"Harrier cross-device parity failed: {self}")


def measure_parity(reference: np.ndarray, candidate: np.ndarray, curriculum: CurriculumVectors) -> EmbeddingParity:
    """Compare normalized vectors and both primary curriculum-area assignments."""
    rows, dimensions = reference.shape
    reference = normalized_vectors(reference, rows, dimensions)
    candidate = normalized_vectors(candidate, rows, dimensions)
    cosines = np.sum(reference * candidate, axis=1)
    reference_matches = rank_task_batch(curriculum, reference)
    candidate_matches = rank_task_batch(curriculum, candidate)
    reference_areas = [{str(row.facet): row.subject_id for row in match.subjects} for match in reference_matches]
    candidate_areas = [{str(row.facet): row.subject_id for row in match.subjects} for match in candidate_matches]
    facets = sorted(reference_areas[0])
    return EmbeddingParity(
        rows=rows,
        min_cosine=float(cosines.min()),
        p01_cosine=float(np.quantile(cosines, 0.01)),
        mean_cosine=float(cosines.mean()),
        area_agreement=sum(left == right for left, right in zip(reference_areas, candidate_areas, strict=True)) / rows,
        facet_agreement={
            facet: (
                sum(left[facet] == right[facet] for left, right in zip(reference_areas, candidate_areas, strict=True))
                / rows
            )
            for facet in facets
        },
    )


def validate_bundle(bundle_uri: str, output_uri: str) -> None:
    """Measure a frozen native59 sample on CUDA and publish the acceptance receipt."""
    import torch  # noqa: PLC0415 - optional GPU inference dependency

    output = StoragePath(output_uri)
    if output.exists():
        raise ValueError("Preserve previous parity receipts; use a new output URI")
    bundle_bytes = StoragePath(bundle_uri).read_bytes()
    bundle = json.loads(bundle_bytes)
    spec = EmbeddingSpec.model_validate(bundle["embedding_spec"])
    if (
        bundle["sample_size"] != PARITY_SAMPLE_SIZE
        or bundle["min_vector_cosine"] != MIN_VECTOR_COSINE
        or bundle["min_area_agreement"] != MIN_AREA_AGREEMENT
        or bundle["embedding_spec_sha256"] != canonical_sha256(spec.model_dump(mode="json"))
    ):
        raise ValueError("Parity bundle does not match the frozen acceptance contract")
    tables = {}
    for name, record in bundle["files"].items():
        encoded = StoragePath(record["uri"]).read_bytes()
        if hashlib.sha256(encoded).hexdigest() != record["sha256"]:
            raise ValueError(f"Parity input changed: {name}")
        tables[name] = pq.read_table(pa.BufferReader(encoded)).to_pylist()
    tasks = tables["tasks.parquet"]
    anchors = tables["anchors.parquet"]
    if len(tasks) != PARITY_SAMPLE_SIZE:
        raise ValueError("Parity bundle has an incomplete task sample")
    curriculum = CurriculumVectors(
        tuple(
            CurriculumAnchor.model_validate({key: value for key, value in row.items() if key != "embedding"})
            for row in anchors
        ),
        np.asarray([row["embedding"] for row in anchors], dtype=np.float32),
    )
    if (
        canonical_sha256([row["embedding_key"] for row in tasks]) != bundle["keys_sha256"]
        or dict(Counter(row["source_id"] for row in tasks)) != bundle["source_counts"]
        or canonical_sha256({"anchors": [anchor.anchor_id for anchor in curriculum.anchors], "spec": spec.model_dump()})
        != bundle["curriculum_key"]
    ):
        raise ValueError("Parity sample or curriculum differs from its frozen identity")
    config = HarrierConfig(
        model=spec.model,
        revision=spec.revision,
        tokenizer_revision=spec.tokenizer_revision,
        dimensions=spec.dimensions,
        max_tokens=spec.max_tokens,
        compute_dtype=spec.compute_dtype,
        device="cuda",
    )
    backend = load_harrier(config)
    started = time.monotonic()
    vectors = backend.embed([row["text"] for row in tasks])
    inference_seconds = time.monotonic() - started
    measured = measure_parity(
        np.asarray([row["reference_embedding"] for row in tasks], dtype=np.float32), vectors, curriculum
    )
    receipt = {
        "as_of": datetime.now(UTC).isoformat(),
        "bundle_uri": bundle_uri,
        "bundle_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        "embedding_spec_sha256": bundle["embedding_spec_sha256"],
        "curriculum_key": bundle["curriculum_key"],
        "keys_sha256": bundle["keys_sha256"],
        "source_counts": bundle["source_counts"],
        "thresholds": {"min_vector_cosine": MIN_VECTOR_COSINE, "min_area_agreement": MIN_AREA_AGREEMENT},
        "measurement": asdict(measured),
        "accepted": measured.min_cosine >= MIN_VECTOR_COSINE and measured.area_agreement >= MIN_AREA_AGREEMENT,
        "inference_seconds": inference_seconds,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
    }
    output.write_text(json.dumps(receipt, indent=2), auto_mkdir=True)
    logging.info("Harrier parity receipt: %s", json.dumps(receipt, sort_keys=True))
    measured.require_accepted()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    validate_bundle(args.bundle, args.output)


if __name__ == "__main__":
    main()
