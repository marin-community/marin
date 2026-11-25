# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate and freeze the published PDBThink coordinate panel."""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from pdbthink.dataset import load_dataset
from pdbthink.evaluation.runner import dataset_fingerprint
from pdbthink.generators import get_generator
from pdbthink.scoring.scorers import SCORING_VERSION
from pdbthink.validate import validate_dataset
from rigging.filesystem.factory import open_url
from rigging.filesystem.storage_path import prefix_join

from taskcompendium.importers.pdbthink import BUNDLE_PATH, PDBTHINK_REVISION, manifest


def prepare_bundle(dataset: Path) -> tuple[bytes, dict]:
    """Validate the dataset and serialize its evaluation records."""
    validation = validate_dataset(dataset)
    if not validation.ok:
        raise ValueError(f"PDBThink validation failed: {validation.errors}")
    instances, renders = load_dataset(dataset)
    build = json.loads((dataset / "build_report.json").read_text())
    if dataset_fingerprint(renders) != build["dataset_fingerprint"]:
        raise ValueError("Dataset fingerprint differs from the published coordinate panel")
    by_instance = {instance.semantic_instance_id: instance for instance in instances}
    records = []
    for render in sorted(renders, key=lambda render: render.render_id):
        instance = by_instance[render.semantic_instance_id]
        if instance.curation_status == "rejected" or instance.is_mechanistic:
            raise ValueError(f"Non-coordinate or rejected question: {render.render_id}")
        if not render.user_prompt:
            raise ValueError(f"Missing prompt for {render.render_id}")
        parameters = dict(instance.question_parameters)
        if instance.question_family != "T01":
            parameters.update(get_generator(instance.question_family).prompt_parameters(parameters))
        metadata = {
            "render_id": render.render_id,
            "semantic_instance_id": render.semantic_instance_id,
            "question_family": render.question_family,
            "protein_group_id": render.protein_group_id,
            "cluster": instance.gold_evidence.get("cluster") or render.protein_group_id,
            "representation": render.representation,
            "is_rotation_variant": render.is_rotation_variant,
            "rotation_seed": render.rotation_seed,
            "state_order_seed": render.state_order_seed,
            "answer_schema": render.answer_schema,
            "is_mechanistic": instance.is_mechanistic,
            "input_token_count": render.input_token_count,
        }
        records.append(
            {
                "render_id": render.render_id,
                "messages": [
                    {"role": "system", "content": render.system_prompt},
                    {"role": "user", "content": render.user_prompt},
                ],
                "answer_schema": render.answer_schema,
                "gold_answer": render.gold_answer,
                "parameters": parameters,
                "report_metadata": metadata,
            }
        )
    bundle = {"source_revision": PDBTHINK_REVISION, "scoring_version": SCORING_VERSION, "renders": records}
    payload = json.dumps(bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    compressed = gzip.compress(payload, mtime=0)
    description = {
        "source_revision": PDBTHINK_REVISION,
        "dataset": build["version"],
        "dataset_fingerprint": build["dataset_fingerprint"],
        "scientific_revision": build["scientific_commit"],
        "status": "development",
        "bundle_sha256": hashlib.sha256(compressed).hexdigest(),
        "bundle_bytes": len(compressed),
        "scoring_version": SCORING_VERSION,
        "prompt_version": build["prompt_version"],
        "definitions_sha256": build["definitions_sha256"],
        "parent_archive_sha256": build["parent_archive_sha256"],
        "families": build["families"],
        "n_instances": len({render.semantic_instance_id for render in renders}),
        "n_renders": len(records),
        "curation": dict(sorted(Counter(instance.curation_status for instance in instances).items())),
        "representations": dict(sorted(Counter(render.representation for render in renders).items())),
        "source_structure_sha256s": sorted(
            {digest for instance in instances for digest in instance.source_file_sha256s}
        ),
    }
    return compressed, description


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Built PDBThink dataset including prompts and gold")
    parser.add_argument(
        "--cache-prefix", required=True, help="Local or object-store dataset cache for corpus acquisition"
    )
    args = parser.parse_args()
    payload, description = prepare_bundle(args.dataset)
    if description != manifest():
        raise ValueError("Dataset differs from the frozen PDBThink manifest; do not publish it under this version")
    destination = prefix_join(args.cache_prefix, BUNDLE_PATH)
    with open_url(destination, "wb") as stream:
        stream.write(payload)
    with open_url(destination, "rb") as stream:
        if hashlib.sha256(stream.read()).hexdigest() != description["bundle_sha256"]:
            raise ValueError(f"Published bundle verification failed: {destination}")
    print(f"Prepared {description['n_renders']} PDBThink renders at {destination}")


if __name__ == "__main__":
    main()
