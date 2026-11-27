# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish a converted cohort as a private TaskCompendium dataset.

Curation that converts data must publish the converted artifact, so the Atlas can
point at something a reader can actually load instead of at the deprecated native
release. This uploads the lowered task spec rows beside the licensed dataset card,
verifies the result through an authenticated read-back, and prints the revision to
record. The licence carries forward from the parent pin's receipt through the
publication chokepoint: the card declares the parent's resolved upstream and the new
pin's receipt registers before the revision is returned.

Campaign datasets stay private until release, so the repo is created private and the
artifact is verified by reading it back through the cached login and comparing hashes.
Run it with that login (``HfApi()``/``hf_hub_download`` need no explicit token)::

    uv run python -m experiments.post_training.task_curation.calibforge_publish \
        --tasks /path/to/lowered-tasks.parquet \
        --card experiments/post_training/task_curation/datasets/calibforge_rebuilt_card.md \
        --repo open-athena/calibforge-v1-rebuilt-taskcompendium-20261111
"""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.models import TaskSpec

from experiments.post_training.task_curation.license_binding import LICENSE_RECEIPT_INDEX
from experiments.post_training.task_curation.license_publication import (
    PublishedLicensedPopulation,
    publish_licensed_population,
)
from experiments.post_training.task_curation.publication_gate import require_public_data

TASKS_FILENAME = "tasks.parquet"
CARD_FILENAME = "README.md"


def _require_loader_row_integrity(tasks: Path) -> None:
    """The cohort loader's per-row integrity, checked before any upload.

    A canonical row records its task_spec digest; a lowered row is the rollout
    envelope. A publication whose rows the cohort loader would refuse fails here
    rather than at a reader (ruling 66).
    """
    table = pq.read_table(tasks)
    if "task_spec" in table.column_names:
        if "task_sha256" not in table.column_names:
            raise ValueError(f"Cohort publication lacks the row-level task_sha256 column the loader requires: {tasks}")
        specs = table.column("task_spec").to_pylist()
        recorded = table.column("task_sha256").to_pylist()
        for serialized, digest in zip(specs, recorded, strict=True):
            if hashlib.sha256(serialized.encode()).hexdigest() != digest:
                raise ValueError(f"Published task hash differs from its task_spec content: {tasks}")
        return
    if "lowered_task_spec" in table.column_names:
        for row in table.to_pylist():
            envelope = json.loads(row["lowered_task_spec"])
            if not envelope.get("runtime"):
                raise ValueError(f"{row['path']}: envelope carries no runtime machines")
            TaskSpec.model_validate_json(json.dumps(envelope["task"]))
        return
    raise ValueError(f"Cohort publication carries neither the canonical nor the lowered task encoding: {tasks}")


def _source_pin_line(card_text: str) -> str | None:
    """The ``Source pin:`` line a publication card declares, for content-pin comparison."""
    for line in card_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Source pin:"):
            return stripped
    return None


def card_body(card: Path) -> str:
    """The card's provenance body; the chokepoint writes the licensed front matter."""
    text = card.read_text()
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            return parts[-1].strip()
    return text.strip()


def publish(
    tasks: Path,
    card: Path,
    repo: str,
    *,
    data_restriction: DataRestriction,
    evidence_files: tuple[Path, ...] = (),
    source_id: str,
    use: str,
    pretty_name: str,
    index_path: Path = LICENSE_RECEIPT_INDEX,
) -> PublishedLicensedPopulation:
    """Upload ``tasks`` beside its licensed card to a private ``repo``; return the publication.

    The caller passes the source population's tier: a PRIVATE_ALWAYS payload is
    refused here at the shared helper, never published anywhere. The licence moves
    forward mechanically from the parent pin's receipt: the card declares the parent's
    resolved upstream, every uploaded byte is read back, and the new pin's receipt
    lands in the registry before the revision is returned.
    """
    require_public_data(data_restriction, destination=repo)
    _require_loader_row_integrity(tasks)
    with tempfile.TemporaryDirectory(prefix="calibforge-publish-") as directory:
        staging = Path(directory)
        shutil.copyfile(tasks, staging / TASKS_FILENAME)
        for source in evidence_files:
            shutil.copyfile(source, staging / source.name)
        return publish_licensed_population(
            source_id,
            repo,
            staging,
            (TASKS_FILENAME, *(source.name for source in evidence_files)),
            body=card_body(card),
            pretty_name=pretty_name,
            use=use,
            index_path=index_path,
        )


def main() -> None:
    """Publish the cohort and print the revision to pin in the Atlas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True, help="lowered task spec parquet")
    parser.add_argument("--card", type=Path, required=True, help="dataset card body markdown")
    parser.add_argument("--repo", required=True, help="dataset repo, e.g. open-athena/<name> (created private)")
    parser.add_argument(
        "--data-restriction", type=DataRestriction, required=True, help="the source's typed restriction tier"
    )
    parser.add_argument("--source-id", required=True, help="the cohort source id the publication belongs to")
    parser.add_argument("--use", required=True, choices=["train", "eval"], help="the publication use")
    parser.add_argument("--pretty-name", required=True, help="the card's pretty_name")
    args = parser.parse_args()
    publication = publish(
        args.tasks,
        args.card,
        args.repo,
        data_restriction=args.data_restriction,
        source_id=args.source_id,
        use=args.use,
        pretty_name=args.pretty_name,
    )
    print(f"{args.repo}@{publication.revision}")


if __name__ == "__main__":
    main()
