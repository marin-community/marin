# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a local public Workplace candidate for the TaskCompendium alpha dataset."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from nemo_workplace.provider import ACTION_INTERFACE, PROVIDER_REVISION, SEED_SHA256, TOOLS_SHA256
from taskcompendium.importers.nemo_workplace import (
    DATASET,
    DATASET_REVISION,
    DATASET_SPLIT_ROW_COUNTS,
    DATASET_SPLIT_SHA256,
    IMPORTER_REVISION,
    PROVIDER_GIT_REVISION,
    WorkplaceSplit,
    import_dataset_split,
    select_dataset_rows,
    workplace_environment_config,
)
from taskcompendium.models import SCHEMA_VERSION
from taskcompendium.public_projection import public_task

REPO_ID = "open-athena/taskcompendium-alpha-1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_workplace_split(source_path: Path, destination: Path, split: WorkplaceSplit) -> tuple[int, dict[str, int]]:
    source = source_path.read_bytes()
    selected = select_dataset_rows(source, split)
    imported = import_dataset_split(source, split)
    categories: Counter[str] = Counter()
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for raw_row, task in zip(selected, imported, strict=True):
            category = json.loads(raw_row)["category"]
            categories[category] += 1
            record = public_task(
                task.specification,
                task.convention,
                task.environment_config,
                tags=(),  # The Workplace source has categories but no tags field.
                source_category=category,
            )
            stream.write(record.model_dump_json() + "\n")
    return len(imported), dict(sorted(categories.items()))


def _card(train_rows: int, validation_rows: int) -> str:
    return f"""---
pretty_name: TaskCompendium Alpha 1 Candidate
license: cc-by-4.0
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train.jsonl
      - split: validation
        path: data/validation.jsonl
---

# TaskCompendium Alpha 1 Candidate

This candidate contains agent-visible NeMo Workplace tasks: {train_rows} train and
{validation_rows} validation rows. It contains no verifier configuration, expected
state, source gold actions, or private resources. The complete task specifications
remain outside this release.

The source is NVIDIA's
[Nemotron-RL-agent-workplace_assistant](https://huggingface.co/datasets/{DATASET})
at commit `{DATASET_REVISION}`, licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit: NVIDIA Corporation.
The source card describes 1,260 records; the two pinned source files contain
{train_rows} + {validation_rows} = {train_rows + validation_rows} rows. File digests
and category counts are in `manifest.json`.

Tool execution uses the separately licensed
[NeMo Workplace provider](https://github.com/marin-community/nemo_workplace) at
commit `{PROVIDER_GIT_REVISION}`, derived from NVIDIA NeMo Gym under Apache 2.0.
The provider implementation and seed are referenced by digest; they are not copied
into the data files.

This is a local candidate. The broader mixed-source alpha release still needs
completed TaskTrove conversions and Harbor sample evidence for each included source.
"""


def build_workplace_candidate(
    train_source: Path,
    validation_source: Path,
    destination: Path,
    *,
    builder_revision: str,
) -> Path:
    """Write a deterministic public Workplace candidate from trusted pinned files."""
    if len(builder_revision) != 40 or any(character not in "0123456789abcdef" for character in builder_revision):
        raise ValueError("Builder revision must be a full lowercase Git commit")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        data = temporary / "data"
        data.mkdir()
        source_files: dict[WorkplaceSplit, Path] = {"train": train_source, "validation": validation_source}
        files: dict[str, dict[str, object]] = {}
        row_counts: dict[WorkplaceSplit, int] = {}
        for split, path in source_files.items():
            output = data / f"{split}.jsonl"
            rows, categories = _write_workplace_split(path, output, split)
            row_counts[split] = rows
            files[split] = {
                "path": f"data/{split}.jsonl",
                "rows": rows,
                "sha256": _sha256(output),
                "categories": categories,
            }
        manifest: dict[str, object] = {
            "format_version": 1,
            "visibility": "agent",
            "repo_id": REPO_ID,
            "builder_revision": builder_revision,
            "public_record_version": 1,
            "publication_ready": False,
            "families": {
                "nemo_workplace": {
                    "rows": sum(row_counts.values()),
                    "task_spec_schema": SCHEMA_VERSION,
                    "importer_revision": IMPORTER_REVISION,
                    "builder_revision": builder_revision,
                    "harbor_samples": [],
                }
            },
            "sources": {
                "nemo_workplace": {
                    "dataset": DATASET,
                    "revision": DATASET_REVISION,
                    "license": "cc-by-4.0",
                    "attribution": "NVIDIA Corporation",
                    "importer_revision": IMPORTER_REVISION,
                    "action_interface": ACTION_INTERFACE,
                    "provider_revision": PROVIDER_GIT_REVISION,
                    "provider_implementation_revision": PROVIDER_REVISION,
                    "seed_sha256": SEED_SHA256,
                    "tools_sha256": TOOLS_SHA256,
                    "provider_binding": {
                        "provider": (
                            "python+git+https://github.com/marin-community/nemo_workplace@"
                            f"{PROVIDER_GIT_REVISION}:nemo_workplace.provider:NemoWorkplaceProvider"
                        ),
                        "tools": list(workplace_environment_config().tool_providers["workplace"].tools),
                    },
                    "submission_convention": {"id": "state", "answer_format": "state", "provider": "workplace"},
                    "split_files": {
                        split: {
                            "path": f"{split}.jsonl",
                            "sha256": DATASET_SPLIT_SHA256[split],
                            "rows": DATASET_SPLIT_ROW_COUNTS[split],
                        }
                        for split in source_files
                    },
                }
            },
            "data_files": files,
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (temporary / "README.md").write_text(_card(row_counts["train"], row_counts["validation"]))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local, agent-visible TaskCompendium alpha candidate")
    parser.add_argument("--workplace-train", type=Path, required=True)
    parser.add_argument("--workplace-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--builder-revision", required=True)
    arguments = parser.parse_args()
    build_workplace_candidate(
        arguments.workplace_train,
        arguments.workplace_validation,
        arguments.output,
        builder_revision=arguments.builder_revision,
    )


if __name__ == "__main__":
    main()
