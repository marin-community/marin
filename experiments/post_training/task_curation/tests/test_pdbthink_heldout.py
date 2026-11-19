# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Held-out references preserve the complete released public prompts and provenance."""

import hashlib
import io
import json
import tarfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from taskcompendium.request_budget import tool_free_prompt_sha256

from experiments.post_training.task_curation.datasets.pdbthink.tasks import DATASET, REVISION
from experiments.post_training.task_curation.pdbthink_heldout import SOURCE, ReferenceConfig, prepare_reference


def reference_fixture(tmp_path, *, corrupt_prompt=False):
    system = "Use the complete coordinate text."
    user = "Coordinates \u03b1\n" + "ATOM 1 CA A 1 1.123 2.456 3.789\n" * 5000
    prompt = {"system_prompt": system, "user_prompt": user}
    blob = io.BytesIO()
    with tarfile.open(fileobj=blob, mode="w:gz") as archive:
        for path, content in {
            "instruction.md": (system + "\n\n" + user).encode(),
            "prompt.json": json.dumps(prompt).encode(),
            "tests/answer.json": b'{"private_reference":"never include this hidden answer"}',
        }.items():
            member = tarfile.TarInfo(path)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    archive_bytes = blob.getvalue()
    prompt_hash = tool_free_prompt_sha256(({"role": "system", "content": system}, {"role": "user", "content": user}))
    files = []
    for split in ("validation", "test"):
        path = tmp_path / f"{split}.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "path": "native-task",
                        "split": split,
                        "task_binary": archive_bytes,
                        "prompt_sha256": "0" * 64 if corrupt_prompt else prompt_hash,
                    }
                ]
            ),
            path,
        )
        files.append(
            {
                "path": f"data/{split}-00000.parquet",
                "uri": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = tmp_path / "raw.json"
    manifest.write_text(json.dumps([{"source_id": SOURCE, "dataset_id": DATASET, "revision": REVISION, "files": files}]))
    config = ReferenceConfig(
        str(manifest),
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
        (("validation", 1), ("test", 1)),
        str(tmp_path / "reference"),
    )
    return config, system, user, prompt_hash, archive_bytes


def test_reference_preserves_full_public_coordinates_and_both_split_identities(tmp_path):
    config, system, user, prompt_hash, archive = reference_fixture(tmp_path)
    prepare_reference(config)
    output = tmp_path / "reference"
    rows = [json.loads(line) for line in (output / "prompts.jsonl").read_text().splitlines()]
    assert {row["id"] for row in rows} == {
        "pdbthink-heldout:validation:native-task",
        "pdbthink-heldout:test:native-task",
    }
    assert all(row["text"] == f"System:\n{system}\n\nUser:\n{user.strip()}" for row in rows)
    assert all(row["prompt_sha256"] == prompt_hash for row in rows)
    assert all(row["archive_sha256"] == hashlib.sha256(archive).hexdigest() for row in rows)
    assert all("never include this hidden answer" not in row["text"] for row in rows)
    receipt = json.loads((output / "reference-manifest.json").read_text())
    assert receipt["split_counts"] == {"validation": 1, "test": 1}
    assert receipt["records"] == 2
    assert receipt["prompts_sha256"] == hashlib.sha256((output / "prompts.jsonl").read_bytes()).hexdigest()


def test_reference_rejects_a_prompt_that_does_not_match_its_released_hash(tmp_path):
    config, *_ = reference_fixture(tmp_path, corrupt_prompt=True)
    with pytest.raises(ValueError, match="differs from its released hash"):
        prepare_reference(config)
    assert not (tmp_path / "reference/reference-manifest.json").exists()
