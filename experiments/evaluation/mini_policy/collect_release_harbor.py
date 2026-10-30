# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Package the selected native Harbor tasks with their private original verifiers."""

import asyncio
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

from harbor.models.task.id import GitTaskId, PackageTaskId
from harbor.tasks.client import TaskClient
from rigging.filesystem.buckets import filesystem_for
from taskcompendium.models import Source, TaskSpec

from experiments.evaluation.mini_policy.identities import document_identity

BASE = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy"
DEST = BASE + "/dataset-release-20261030"
BENCHMARKS = (
    "bfclparity-pi",
    "bixbench-pi",
    "ds-1000-local",
    "ot-tblite-recovery",
    "swebench-verified",
    "tau3-pi",
    "tb2-recovery",
)


def read_json(url):
    fs, path = filesystem_for(url)
    return json.loads(fs.cat(path))


def publish(name, value):
    content = json.dumps(value, ensure_ascii=False).encode()
    (Path(os.environ["IRIS_OUTPUT_DIR"]) / name).write_bytes(content)
    fs, path = filesystem_for(DEST + "/" + name)
    fs.pipe_file(path, content)


def files_digest(directory):
    return {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


async def main():
    # Import the standalone converter without initializing SkyRL training datasets.
    importer_path = Path(__file__).resolve().parents[3] / "vendor/skyrl/skyrl-train/skyrl_train/dataset/harbor_import.py"
    importer_spec = importlib.util.spec_from_file_location("mini_release_harbor_import", importer_path)
    importer = importlib.util.module_from_spec(importer_spec)
    sys.modules[importer_spec.name] = importer
    importer_spec.loader.exec_module(importer)
    selection = read_json(DEST + "/selection.json")
    tasks = selection["selection"]["840"]
    client = TaskClient()
    for benchmark in BENCHMARKS:
        wanted = {row["task_id"] for row in tasks if row["benchmark"] == benchmark}
        metadata = read_json(BASE + "/harbor-source-index-20261030/" + benchmark + ".json")
        sources = [
            dict(row, task_id=document_identity(benchmark, {}, row["task_name"]))
            for part in metadata["resolved"]
            for row in part["rows"]
        ]
        sources = [row for row in sources if row["task_id"] in wanted]
        assert len(sources) == len(wanted), (benchmark, wanted - {row["task_id"] for row in sources})
        ids = [
            (
                GitTaskId.model_validate(row["identity"])
                if "git_url" in row["identity"]
                else PackageTaskId.model_validate(row["identity"])
            )
            for row in sources
        ]
        root = Path(tempfile.mkdtemp(prefix="release-" + benchmark + "-"))
        result = await client.download_tasks(ids, output_dir=root / "sources", export=True)
        rows = []
        downloaded_by_name = {item.path.name: item for item in result.results}
        for original in sources:
            identity = original["identity"]
            source_name = Path(identity["path"]).name if "path" in identity else identity["name"]
            downloaded = downloaded_by_name[source_name]
            source = Source(
                dataset=benchmark,
                revision=identity.get("git_commit_id") or identity["ref"],
                row=original["task_id"],
                importer_revision="marin-815c4b2c81:mini-policy-release-v1",
            )
            spec = importer.harbor_trial_task(downloaded.path, source=source)
            spec = spec.model_copy(
                update={
                    "id": benchmark + "/" + original["task_id"],
                    "tags": (*spec.tags, "validation", "mini-eval-policy", "harness:pi"),
                }
            )
            TaskSpec.model_validate_json(spec.model_dump_json())
            restored = root / "roundtrip" / original["task_id"]
            importer.restore_harbor_trial_task(spec, restored)
            before, after = files_digest(downloaded.path), files_digest(restored)
            if before != after:
                raise ValueError("Original task package changed on TaskSpec roundtrip")
            rows.append(
                {
                    "benchmark": benchmark,
                    "task_id": original["task_id"],
                    "task_spec": spec.model_dump_json(),
                    "source": original,
                    "file_sha256": before,
                    "task_session": "harbor_trial",
                    "installed_agent": "pi",
                }
            )
        if {r["task_id"] for r in rows} != wanted:
            raise ValueError("Harbor download order or membership mismatch")
        publish("harbor-" + benchmark + ".json", {"benchmark": benchmark, "parent": metadata, "rows": rows})
        print(json.dumps({"benchmark": benchmark, "tasks": len(rows), "roundtrip": "exact"}), flush=True)
    publish("harbor-complete.json", {"complete": True})


if __name__ == "__main__":
    asyncio.run(main())
