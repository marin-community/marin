# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure the quiesce-safe facet_v2 release for its re-pin census.

Reads the published release's tasks.parquet directly (the RetainedCohort
pattern) and runs the shared declared-demand census over all 775 specs: the
actor Daytona machine shapes, the grading-environment assessment, and the
integrity-contract summary the actor census receipt binds. CPU-only; no
Daytona API calls. Writes one measurement document plus per-fact counters.
"""

import asyncio
import json
import os
import time
from collections import Counter

from pathlib import Path

START = time.monotonic()


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}Z +{time.monotonic() - START:7.3f}s] {message}", flush=True)


def main() -> None:
    log("facet re-pin census start")
    from finestore.rl.curation import DatasetRelease
    from huggingface_hub import snapshot_download
    from rigging.filesystem.s3_compat import configure_coreweave_s3
    from rigging.filesystem.storage_path import StoragePath

    configure_coreweave_s3()

    from experiments.post_training.task_curation.datasets.facet.tasks import VERSION
    from experiments.post_training.task_curation.daytona_census import measure_envelopes
    from experiments.post_training.task_curation.retained_release import RetainedCohort
    from experiments.post_training.task_curation.source import SourceReference
    from taskcompendium.pipeline.models import IntendedUse

    RELEASE_REPO = "open-athena/fweindel-facet-v2-quiesce-safe-20261124-v1"
    RELEASE_REVISION = os.environ["RELEASE_REVISION"]
    PARENT_REPO = "open-athena/fweindel-facet-v2-oracle-qualified-20261120-b-facet-v1"
    PARENT_REVISION = "a3c1eeae583f54544e3f50c00255d99623a6ff6f"
    OUTPUT = Path(os.environ["CENSUS_OUTPUT"])

    release = DatasetRelease(
        repo=RELEASE_REPO,
        revision=RELEASE_REVISION,
        url=f"https://huggingface.co/datasets/{RELEASE_REPO}/tree/{RELEASE_REVISION}",
    )
    cohort = RetainedCohort(
        name="facet-v2-quiesce-safe-20261124",
        version=VERSION,
        release=release,
        original_dataset=SourceReference(
            name=PARENT_REPO,
            revision=PARENT_REVISION,
            url=f"https://huggingface.co/datasets/{PARENT_REPO}/tree/{PARENT_REVISION}",
        ),
        expected_rows=775,
        tasks_parquet_sha256="f026ee23f5eee69a5d191f974a3bd72ed681d66c1f5d75afa9cc85a343d7c34d",
        intended_use=IntendedUse.TRAIN,
        evidence_url=f"https://huggingface.co/datasets/{RELEASE_REPO}",
    )

    root = snapshot_download(RELEASE_REPO, repo_type="dataset", revision=RELEASE_REVISION, allow_patterns=["tasks.parquet"])
    parquet = Path(root) / "tasks.parquet"

    import pyarrow.parquet as pq

    def documents():
        for batch in pq.ParquetFile(parquet).iter_batches(batch_size=16):
            for row in batch.to_pylist():
                yield json.loads(row["task_spec"]), json.loads(row["runtime"])

    log("measuring declared demands over 775 specs")
    result = measure_envelopes(documents())
    converted, machine_slots, sizes, failed_tasks, failures, mapping, roles, actor_sources, unsafe_counts = result
    log(
        f"converted={converted} machine_slots={machine_slots} distinct_sizes={len(sizes)} "
        f"failed_tasks={failed_tasks} mappings={dict(mapping)}"
    )
    payload = {
        "source_id": "FWeindel:validated-tasks/facet_v2",
        "release": {
            "repo": release.repo,
            "revision": release.revision,
            "url": release.url,
        },
        "parent_release": {"repo": PARENT_REPO, "revision": PARENT_REVISION},
        "population": {
            "dataset_id": RELEASE_REPO,
            "dataset_revision": RELEASE_REVISION,
            "tasks_parquet_sha256": cohort.tasks_parquet_sha256,
            "measured_tasks": converted,
            "machine_slots": machine_slots,
            "distinct_machine_shapes": len(sizes),
            "sizes": {json.dumps(k): v for k, v in sizes.items()},
            "failed_tasks": failed_tasks,
            "failures": dict(failures),
            "roles": roles,
            "actor_sources": dict(actor_sources),
            "unsafe_task_counts": dict(unsafe_counts),
        },
        "basis": (
            "declared-demand census over the quiesce-safe release's TaskSpec bytes (the "
            "RetainedCohort read path); no Daytona API calls; the grader collect repair does not "
            "change actor-facing requirements, measured here fresh at the new pin"
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "census-measurement.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    log(f"wrote {target}")
    upload()




def upload() -> None:
    """Re-run the measurement and publish the document to the evidence root."""
    import hashlib

    from rigging.filesystem.buckets import filesystem_for

    data = Path(os.environ["CENSUS_OUTPUT"], "census-measurement.json").read_bytes()
    root = "s3://marin-us-east-02a/marin/rl-data/atlas/facet-quiesce-repair/20261124"
    fs, key = filesystem_for(root)
    handle = fs.open(f"{key}/census-measurement.json", "wb", auto_mkdir=True)
    handle.write(data)
    handle.close()
    print("uploaded census-measurement.json sha256", hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    main()
