# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove sources: archived Harbor tasks from the original ``open-thoughts/TaskTrove`` release."""

from taskcompendium.convert.tasktrove import TASKS_FILE, unpack_task_binary
from taskcompendium.pipeline.inputs import SourceFormat

from experiments.post_training.task_curation.pipeline import HfSource
from experiments.post_training.task_curation.source import DataSourceMetadata

TASKTROVE_REPO = "open-thoughts/TaskTrove"
TASKTROVE_REVISION = "02923004846e4e73862c20962f823a6d05100e7a"
TASKTROVE_RELEASE_REPO = "open-athena/task-trove"
TASKTROVE_RELEASE_REVISION = "ec049a4fb541ffbe5bbccb803e826563f5718dbf"
TASKTROVE_RELEASE_URL = f"https://huggingface.co/datasets/{TASKTROVE_RELEASE_REPO}"
TASKTROVE_MANIFEST_URL = f"{TASKTROVE_RELEASE_URL}/blob/{TASKTROVE_RELEASE_REVISION}/manifest.json"
TASKTROVE_RELEASE = DataSourceMetadata(
    id="",
    name="",
    origin="Task Trove",
    url=TASKTROVE_RELEASE_URL,
    dataset_id=TASKTROVE_RELEASE_REPO,
    revision=TASKTROVE_RELEASE_REVISION,
    revised_at="2026-10-08T09:34:47.000Z",
    environment="Harbor",
    type="Agentic",
    turns="Multi-turn",
    count_basis="Released Harbor tasks: manifest by_source.converted",
    count_precision="exact",
    count_url=TASKTROVE_MANIFEST_URL,
    recorded_at="2026-10-08",
    benchmark_basis="Release manifest does not designate benchmarks",
    family_basis="Task Trove release manifest source_verdicts.family",
    family_url=TASKTROVE_MANIFEST_URL,
    classification_basis="Task Trove tasks run as Agentic interactions in Harbor",
    canonical_url=TASKTROVE_RELEASE_URL,
    provenance_url=TASKTROVE_MANIFEST_URL,
)
ANSWER_FILE_DELIVERY = (
    ("write your final answer to `/app/answer.txt`", "return your final answer in the assistant response"),
    ("Write ONLY your final answer to **`/app/answer.txt`**", "Return ONLY your final answer in the assistant response"),
    ("The verifier reads that file", "The verifier reads the assistant response"),
)
"""The answer-file wording of the Nemotron Gym puzzle archives, and its reply-based replacement."""


def tasktrove_source(config: str) -> HfSource:
    """One TaskTrove config's task archives, unpacked into instruction, files and verifier data."""
    return HfSource(
        TASKTROVE_REPO, TASKTROVE_REVISION, (f"{config}/{TASKS_FILE}",), SourceFormat.PARQUET, decode=unpack_task_binary
    )
