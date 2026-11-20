# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pin the released BFCL cohorts and stage their original oracle scripts."""

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from itertools import batched
from typing import Annotated

from marin.datakit.download.http_session import build_retrying_session
from pydantic import StringConstraints, TypeAdapter
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.post_training.task_curation.datasets.bfcl_structure_closure.cases import Split

REPOSITORY = "open-athena/Complex-Closure-RLVR-Snowball-67B-A2B-Artifacts"
REVISION = "d62ce3939fa900b2dad9344428bb7ac16917127a"
INVENTORY_FILE = "artifacts/data/oracle-inventory-v1.json"
SPLIT_FILE = "artifacts/release-review/task-split-audit.json"
TOKENIZER_FILES = (
    "artifacts/initializer/tokenizer.json",
    "artifacts/initializer/tokenizer_config.json",
    "artifacts/initializer/chat_template.jinja",
)
ORACLE_ROOT = (
    "https://raw.githubusercontent.com/laude-institute/harbor-datasets/"
    "6bedd7878dc5d6f3456b4d80b781eb3c2d84f262/datasets/bfcl"
)
COMPLEMENT_SHA256 = "59abec59ea96a73266257bb14299cda0c3f99a8f3f79f2ced83bc417098fa1e1"
ORACLE_DOWNLOAD_TIMEOUT = 30
ORACLES_PER_DOWNLOAD_TASK = 64

_TASK_ID = Annotated[str, StringConstraints(pattern=r"^\S+$")]
_SHA256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


@dataclass(frozen=True)
class BfclParent:
    task: _TASK_ID
    source_name: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")]
    cohort: Split
    solution_sha256: _SHA256

    def staged_path(self, inventory_file: StoragePath) -> StoragePath:
        return inventory_file.parent / "oracles" / f"{self.source_name}.sh"

    def verify(self, raw: bytes) -> bytes:
        if hashlib.sha256(raw).hexdigest() != self.solution_sha256:
            raise ValueError(f"BFCL oracle checksum mismatch for {self.source_name}")
        return raw


@dataclass(frozen=True)
class ComplementManifest:
    source_partition_sha256: _SHA256
    split_manifest_sha256: _SHA256
    records: tuple[BfclParent, ...]
    failures: tuple[object, ...]


@dataclass(frozen=True)
class CohortIds:
    train: tuple[_TASK_ID, ...]
    validation: tuple[_TASK_ID, ...]
    parity: tuple[_TASK_ID, ...]


@dataclass(frozen=True)
class SplitManifest:
    source_split_sha256: _SHA256
    task_ids: CohortIds


_COMPLEMENT = TypeAdapter(ComplementManifest)
_SPLIT = TypeAdapter(SplitManifest)


def _read_complement(path: StoragePath) -> ComplementManifest:
    complement = _COMPLEMENT.validate_json(path.read_bytes(), strict=True)
    if complement.source_partition_sha256 != COMPLEMENT_SHA256:
        raise ValueError("BFCL inventory is from a different complement")
    if complement.failures or not complement.records:
        raise ValueError("BFCL inventory contains no records or reports failed oracle reads")
    if len({parent.task for parent in complement.records}) != len(complement.records) or len(
        {parent.source_name for parent in complement.records}
    ) != len(complement.records):
        raise ValueError("BFCL complement contains duplicate task IDs or directory names")
    return complement


def cohort_parents(inventory_file: StoragePath, split_file: StoragePath, cohort: Split) -> Iterator[BfclParent]:
    """Validate the published split against the inventory of approved oracle scripts."""
    complement = _read_complement(inventory_file)
    split = _SPLIT.validate_json(split_file.read_bytes(), strict=True)
    if split.source_split_sha256 != complement.split_manifest_sha256:
        raise ValueError("BFCL split audit and complement were produced from different split manifests")
    train, validation, parity = (
        set(ids) for ids in (split.task_ids.train, split.task_ids.validation, split.task_ids.parity)
    )
    if (
        len(train) != len(split.task_ids.train)
        or len(validation) != len(split.task_ids.validation)
        or len(parity) != len(split.task_ids.parity)
    ):
        raise ValueError("BFCL split contains duplicate task IDs")
    if not train or not validation or not parity or train & validation or (train | validation) & parity:
        raise ValueError("BFCL training, validation and parity cohorts must be disjoint")
    if train != {parent.task for parent in complement.records if parent.cohort == Split.TRAIN} or validation != {
        parent.task for parent in complement.records if parent.cohort == Split.VALIDATION
    }:
        raise ValueError("BFCL train and validation IDs do not match the audited complement")
    return (parent for parent in complement.records if parent.cohort == cohort)


def _download_batch(parents: tuple[BfclParent, ...], *, inventory_file: StoragePath, oracle_root: str) -> int:
    with build_retrying_session() as session:
        for parent in parents:
            url = f"{oracle_root}/{parent.source_name}/solution/solve.sh"
            with session.get(url, timeout=ORACLE_DOWNLOAD_TIMEOUT) as response:
                response.raise_for_status()
                payload = parent.verify(response.content)
            path = parent.staged_path(inventory_file)
            path.write_bytes(payload, auto_mkdir=True)
    return len(parents)


@dataclass(frozen=True)
class OracleDownload:
    """Prepare verified scripts inside the shared Hugging Face download artifact."""

    oracle_root: str

    def __call__(self, path: str, context: ZephyrContext) -> None:
        inventory_file = StoragePath(path)
        parents = _read_complement(inventory_file).records
        download = partial(_download_batch, inventory_file=inventory_file, oracle_root=self.oracle_root)
        groups = list(batched(parents, ORACLES_PER_DOWNLOAD_TASK))
        result = context.execute(Dataset.from_list(groups).map(download), name="bfcl-oracle-download")
        if sum(result.results) != len(parents):
            raise ValueError("BFCL download did not stage every audited oracle")
