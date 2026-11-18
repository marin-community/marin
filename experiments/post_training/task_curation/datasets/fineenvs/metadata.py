# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source identity and restriction declarations shared by FineEnvs intake stages."""

from finestore.rl.status_taxonomy import DataRestriction

from experiments.post_training.task_curation.source import SourceInfo, SourceOrigin, TaskFormat

DATASET = "FWeindel/validated-tasks"
REVISION = "317659ac88c8b706d318020c5fc3813a931fb6ec"
TASK_COUNTS = {"fineenvs_v2": 4001, "fineenvs_v3": 515}


def source_info(folder: str) -> SourceInfo:
    return SourceInfo(
        data_restriction=DataRestriction.OPEN,
        origin=SourceOrigin.FWEINDEL,
        upstream_dataset="validated-tasks",
        upstream_name="FineEnvs packaged-data tasks",
        upstream_url=f"https://huggingface.co/datasets/{DATASET}/tree/{REVISION}/{folder}",
        subset=folder,
        count=TASK_COUNTS[folder],
        family="terminal",
        build_path="Pinned native base image with checksum-verified grouped public inputs",
        task_format=TaskFormat.TASKCOMPENDIUM,
    )
