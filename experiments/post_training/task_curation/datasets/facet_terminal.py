# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FACET-Terminal-Tasks-6k at its contract-repaired prepared-image pin.

The public upstream publishes 6,020 terminal tasks; the prepared-image pin converted
them to 6,010 canonical TaskSpecs whose environments pin the
``ghcr.io/marin-community/atlas-facet`` runtime images by digest. The prepared pin
bound no SHARED integrity contracts, so the versioned source repair
(``facet-terminal-contract-v1``, coordinator-approved 2026-11-24) republished every
TaskSpec with the RootDeliveryContract derived from its own pinned private verifier
files; that repaired pin is the declared conversion input. The census, controls
re-derivation, lexical scan, SAMPLE review and oracle-filtered admission all measure
it, and ``import_taskcompendium_prebuilt`` admits the pinned tasks without repeating
conversion. The oracle-filtered admission selects the sealed private
``open-athena/facet-terminal-tasks-6k-oracle-filtered-20261124-contract-v1`` release.
"""

from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.prebuilt import PrebuiltCohort, import_taskcompendium_prebuilt
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

UPSTREAM = "FACET-Terminal/FACET-Terminal-Tasks-6k"
UPSTREAM_REVISION = "b2d02645932e3989c8332a41e57d0f0855de7002"
PREPARED_REPO = "open-athena/facet-terminal-tasks-6k-prepared-images-20261124-contract-v1"
PREPARED_REVISION = "71fe5ec64454b53350ec9a8d281ce39627ab67db"
CONVERTER_VERSION = "2026.11.24.facet-contract-repair.1"
COUNT = 6010
SOURCE_ID = "FACET-Terminal:FACET-Terminal/FACET-Terminal-Tasks-6k"
REPAIR_PROTOCOL = "facet-terminal-contract-v1"


def sources() -> list[RlDataSource[PrebuiltCohort]]:
    return [
        RlDataSource(
            pipeline=import_taskcompendium_prebuilt,
            config=PrebuiltCohort(
                intended_use=IntendedUse.TRAIN,
                name="facet-terminal-tasks-6k",
                version=CONVERTER_VERSION,
                repo=PREPARED_REPO,
                revision=PREPARED_REVISION,
                expected_rows=COUNT,
            ),
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                origin=SourceOrigin.FACET_TERMINAL,
                upstream_dataset=UPSTREAM,
                upstream_name="FACET-Terminal-Tasks-6k",
                upstream_url=f"https://huggingface.co/datasets/{UPSTREAM}/tree/{UPSTREAM_REVISION}",
                build_path=(
                    "Prepared native-image pin with the versioned SHARED-contract repair "
                    f"({REPAIR_PROTOCOL}): RootDeliveryContracts derived from each package's pinned "
                    "private verifier files"
                ),
                family="terminal",
                tags=("license:apache-2.0", "facet-terminal", "terminal", "pristine-image", "root-delivery"),
                task_format=TaskFormat.TASKSPEC,
                count=COUNT,
                notes=(
                    "Canonical TaskSpecs at the contract-repair pin; pending the repaired chain's controls "
                    "re-derivation, SAMPLE qualification and admission."
                ),
            ),
        )
    ]
