# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Atlas-admitted records that run on main through the retained reader.

The Atlas admitted these 12 source populations and their sealed retained
receipts resolve in the retained-pin table; each record carries its generated
identity, admitted pin and lineage notes, and ``sources()`` binds every record
to its pin entry and the retained reader, so the declaration lands together
with the import that runs it. The 21 import-pending sources whose retained
receipts do not resolve here (prebuilt cohorts, native Harbor packages,
converter-fingerprint pins) stay recorded in the coverage fixture's
``declaration_blockers`` until their reader PRs land.
"""

from dataclasses import dataclass
from functools import cache

from finestore.rl.curation import OracleValidation, SourceOrigin, source_identifier
from finestore.rl.status_taxonomy import DataRestriction

import experiments.post_training.task_curation.datasets.atlas_retained_pins as atlas_retained_pins
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, TaskFormat


@dataclass(frozen=True)
class AtlasAdmission:
    """One Atlas-admitted record, pending the import main needs to run it."""

    origin: SourceOrigin
    upstream_dataset: str
    upstream_name: str
    build_path: str
    task_format: TaskFormat
    family: str = ""
    subset: str | None = None
    variant: str | None = None
    tags: tuple[str, ...] = ()
    upstream_url: str | None = None
    oracle_validation: OracleValidation | None = None
    notes: str = ""

    def bound(self, pin: atlas_retained_pins.RetainedPin) -> RlDataSource:
        """Bind the record's identity to its sealed retained pin and the retained reader.

        The pin supplies the released dataset, evidence, censuses and license; the
        record supplies only what the pin cannot: the Atlas identity and its own
        notes.
        """
        info = SourceInfo(
            data_restriction=DataRestriction.OPEN,
            origin=self.origin,
            upstream_dataset=self.upstream_dataset,
            upstream_name=self.upstream_name,
            subset=self.subset,
            variant=self.variant,
            build_path=self.build_path,
            family=self.family,
            task_format=self.task_format,
            tags=self.tags,
            upstream_url=self.upstream_url,
            oracle_validation=self.oracle_validation,
            notes=self.notes,
        )
        return atlas_retained_pins.retained_source(info, pin, pin.original_pipeline.name)


RECORDS = (
    # AllenAI:IF_multi_constraints_upto5
    AtlasAdmission(
        origin=SourceOrigin.ALLENAI,
        upstream_dataset="IF_multi_constraints_upto5",
        upstream_name="IFBench training split",
        build_path="IFBench training intake",
        family="instruction-following",
        upstream_url=(
            "https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5/tree/2e3a77407b7fce69f95b248d"
            "64a884e3ae1c2423"
        ),
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # Nemotron:Nemotron-Cascade-2-RL-data/MOPD
    AtlasAdmission(
        origin=SourceOrigin.NEMOTRON,
        upstream_dataset="Nemotron-Cascade-2-RL-data",
        upstream_name="Nemotron Cascade-2 RL data",
        build_path="Cascade-2 training intake",
        family="mixed",
        subset="MOPD",
        upstream_url=(
            "https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-RL-data/tree/05bbaf03bac608e804efc86c5"
            "f8bf99844f2d197"
        ),
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # Nemotron:Nemotron-Cascade-2-RL-data/multi-domain-RL
    AtlasAdmission(
        origin=SourceOrigin.NEMOTRON,
        upstream_dataset="Nemotron-Cascade-2-RL-data",
        upstream_name="Nemotron Cascade-2 RL data",
        build_path="Cascade-2 training intake",
        family="mixed",
        subset="multi-domain-RL",
        upstream_url=(
            "https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-RL-data/tree/05bbaf03bac608e804efc86c5"
            "f8bf99844f2d197"
        ),
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # Nemotron:Nemotron-RL-instruction_following-structured_outputs
    AtlasAdmission(
        origin=SourceOrigin.NEMOTRON,
        upstream_dataset="Nemotron-RL-instruction_following-structured_outputs",
        upstream_name="NeMo structured outputs",
        build_path="Pinned NeMo native structured-output verifier with unchanged private schema",
        family="instruction_following",
        upstream_url=(
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following-structured_outputs/tree"
            "/4ed3d0e404c03dd47453217ce498c3511aa90a44"
        ),
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # OpenResearcher:OpenResearcher-Dataset
    AtlasAdmission(
        origin=SourceOrigin.OPENRESEARCHER,
        upstream_dataset="OpenResearcher-Dataset",
        upstream_name="OpenResearcher TRAIN",
        build_path="Grounded offline research intake",
        family="offline-search",
        upstream_url=(
            "https://huggingface.co/datasets/OpenResearcher/OpenResearcher-Dataset/tree/1edca18c5d7c9c105762d"
            "a4f3c5c2b83855b3c6e"
        ),
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # POLARIS-Project:Polaris-Dataset-53K
    AtlasAdmission(
        origin=SourceOrigin.POLARIS_PROJECT,
        upstream_dataset="Polaris-Dataset-53K",
        upstream_name="POLARIS-53K",
        build_path="POLARIS training intake",
        family="math-answer",
        upstream_url=(
            "https://huggingface.co/datasets/POLARIS-Project/Polaris-Dataset-53K/tree/296f8e34132e63f4a1d70e0"
            "dcc8bddebb43f03e4"
        ),
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # TaskTrove:tasktrove-knowledge-openqa-qa
    AtlasAdmission(
        origin=SourceOrigin.TASKTROVE,
        upstream_dataset="tasktrove-knowledge-openqa-qa",
        upstream_name="knowledge-openqa",
        build_path="Campaign Release",
        family="qa-short-answer",
        upstream_url="https://huggingface.co/datasets/open-thoughts/TaskTrove/tree/02923004846e4e73862c20962f823a6d05100e7a",
        oracle_validation=OracleValidation.VERIFIER_EXECUTED,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # TaskTrove:tasktrove-math_gym-qa
    AtlasAdmission(
        origin=SourceOrigin.TASKTROVE,
        upstream_dataset="tasktrove-math_gym-qa",
        upstream_name="math_gym",
        build_path="Campaign Release",
        family="math-answer",
        upstream_url="https://huggingface.co/datasets/open-thoughts/TaskTrove/tree/02923004846e4e73862c20962f823a6d05100e7a",
        oracle_validation=OracleValidation.VERIFIER_EXECUTED,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # TaskTrove:tasktrove-puzzles-qa
    AtlasAdmission(
        origin=SourceOrigin.TASKTROVE,
        upstream_dataset="tasktrove-puzzles-qa",
        upstream_name="tasktrove-puzzles-qa",
        build_path="Campaign Release",
        family="math-answer",
        upstream_url="https://huggingface.co/datasets/open-thoughts/TaskTrove/tree/02923004846e4e73862c20962f823a6d05100e7a",
        oracle_validation=OracleValidation.VERIFIER_EXECUTED,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # TaskTrove:tasktrove-safety-qa
    AtlasAdmission(
        origin=SourceOrigin.TASKTROVE,
        upstream_dataset="tasktrove-safety-qa",
        upstream_name="safety",
        build_path="Campaign Release",
        family="llm-judge-freeform",
        upstream_url="https://huggingface.co/datasets/open-thoughts/TaskTrove/tree/02923004846e4e73862c20962f823a6d05100e7a",
        oracle_validation=OracleValidation.VERIFIER_EXECUTED,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # TaskTrove:tasktrove-science-openqa-qa
    AtlasAdmission(
        origin=SourceOrigin.TASKTROVE,
        upstream_dataset="tasktrove-science-openqa-qa",
        upstream_name="science-openqa",
        build_path="Campaign Release",
        family="llm-judge-freeform",
        upstream_url="https://huggingface.co/datasets/open-thoughts/TaskTrove/tree/02923004846e4e73862c20962f823a6d05100e7a",
        oracle_validation=OracleValidation.VERIFIER_EXECUTED,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # pafitis:hardmath@rhs
    AtlasAdmission(
        origin=SourceOrigin.PAFITIS,
        upstream_dataset="hardmath",
        upstream_name="hardmath",
        build_path="MarinSkyRL",
        family="math-answer",
        variant="rhs",
        upstream_url=(
            "https://huggingface.co/datasets/pafitis/HARDMath_processed_training/tree/937e9f10356e31e854f6efb"
            "9a2507f1e200c8b25"
        ),
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
)


@cache
def _records_with_pins() -> tuple[RlDataSource, ...]:
    """Every record bound to its sealed retained pin."""
    pins = atlas_retained_pins.atlas_retained_pins()
    bound = []
    for record in RECORDS:
        generated = source_identifier(record.origin, record.upstream_dataset, record.subset, record.variant)
        pin = pins.get(generated)
        if pin is None:
            raise ValueError(f"No sealed retained pin for the Atlas-admitted record: {generated}")
        bound.append(record.bound(pin))
    return tuple(bound)


def sources() -> list[RlDataSource]:
    """The Atlas-admitted records declared at their sealed retained pins."""
    return list(_records_with_pins())


if __name__ == "__main__":
    raise SystemExit("regeneration runs through the retained-pin refresh; see atlas_retained_pins")
