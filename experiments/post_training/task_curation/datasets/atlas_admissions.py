# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Atlas-admitted RL sources whose import path on main is still pending.

The Atlas declared these 32 source populations Available at the cut the
receipts resolved (2026-11-24). Each record carries its generated identity,
pinned release and stage-0 license receipt, and states its ``open`` restriction
tier explicitly. Their imports run through the Atlas's canonical and retained
prebuilt readers, which main does not carry yet, so the records are typed
``Atlas-admitted``: admitted upstream, never treated as runnable here until
such an import lands.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from finestore.rl.curation import (
    DatasetRelease,
    OracleValidation,
    SourceOrigin,
    huggingface_dataset_url,
    source_identifier,
)
from finestore.rl.licenses import LicenseReceipt
from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.source import (
    ATLAS_ADMITTED_TAG,
    RlDataSource,
    SourceInfo,
    SourceReference,
    TaskFormat,
)

RECEIPTS_PATH = Path(__file__).with_name("atlas_license_receipts.json")


@dataclass(frozen=True)
class AtlasAdmission:
    """One Atlas-admitted record, pending the import main needs to run it."""

    origin: SourceOrigin
    upstream_dataset: str
    upstream_name: str
    release_repo: str
    release_revision: str
    build_path: str
    intended_use: IntendedUse
    task_format: TaskFormat
    family: str = ""
    subset: str | None = None
    variant: str | None = None
    tags: tuple[str, ...] = ()
    count: int | None = None
    upstream_url: str | None = None
    daytona_snapshot_count: int | None = None
    daytona_count_basis: str | None = None
    oracle_validation: OracleValidation | None = None
    notes: str = ""

    def source(self, receipt: LicenseReceipt) -> RlDataSource:
        """Bind the receipt and pin, typed as admitted in the Atlas, import pending."""
        url = huggingface_dataset_url(self.release_repo, self.release_revision)
        release = DatasetRelease(repo=self.release_repo, revision=self.release_revision, url=url)
        return RlDataSource(
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                origin=self.origin,
                upstream_dataset=self.upstream_dataset,
                upstream_name=self.upstream_name,
                subset=self.subset,
                variant=self.variant,
                build_path=self.build_path,
                family=self.family,
                task_format=self.task_format,
                tags=(*self.tags, ATLAS_ADMITTED_TAG),
                count=self.count,
                upstream_url=self.upstream_url,
                dataset=SourceReference(release.repo, release.revision, release.url),
                current_release=release,
                intended_use=self.intended_use,
                license_receipt=receipt,
                daytona_snapshot_count=self.daytona_snapshot_count,
                daytona_count_basis=self.daytona_count_basis,
                oracle_validation=self.oracle_validation,
                notes=self.notes,
            ),
        )


RECORDS = (
    # AllenAI:IF_multi_constraints_upto5
    AtlasAdmission(
        origin=SourceOrigin.ALLENAI,
        upstream_dataset="IF_multi_constraints_upto5",
        upstream_name="IFBench training split",
        release_repo="open-athena/allenai-if-multi-constraints-upto5-7bcdabaf-retained-20261116-b-v1",
        release_revision="1b4076f63af29e34aee43e16781784890b1e9835",
        build_path="IFBench training intake",
        family="instruction-following",
        upstream_url=(
            "https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5/tree/2e3a77407b7fce69f95b248d"
            "64a884e3ae1c2423"
        ),
        daytona_count_basis=(
            "Complete 95357-TaskSpec census from open-athena/allenai-if-multi-constraints-upto5-7bcdabaf-reta"
            "ined-20261116-b-v1@1b4076f63af29e34aee43e16781784890b1e9835; native Harbor task.toml or"
            "published runtime allocations; actor only on Daytona, grader roles separate; no provider calls;"
            "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/retained-census/ifbench/2026.11.16.1/7bcdaba"
            "f57071c23bcdc/population"
        ),
        notes=(
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original"
            "oversight and quality evidence apply through this subset lineage:"
            "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/retained-publications/ifbench/2026.11.16.1/7"
            "bcdabaf57071c23bcdc/result.json. Original IFEvalG scorer 28c1816664491493242d453c2c7280ee56e6c20"
            "a; no released oracle; pending final-pin QA."
        ),
        tags=("license:odc-by",),
        count=95357,
        daytona_snapshot_count=0,
        intended_use=IntendedUse.TRAIN,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # Evalchemy:gsm8k-0shot
    AtlasAdmission(
        origin=SourceOrigin.EVALCHEMY,
        upstream_dataset="gsm8k-0shot",
        upstream_name="gsm8k-0shot",
        release_repo="openai/gsm8k",
        release_revision="740312add88f781978c0658806c59bc2815b9866",
        build_path="Evalchemy",
        family="short-answer",
        daytona_count_basis=(
            "Complete 1319-TaskSpec census from openai/gsm8k@740312add88f781978c0658806c59bc2815b9866; native"
            "Harbor task.toml or published runtime allocations; actor only on Daytona, grader roles separate;"
            "no provider calls; s3://marin-us-east-02a/marin/rl-data/atlas/lexical/text/7fb4b303eea969cd81a4/"
            "2026.11.19.3"
        ),
        tags=("eval", "source:evalchemy", "benchmark", "family:short-answer", "task:gsm8k"),
        count=1319,
        daytona_snapshot_count=0,
        intended_use=IntendedUse.EVAL,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # Evalchemy:mmlu-smoke
    AtlasAdmission(
        origin=SourceOrigin.EVALCHEMY,
        upstream_dataset="mmlu-smoke",
        upstream_name="mmlu-smoke",
        release_repo="cais/mmlu",
        release_revision="c30699e8356da336a370243923dbaf21066bb9fe",
        build_path="Evalchemy",
        family="likelihood-choice",
        daytona_count_basis=(
            "Complete 100-TaskSpec census from cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe; native"
            "Harbor task.toml or published runtime allocations; actor only on Daytona, grader roles separate;"
            "no provider calls; s3://marin-us-east-02a/marin/rl-data/atlas/lexical/text/52eb6f94e14e30ec7639/"
            "2026.11.12.2"
        ),
        tags=("eval", "source:evalchemy", "single-turn", "benchmark", "family:likelihood", "task:mmlu_abstract_algebra"),
        count=100,
        daytona_snapshot_count=0,
        intended_use=IntendedUse.EVAL,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # FWeindel:validated-tasks/fineenvs_v2
    AtlasAdmission(
        origin=SourceOrigin.FWEINDEL,
        upstream_dataset="validated-tasks",
        upstream_name="FineEnvs packaged-data tasks",
        release_repo="open-athena/fweindel-fineenvs-v2-oracle-qualified-20261120-b-v2",
        release_revision="d6c199c7b895c9f22153c03fbd02d044d25db6cb",
        build_path="Pinned native base image with checksum-verified grouped public inputs",
        family="terminal",
        subset="fineenvs_v2",
        upstream_url=(
            "https://huggingface.co/datasets/FWeindel/validated-tasks/tree/317659ac88c8b706d318020c5fc3813a93"
            "1fb6ec/fineenvs_v2"
        ),
        daytona_count_basis=(
            "Complete 2582-TaskSpec census from open-athena/fweindel-fineenvs-v2-oracle-qualified-20261120-b-"
            "v2@d6c199c7b895c9f22153c03fbd02d044d25db6cb; native Harbor task.toml or published runtime"
            "allocations; actor only on Daytona, grader roles separate; no provider calls;"
            "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/fweindel-fineenvs/2026.11.17.1/oracle-qualif"
            "ied-v2/fineenvs_v2/census/population"
        ),
        notes=(
            "Qualification retained 2582/4001. Typed exclusions: {'negative_control_pending': 2,"
            "'oracle_failed': 1412, 'resource_calibration:disk_full': 1, 'resource_calibration:infrastructure"
            "': 4}. Original source and prior retention evidence remain archived in the release receipt."
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original"
            "oversight and quality evidence apply through this subset lineage:"
            "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/fweindel-fineenvs/2026.11.17.1/publication-v"
            "2/bf2865351cf4fd57e757/result.json. "
        ),
        count=2582,
        daytona_snapshot_count=9,
        oracle_validation=OracleValidation.EXECUTED,
        intended_use=IntendedUse.TRAIN,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # FWeindel:validated-tasks/fineenvs_v3
    AtlasAdmission(
        origin=SourceOrigin.FWEINDEL,
        upstream_dataset="validated-tasks",
        upstream_name="FineEnvs packaged-data tasks",
        release_repo="open-athena/fweindel-fineenvs-v3-oracle-qualified-20261120-b-v2",
        release_revision="6c0d2834431e65823fd3ee08555d55aff6cf83f7",
        build_path="Pinned native base image with checksum-verified grouped public inputs",
        family="terminal",
        subset="fineenvs_v3",
        upstream_url=(
            "https://huggingface.co/datasets/FWeindel/validated-tasks/tree/317659ac88c8b706d318020c5fc3813a93"
            "1fb6ec/fineenvs_v3"
        ),
        daytona_count_basis=(
            "Complete 351-TaskSpec census from open-athena/fweindel-fineenvs-v3-oracle-qualified-20261120-b-v"
            "2@6c0d2834431e65823fd3ee08555d55aff6cf83f7; native Harbor task.toml or published runtime"
            "allocations; actor only on Daytona, grader roles separate; no provider calls;"
            "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/fweindel-fineenvs/2026.11.17.1/oracle-qualif"
            "ied-v2/fineenvs_v3/census/population"
        ),
        notes=(
            "Qualification retained 351/515. Typed exclusions: {'oracle_failed': 163,"
            "'resource_calibration:infrastructure': 1}. Original source and prior retention evidence remain"
            "archived in the release receipt. remediated: dedup/decon. Retained TaskSpec and runtime bytes"
            "match the original pin; original oversight and quality evidence apply through this subset"
            "lineage: s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/fweindel-fineenvs/2026.11.17.1/publ"
            "ication-v2/76d5b069f74e483412e3/result.json. "
        ),
        count=351,
        daytona_snapshot_count=9,
        oracle_validation=OracleValidation.EXECUTED,
        intended_use=IntendedUse.TRAIN,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # Lego-X:Terminal-Lego-15k
    AtlasAdmission(
        origin=SourceOrigin.LEGO_X,
        upstream_dataset="Terminal-Lego-15k",
        upstream_name="Terminal-Lego-15k",
        release_repo="open-athena/terminal-lego-root-delivery-qualified-20261122-b-v1",
        release_revision="01d71ba8b6dd0458c5552445c8b7b73b2d8fc4fa",
        build_path="Native task scripts and budgets with versioned root-delivery verifier transport",
        family="terminal",
        upstream_url="https://huggingface.co/datasets/Lego-X/Terminal-Lego-15k",
        daytona_count_basis=(
            "Complete 1216-TaskSpec census from open-athena/terminal-lego-root-delivery-qualified-20261122-b-"
            "v1@01d71ba8b6dd0458c5552445c8b7b73b2d8fc4fa; native Harbor task.toml or published runtime"
            "allocations; actor only on Daytona, grader roles separate; no provider calls;"
            "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/terminal-lego/2026.11.20.1/root-delivery-cen"
            "sus-correction-v1/qualified/census/population"
        ),
        notes=(
            "Qualification retained 1216/1241. Typed exclusions: {'negative_control_pending': 2,"
            "'oracle_failed': 19, 'resource_calibration:infrastructure': 1,"
            "'resource_calibration:verifier_timeout': 3}. Original source and prior retention evidence remain"
            "archived in the release receipt. Transport version 2026.11.22.root-delivery.1 preserves public"
            "prompts, archived source files and role budgets. Private verifier delivery and actor quiescence"
            "use the maintained root-delivery contract. Fresh DATA oracle and NoP filtration retains 1,216 of"
            "1,241 tasks; all 25 typed exclusions remain archived. Legacy repaired, unprocessed and"
            "policy-gap tasks remain outside this population and uncounted. All ten SAMPLE issues and the"
            "unavailable observation remain recorded."
        ),
        count=1216,
        daytona_snapshot_count=1216,
        oracle_validation=OracleValidation.EXECUTED,
        intended_use=IntendedUse.TRAIN,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # Lite-Coder:LiteCoder-Terminal-RL-preview
    AtlasAdmission(
        origin=SourceOrigin.LITE_CODER,
        upstream_dataset="LiteCoder-Terminal-RL-preview",
        upstream_name="LiteCoder Terminal RL preview",
        release_repo="open-athena/litecoder-file-state-qualified-20261123-b-v1",
        release_revision="ac69555e731ec72b83104f8c208aa9b7c88e1bc2",
        build_path="Native file-state task controls with RootDeliveryContract quiescence",
        family="terminal",
        upstream_url="https://huggingface.co/datasets/Lite-Coder/LiteCoder-Terminal-RL-preview",
        daytona_count_basis=(
            "Complete 25-task census with physical declarative execution; zero named snapshots required by"
            "shellbox.backends.daytona.machine.DaytonaDeclarativeMachineFactory. Exact task/runtime and"
            "owned-ID evidence: s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/litecoder-terminalworld/2"
            "026.11.22.1/file-state-admission-v1/declarative-v2/qualified/qualification.json SHA256"
            "0d6ac5741a80a1924d0cf88c767ce74a70d3178eb2e9026b06fd205b7b20b7c5; original image identities:"
            "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/litecoder-terminalworld/2026.11.22.1/file-st"
            "ate-qualified-publication-census-v1/census/actor/.artifact.json SHA256"
            "74b6c8e87b60460eca1676217c44effb8bbe5310d3539e8470a5b4c1327a7791."
        ),
        notes=(
            "Qualification retained 25/48. Typed exclusions: {'oracle_failed': 23}. Original source and prior"
            "retention evidence remain archived in the release receipt. Only the 48-task file-state execution"
            "scope is selected before filtration. Native prompts, archived task files and declared role"
            "budgets are preserved. RootDeliveryContract kills agent processes before verification;"
            "service-dependent rows are outside this population. Fresh original-oracle and physical NoP"
            "controls retain 25 tasks and record 23 oracle failures. The other 554 source rows remain held"
            "and uncounted, including service rows without a declared verifier-side restart command. All 25"
            "final tasks received content review: GOOD with one recorded concern about an exact-match DSA"
            "reference. The finding remains visible; no task or grader repair is claimed."
        ),
        count=25,
        daytona_snapshot_count=0,
        oracle_validation=OracleValidation.EXECUTED,
        intended_use=IntendedUse.TRAIN,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # Marin:forge-main-20261117
    AtlasAdmission(
        origin=SourceOrigin.MARIN,
        upstream_dataset="forge-main-20261117",
        upstream_name="Forge main split, 2026-11-17",
        release_repo="open-athena/forge-main-atlas-20261122-v1",
        release_revision="f5766a0e3da2320d30af7c5d6c245f1f1fa711d5",
        build_path="Forge / Campaign Release",
        family="synthetic-cross-domain",
        daytona_count_basis=(
            "Complete 3577-TaskSpec census from open-athena/forge-main-atlas-20261122-v1@f5766a0e3da2320d30af"
            "7c5d6c245f1f1fa711d5; native Harbor task.toml or published runtime allocations; actor only on"
            "Daytona, grader roles separate; no provider calls; s3://marin-us-east-02a/marin/rl-data/atlas/in"
            "gest-b/forge-main/2026.11.21.2/selected-qualification-v3/population"
        ),
        notes=(
            "Qualification retained 3577/4000. Typed exclusions: {'backend_contract_unsupported': 352,"
            "'negative_control_pending': 3, 'oracle_failed': 59, 'resource_calibration:infrastructure': 7,"
            "'resource_calibration:verifier_timeout': 1, 'shared_executor_required': 1}. Original source and"
            "prior retention evidence remain archived in the release receipt. 4,000 synthetic tasks written"
            "by forge from a 1,999-skill, 45-subject catalog (1,598 skills covered), selected coverage first"
            "from 4,378 accepts after dropping 244 that carried a reviewer note or missing evidence; the"
            "ranking favoured GLM-5.3 solve rates near one half. 955 in band, 3,021 too easy and 24 too hard"
            "for GLM-5.3 over eight trials; 2,978 answer-only, 670 Linux and 352 simulated-shell tasks."
            "Machines that forge's local sandbox recorded without an image are restored to the image their"
            "authors declared. The 2026.11.21.1 conversion re-roots the 6,855 workspace resources of 850"
            "tasks at their declared working directories, so the runtime's root install lands each file where"
            "its author declared it, and the 2026.11.21.2 conversion preserves each task's authored golden"
            "control as private oracle resources on 1,024 tasks (the card's revision history carries both)."
            "Not re-executed through the Atlas rollout path. The repository is private pending release."
        ),
        tags=("rlvr", "agentic", "synthetic", "campaign-release"),
        count=3577,
        daytona_snapshot_count=1,
        oracle_validation=OracleValidation.EXECUTED,
        intended_use=IntendedUse.TRAIN,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # Nemotron:Nemotron-Cascade-2-RL-data/MOPD
    AtlasAdmission(
        origin=SourceOrigin.NEMOTRON,
        upstream_dataset="Nemotron-Cascade-2-RL-data",
        upstream_name="Nemotron Cascade-2 RL data",
        release_repo="open-athena/nemotron-nemotron-cascade-2-rl-data-mopd-d9411399-retained-20261116-b-v1",
        release_revision="7ac886010b3706345337eed86330b484111cfbf8",
        build_path="Cascade-2 training intake",
        family="mixed",
        subset="MOPD",
        upstream_url=(
            "https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-RL-data/tree/05bbaf03bac608e804efc86c5"
            "f8bf99844f2d197"
        ),
        daytona_count_basis=(
            "Complete 3698-TaskSpec census from open-athena/nemotron-nemotron-cascade-2-rl-data-mopd-d9411399"
            "-retained-20261116-b-v1@7ac886010b3706345337eed86330b484111cfbf8; native Harbor task.toml or"
            "published runtime allocations; actor only on Daytona, grader roles separate; no provider calls;"
            "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/retained-census/cascade/2026.11.16.1/d941139"
            "928857a09768b/population"
        ),
        notes=(
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original"
            "oversight and quality evidence apply through this subset lineage:"
            "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/retained-publications/cascade/2026.11.16.1/d"
            "941139928857a09768b/result.json. Original NeMo terminal graders"
            "39d2bde61e426f93e4bc00c0255fc1d839f859e6; unsupported components retain typed rejections."
        ),
        tags=("license:odc-by",),
        count=3698,
        daytona_snapshot_count=0,
        intended_use=IntendedUse.TRAIN,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # Nemotron:Nemotron-Cascade-2-RL-data/multi-domain-RL
    AtlasAdmission(
        origin=SourceOrigin.NEMOTRON,
        upstream_dataset="Nemotron-Cascade-2-RL-data",
        upstream_name="Nemotron Cascade-2 RL data",
        release_repo="open-athena/nemotron-nemotron-cascade-2-rl-data-multi-domain-rl-ee15b18a-retained-20261116-b-v1",
        release_revision="52522171b690b25e50faa954619ab56362c0d55a",
        build_path="Cascade-2 training intake",
        family="mixed",
        subset="multi-domain-RL",
        upstream_url=(
            "https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-RL-data/tree/05bbaf03bac608e804efc86c5"
            "f8bf99844f2d197"
        ),
        daytona_count_basis=(
            "Complete 12903-TaskSpec census from open-athena/nemotron-nemotron-cascade-2-rl-data-multi-domain"
            "-rl-ee15b18a-retained-20261116-b-v1@52522171b690b25e50faa954619ab56362c0d55a; native Harbor"
            "task.toml or published runtime allocations; actor only on Daytona, grader roles separate; no"
            "provider calls; s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/retained-census/cascade/2026"
            ".11.16.1/ee15b18affec7822f712/population"
        ),
        notes=(
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original"
            "oversight and quality evidence apply through this subset lineage:"
            "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/retained-publications/cascade/2026.11.16.1/e"
            "e15b18affec7822f712/result.json. Original NeMo terminal graders"
            "39d2bde61e426f93e4bc00c0255fc1d839f859e6; unsupported components retain typed rejections."
        ),
        tags=("license:odc-by",),
        count=12903,
        daytona_snapshot_count=0,
        intended_use=IntendedUse.TRAIN,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # Nemotron:Nemotron-RL-instruction_following-structured_outputs
    AtlasAdmission(
        origin=SourceOrigin.NEMOTRON,
        upstream_dataset="Nemotron-RL-instruction_following-structured_outputs",
        upstream_name="NeMo structured outputs",
        release_repo=(
            "open-athena/nemotron-nemotron-rl-instruction-following-structured-o-6abcd1f2-retained-20261124-b"
            "-final-nemo"
        ),
        release_revision="be4cd53e7d59041e5a401b03a0fe4018ca401c29",
        build_path="Pinned NeMo native structured-output verifier with unchanged private schema",
        family="instruction_following",
        upstream_url=(
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following-structured_outputs/tree"
            "/4ed3d0e404c03dd47453217ce498c3511aa90a44"
        ),
        daytona_count_basis=(
            "Complete 9213-TaskSpec census from open-athena/nemotron-nemotron-rl-instruction-following-struct"
            "ured-o-6abcd1f2-retained-20261124-b-final-nemo@be4cd53e7d59041e5a401b03a0fe4018ca401c29; native"
            "Harbor task.toml or published runtime allocations; actor only on Daytona, grader roles separate;"
            "no provider calls; s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/nemo-gym-remaining/2026.1"
            "1.24.1/structured-release-sample-v1/census/population"
        ),
        notes=(
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original"
            "oversight and quality evidence apply through this subset lineage:"
            "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/nemo-gym-remaining/2026.11.24.1/structured-r"
            "elease-sample-v1/retention-publication/6abcd1f2134e7ef00f0d/result.json. The native TRAIN9437"
            "population converts9213 TaskSpecs and records224 native schema failures outside selection;"
            "validation512 is archived and not selected. Original prompts/private schema/source native"
            "verifier are unchanged. No per-task original answer exists; no generated oracle or empty"
            "liveness is required under the maintained no-oracle static-retention policy. Final-pin static"
            "retention preserves all9213. SAMPLE99reviewed/100attempted,12issues/1unavailable (within5%"
            "tolerance); every quality finding stays recorded. Six measured cross-source partial overlaps are"
            "below95% containment in both directions and stay directional advisory notes; no display fold or"
            "source task removal. Actor has no environment; separate native script grader declares its pinned"
            "build context; snapshot-census grader unresolved-build notes are retained, not relabelled as"
            "live execution support."
        ),
        count=9213,
        daytona_snapshot_count=0,
        intended_use=IntendedUse.TRAIN,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # Nemotron:nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1
    AtlasAdmission(
        origin=SourceOrigin.NEMOTRON,
        upstream_dataset="nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1",
        upstream_name="Nemotron-RL-Agentic-Terminal-Pivot-v1",
        release_repo="open-athena/atlas-terminal-pivot-oracle-20261115-v1",
        release_revision="749601f48a75f03d4068e5d2106ffed706f5fec7",
        build_path="NeMo terminal pivot intake",
        family="terminal-action",
        upstream_url=(
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1/tree/eaef2694464364"
            "4c8a3dbbf361ce6128142f5976"
        ),
        daytona_count_basis=(
            "Complete 31104-TaskSpec census from open-athena/atlas-terminal-pivot-oracle-20261115-v1@749601f4"
            "8a75f03d4068e5d2106ffed706f5fec7; native Harbor task.toml or published runtime allocations;"
            "actor only on Daytona, grader roles separate; no provider calls;"
            "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-a/2026.11.15.1/pivot-admission-v1/census/21a3a"
            "60dd01975fcee1f/population"
        ),
        notes=(
            "Qualification retained 31104/31104. Typed exclusions: {}. Original source and prior retention"
            "evidence remain archived in the release receipt. remediated: dedup/decon. Retained TaskSpec and"
            "runtime bytes match the original pin; original oversight and quality evidence apply through this"
            "subset lineage: s3://marin-us-east-02a/marin/rl-data/atlas/retained-publications/ingest-a/2026.1"
            "1.15.1/21a3a60dd01975fcee1f/result.json. Frozen next-action replies with the native string-only"
            "NeMo scorer. No terminal environment is executed. The 0.9 keystroke-similarity reward can reject"
            "valid alternative command strings; adding a harmless shell comment to gold commands changes"
            "reward from 1 to 0. Evidence: s3://marin-rno2a/marin/rl-data/atlas/ingest-a/2026.11.15.1/pivot-w"
            "eak-grader-v1/evidence.json."
        ),
        tags=("license:cc-by-4.0",),
        count=31104,
        daytona_snapshot_count=0,
        oracle_validation=OracleValidation.EXECUTED,
        intended_use=IntendedUse.TRAIN,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # OpenAthena:pdbthink-coordinate-tasks
    AtlasAdmission(
        origin=SourceOrigin.OPENATHENA,
        upstream_dataset="pdbthink-coordinate-tasks",
        upstream_name="PDBThink v1.3.0 coordinate tasks",
        release_repo="open-athena/pdbthink-coordinate-tasks-oracle-qualified-20261122-b-v1",
        release_revision="25265b8ecb2155015eae283fa383452d1a439ffa",
        build_path="Original tool-free coordinate tasks",
        family="protein-coordinate",
        upstream_url="https://huggingface.co/datasets/open-athena/pdbthink-coordinate-tasks",
        daytona_count_basis=(
            "Complete 46805-TaskSpec census from open-athena/pdbthink-coordinate-tasks-oracle-qualified-20261"
            "122-b-v1@25265b8ecb2155015eae283fa383452d1a439ffa; native Harbor task.toml or published runtime"
            "allocations; actor only on Daytona, grader roles separate; no provider calls;"
            "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/pdbthink/2026.11.16.1/oracle-qualified-publi"
            "cation-census-v1/census/population"
        ),
        notes=(
            "Qualification retained 46805/80460. Typed exclusions: {'resource_calibration:infrastructure':"
            "33651, 'resource_calibration:verifier_timeout': 4}. Original source and prior retention evidence"
            "remain archived in the release receipt. remediated: dedup/decon. Retained TaskSpec and runtime"
            "bytes match the original pin; original oversight and quality evidence apply through this subset"
            "lineage: s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/pdbthink/2026.11.16.1/daytona-reten"
            "tion-publication-v1/publication/5066941295e4fe5409cc/result.json. Original train split only;"
            "exact native budgets and provider status required; original verifiers and oracles preserved."
            "User ruling 2026-11-17 versions grader allocation to 1 CPU/1024 MiB/1024 MiB with network deny;"
            "task allocation, prompt, grader logic and payload unchanged. Ruling:"
            "/Users/benfeuer/Documents/experiments/active/atlas-rulings/pdbthink-daytona-limits-20261117.md."
            "Infrastructure-held rows remain uncounted (33651 total, including 32794 legacy-held); 4 source"
            "verifier timeouts excluded. NoP is not applicable to answer-graded tasks."
        ),
        tags=("single-turn", "tool-free", "license:apache-2.0"),
        count=46805,
        daytona_snapshot_count=0,
        oracle_validation=OracleValidation.EXECUTED,
        intended_use=IntendedUse.TRAIN,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # OpenResearcher:OpenResearcher-Dataset
    AtlasAdmission(
        origin=SourceOrigin.OPENRESEARCHER,
        upstream_dataset="OpenResearcher-Dataset",
        upstream_name="OpenResearcher TRAIN",
        release_repo="open-athena/openresearcher-openresearcher-dataset-8d7d09e1-retained-20261121-bootstrap-v1",
        release_revision="8d5a48bf4994f861c903cc90f34e1530c25ea052",
        build_path="Grounded offline research intake",
        family="offline-search",
        upstream_url=(
            "https://huggingface.co/datasets/OpenResearcher/OpenResearcher-Dataset/tree/1edca18c5d7c9c105762d"
            "a4f3c5c2b83855b3c6e"
        ),
        daytona_count_basis=(
            "Complete 283-TaskSpec census from open-athena/openresearcher-openresearcher-dataset-8d7d09e1-ret"
            "ained-20261121-bootstrap-v1@8d5a48bf4994f861c903cc90f34e1530c25ea052; native Harbor task.toml or"
            "published runtime allocations; actor only on Daytona, grader roles separate; no provider calls;"
            "s3://marin-us-east-02a/marin/rl-data/atlas/consumer/20261121/openresearcher-bootstrap-v2/retaine"
            "d-census/8d7d09e17e8f7692ed07/population"
        ),
        notes=(
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original"
            "oversight and quality evidence apply through this subset lineage:"
            "s3://marin-us-east-02a/marin/rl-data/atlas/consumer/20261121/openresearcher-bootstrap-v2/publica"
            "tions/8d7d09e17e8f7692ed07/result.json. "
        ),
        tags=("license:mit", "corpus-license:odc-by"),
        count=283,
        daytona_snapshot_count=1,
        intended_use=IntendedUse.TRAIN,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # POLARIS-Project:Polaris-Dataset-53K
    AtlasAdmission(
        origin=SourceOrigin.POLARIS_PROJECT,
        upstream_dataset="Polaris-Dataset-53K",
        upstream_name="POLARIS-53K",
        release_repo="open-athena/polaris-project-polaris-dataset-53k-b25e0cb8-retained-20261116-b-v1",
        release_revision="9079512fe3d2d332b446b1b37deb9d2898f5461e",
        build_path="POLARIS training intake",
        family="math-answer",
        upstream_url=(
            "https://huggingface.co/datasets/POLARIS-Project/Polaris-Dataset-53K/tree/296f8e34132e63f4a1d70e0"
            "dcc8bddebb43f03e4"
        ),
        daytona_count_basis=(
            "Complete 53167-TaskSpec census from open-athena/polaris-project-polaris-dataset-53k-b25e0cb8-ret"
            "ained-20261116-b-v1@9079512fe3d2d332b446b1b37deb9d2898f5461e; native Harbor task.toml or"
            "published runtime allocations; actor only on Daytona, grader roles separate; no provider calls;"
            "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/polaris/2026.11.16.1/retained-census/2026.11"
            ".16.1/b25e0cb84e99b1f26085/population"
        ),
        notes=(
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original"
            "oversight and quality evidence apply through this subset lineage:"
            "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/polaris/2026.11.16.1/publication/2026.11.16."
            "1/b25e0cb84e99b1f26085/result.json. Original deterministic DeepScaleR reward"
            "eda12fe22dd83aca59ec46a67147240dc483f62b; LLM fallback disabled."
        ),
        tags=("rlvr", "single-turn", "license:apache-2.0"),
        count=53167,
        daytona_snapshot_count=0,
        intended_use=IntendedUse.TRAIN,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # RST:laion/rts-1k-rootless-apptainer
    AtlasAdmission(
        origin=SourceOrigin.RST,
        upstream_dataset="laion/rts-1k-rootless-apptainer",
        upstream_name="Recursive Task Synthesis validated RTS-1K",
        release_repo="open-athena/atlas-oracle-1e0af0ebe59e0eb380bf-20261117of1",
        release_revision="3fe05149a182c3d9ab141ccc7ef2b8fb27b44d00",
        build_path="Original-root TaskSpec with grading-time verifier delivery",
        family="terminal",
        upstream_url=(
            "https://huggingface.co/datasets/laion/rts-1k-rootless-apptainer/tree/6dbb24eedc0d7ffe5431e1b052b"
            "0de8e19eaf6fe"
        ),
        daytona_count_basis=(
            "Complete 623-TaskSpec census from open-athena/atlas-oracle-1e0af0ebe59e0eb380bf-20261117of1@3fe0"
            "5149a182c3d9ab141ccc7ef2b8fb27b44d00; native Harbor task.toml or published runtime allocations;"
            "actor only on Daytona, grader roles separate; no provider calls;"
            "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-a/2026.11.17.1/rst-multiswe-passing-census-v1/"
            "1e0af0ebe59e0eb380bf/population"
        ),
        notes=(
            "Qualification retained 623/984. Typed exclusions: {'oracle_failed': 237,"
            "'resource_calibration:infrastructure': 124}. Original source and prior retention evidence remain"
            "archived in the release receipt. remediated: dedup/decon. Retained TaskSpec and runtime bytes"
            "match the original pin; original oversight and quality evidence apply through this subset"
            "lineage: s3://marin-us-east-02a/marin/rl-data/atlas/ingest-a/2026.11.16.1/rst-multiswe-retained-"
            "publication-v1/1e0af0ebe59e0eb380bf/result.json. 984 source-validated packages retain original"
            "root actors, instructions, builds and solutions. The original verifier is host-delivered at"
            "grading time and its reward copied for numeric parsing. Mutable Dockerfile bases and build"
            "downloads use explicit Daytona EVAL declarative provisioning. Pending current-pin execution"
            "qualification and admission."
        ),
        tags=("license:cc-by-4.0", "recursive-task-synthesis"),
        count=623,
        daytona_snapshot_count=503,
        oracle_validation=OracleValidation.EXECUTED,
        intended_use=IntendedUse.TRAIN,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # TaskTrove:DCAgent2__nl2bash-tasks-cleaned-oracle-v2@repaired-20261118
    AtlasAdmission(
        origin=SourceOrigin.TASKTROVE,
        upstream_dataset="DCAgent2__nl2bash-tasks-cleaned-oracle-v2",
        upstream_name="DCAgent2__nl2bash-tasks-cleaned-oracle-v2",
        release_repo="open-athena/atlas-repair-nl2bash-20261118-oracle-filter-v1",
        release_revision="4cf30f2874864d8549c7adfa7f6516cf414825a5",
        build_path="Repair Lane",
        family="shell-cmd",
        variant="repaired-20261118",
        upstream_url="https://huggingface.co/datasets/open-thoughts/TaskTrove/tree/02923004846e4e73862c20962f823a6d05100e7a",
        daytona_count_basis=(
            "Complete 372-TaskSpec census from open-athena/atlas-repair-nl2bash-20261118-oracle-filter-v1@4cf"
            "30f2874864d8549c7adfa7f6516cf414825a5; native Harbor task.toml or published runtime allocations;"
            "actor only on Daytona, grader roles separate; no provider calls;"
            "s3://marin-us-east-02a/marin/rl-data/atlas/repair-candidates/20261116/nl2bash-seal-census-v2/pop"
            "ulation"
        ),
        notes=(
            "Qualification retained 372/372. Typed exclusions: {}. Original source and prior retention"
            "evidence remain archived in the release receipt. Repaired sub-population of"
            "TaskTrove:DCAgent2__nl2bash-tasks-cleaned-oracle-v2: the 372 comparative-review winners rebuilt"
            "through the tracked tasktrove-nl2bash recipe v3 at converter 900cb18689, admitted by the"
            "full-population oracle campaign with physical nop controls. The tracked source itself is"
            "unchanged. Actor census pending: input or converter mismatch: expected"
            "('TaskTrove:DCAgent2__nl2bash-tasks-cleaned-oracle-v2@repaired-20261118',"
            "'open-thoughts/TaskTrove', '02923004846e4e73862c20962f823a6d05100e7a', '3'), measured"
            "('TaskTrove:DCAgent2__nl2bash-tasks-cleaned-oracle-v2@repaired-20261118',"
            "'open-athena/atlas-repair-nl2bash-20261118-oracle-filter-v1',"
            "'4cf30f2874864d8549c7adfa7f6516cf414825a5', '3')."
        ),
        tags=("agentic", "language:bash", "repair-release", "converted"),
        count=372,
        daytona_snapshot_count=1,
        oracle_validation=OracleValidation.EXECUTED,
        intended_use=IntendedUse.TRAIN,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # TaskTrove:DCAgent__exp_rpt_pymethods2test-large-v2@repaired-20261118
    AtlasAdmission(
        origin=SourceOrigin.TASKTROVE,
        upstream_dataset="DCAgent__exp_rpt_pymethods2test-large-v2",
        upstream_name="DCAgent__exp_rpt_pymethods2test-large-v2",
        release_repo="open-athena/atlas-repair-pymethods-20261118-oracle-filter-v1",
        release_revision="3699e8ea33bda5774a045db8a25f09aa878628a8",
        build_path="Repair Lane",
        family="unit-test-gen",
        variant="repaired-20261118",
        upstream_url="https://huggingface.co/datasets/open-thoughts/TaskTrove/tree/02923004846e4e73862c20962f823a6d05100e7a",
        daytona_count_basis=(
            "Complete 3754-TaskSpec census from open-athena/atlas-repair-pymethods-20261118-oracle-filter-v1@"
            "3699e8ea33bda5774a045db8a25f09aa878628a8; native Harbor task.toml or published runtime"
            "allocations; actor only on Daytona, grader roles separate; no provider calls;"
            "s3://marin-us-east-02a/marin/rl-data/atlas/repair-candidates/20261116/pymethods-seal-census-v2/p"
            "opulation"
        ),
        notes=(
            "Qualification retained 3754/3754. Typed exclusions: {}. Original source and prior retention"
            "evidence remain archived in the release receipt. Repaired sub-population of"
            "TaskTrove:DCAgent__exp_rpt_pymethods2test-large-v2: the 3,754 comparative-review winners rebuilt"
            "through the tracked tasktrove-pymethods_large recipe v5 at converter 900cb18689 (wrapper"
            "solve.sh normalization applied), admitted by the full-population oracle campaign with physical"
            "nop controls. The tracked source itself is unchanged. Actor census pending: input or converter"
            "mismatch: expected ('TaskTrove:DCAgent__exp_rpt_pymethods2test-large-v2@repaired-20261118',"
            "'open-thoughts/TaskTrove', '02923004846e4e73862c20962f823a6d05100e7a', '5'), measured"
            "('TaskTrove:DCAgent__exp_rpt_pymethods2test-large-v2@repaired-20261118',"
            "'open-athena/atlas-repair-pymethods-20261118-oracle-filter-v1',"
            "'3699e8ea33bda5774a045db8a25f09aa878628a8', '5')."
        ),
        tags=("agentic", "language:python", "repair-release", "converted"),
        count=3754,
        daytona_snapshot_count=1,
        oracle_validation=OracleValidation.EXECUTED,
        intended_use=IntendedUse.TRAIN,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # TaskTrove:laion__exp_rpt_crosscodeeval-python-v2
    AtlasAdmission(
        origin=SourceOrigin.TASKTROVE,
        upstream_dataset="laion__exp_rpt_crosscodeeval-python-v2",
        upstream_name="laion/exp_rpt_crosscodeeval-python-v2",
        release_repo="open-athena/crosscode-python-b3fb68e4-oracle-filter-20261114-v1",
        release_revision="5c1270d7d2ec851e65d360e8efe752c556fe14ea",
        build_path="Native Harbor QUICK conversion",
        family="code-completion",
        upstream_url="https://huggingface.co/datasets/open-thoughts/TaskTrove/tree/02923004846e4e73862c20962f823a6d05100e7a",
        daytona_count_basis=(
            "Complete 500-TaskSpec census from open-athena/crosscode-python-b3fb68e4-oracle-filter-20261114-v"
            "1@5c1270d7d2ec851e65d360e8efe752c556fe14ea; native Harbor task.toml or published runtime"
            "allocations; actor only on Daytona, grader roles separate; no provider calls;"
            "s3://marin-us-east-02a/marin/rl-data/atlas/oracle-filter-code/2026.11.15.162-six-final-pin-popul"
            "ations/b3fb68e474a864d2f59c/population"
        ),
        notes=(
            "Qualification retained 500/500. Typed exclusions: {}. Original source and prior retention"
            "evidence remain archived in the release receipt. 0.25 for any non-empty output; instruction"
            "discloses the tiers. Evaluation only for CrossCodeEval-derived sources: Python's native tests"
            "contain the Terminal-Bench canary. An inspected Python tests/test_state.py marker is preserved"
            "at private pin a953f93bc830704525ff2cce0b296dd262009662, file SHA-256"
            "335184f930fd957094b86f8f5402465766c40674de9044e2feb6faf21d904789. Within-source dedup applies;"
            "lexical contamination is measured only. Native Harbor package bytes are preserved in a private"
            "TaskCompendium QUICK release. QUICK supplies no oracle or admission claim. Native Dockerfile"
            "base tags and build downloads may remain mutable. Source defects remain recorded."
        ),
        tags=("agentic", "single-turn", "eval"),
        count=500,
        daytona_snapshot_count=1,
        oracle_validation=OracleValidation.EXECUTED,
        intended_use=IntendedUse.EVAL,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # TaskTrove:tasktrove-knowledge-openqa-qa
    AtlasAdmission(
        origin=SourceOrigin.TASKTROVE,
        upstream_dataset="tasktrove-knowledge-openqa-qa",
        upstream_name="knowledge-openqa",
        release_repo="open-athena/campaign-release-tasktrove-knowledge-openqa-qa-2b5961f4-retained-20261116-v1",
        release_revision="9d3fbb17806c98a9d6c4299807ab586f3620cad9",
        build_path="Campaign Release",
        family="qa-short-answer",
        upstream_url="https://huggingface.co/datasets/open-thoughts/TaskTrove/tree/02923004846e4e73862c20962f823a6d05100e7a",
        daytona_count_basis=(
            "Complete 122314-TaskSpec census from open-athena/campaign-release-tasktrove-knowledge-openqa-qa-"
            "2b5961f4-retained-20261116-v1@9d3fbb17806c98a9d6c4299807ab586f3620cad9; native Harbor task.toml"
            "or published runtime allocations; actor only on Daytona, grader roles separate; no provider"
            "calls; s3://marin-us-east-02a/marin/rl-data/atlas/retained-census/2026.11.16.1-r1/2b5961f4fdc08f"
            "3d4218/population"
        ),
        notes=(
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original"
            "oversight and quality evidence apply through this subset lineage:"
            "s3://marin-us-east-02a/marin/rl-data/atlas/private-retention/2026.11.16.1/2b5961f4fdc08f3d4218/r"
            "esult.json. Exact gate then reference-based judge. No endpoint is hard-coded in any task spec;"
            "the MEDIUM JUDGE_TIERS definition pins model and completion settings. Private release;"
            "conversation-only tasks import canonical task specs. The pinned cohort revision was downloaded"
            "back and every task hash re-validated."
        ),
        tags=("rlvr", "single-turn", "campaign-release", "converted"),
        count=122314,
        daytona_snapshot_count=0,
        oracle_validation=OracleValidation.VERIFIER_EXECUTED,
        intended_use=IntendedUse.TRAIN,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # TaskTrove:tasktrove-math_gym-qa
    AtlasAdmission(
        origin=SourceOrigin.TASKTROVE,
        upstream_dataset="tasktrove-math_gym-qa",
        upstream_name="math_gym",
        release_repo="open-athena/campaign-release-tasktrove-math-gym-qa-03ef831a-retained-2026.11.16.1-mathgym",
        release_revision="2ec57f36b9ca674c30ec7a5fa514bebf242cc97d",
        build_path="Campaign Release",
        family="math-answer",
        upstream_url="https://huggingface.co/datasets/open-thoughts/TaskTrove/tree/02923004846e4e73862c20962f823a6d05100e7a",
        daytona_count_basis=(
            "Complete 3922-TaskSpec census from open-athena/campaign-release-tasktrove-math-gym-qa-03ef831a-r"
            "etained-2026.11.16.1-mathgym@2ec57f36b9ca674c30ec7a5fa514bebf242cc97d; native Harbor task.toml"
            "or published runtime allocations; actor only on Daytona, grader roles separate; no provider"
            "calls; s3://marin-us-east-02a/marin/rl-data/atlas/retained-census/2026.11.16.1-mathgym/03ef831ab"
            "6bc99825711/population"
        ),
        notes=(
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original"
            "oversight and quality evidence apply through this subset lineage:"
            "s3://marin-us-east-02a/marin/rl-data/atlas/private-retention/2026.11.16.1-mathgym/03ef831ab6bc99"
            "825711/result.json. Strict trailing boxed compare with no oracle. 155 rows whose typed"
            "references the production math-verify parser rejects are excluded at admission. Private release;"
            "conversation-only tasks import canonical task specs. The pinned cohort revision was downloaded"
            "back and every task hash re-validated. Content-quality finding 2026-11-18 (SAMPLE receipt"
            "fb053f6439…, pinned GLM-5.3 v3, seed-0 panel at the r23 pin): 91 of 3,941 current-population"
            "rows (2.3%) carry LaTeX-wrapped expected-answer strings (\\(\\displaystyle …\\)) in grader"
            "parameters.expected with math_type=scalar, which the scalar comparison path cannot parse; the"
            "SAMPLE review flags 1 of its 8 issues naming that symptom (tasktrove-math_gym-bb15c01fe058e37e…)"
            ". Grader and scorer stay unchanged (faithful environment). A private remediated version"
            "normalizing ONLY those expected-answer strings remains available as data repair if ruled"
            "material; it would need its own SAMPLE QA and admission, and this original stays as is."
        ),
        tags=("rlvr", "single-turn", "campaign-release", "converted"),
        count=3922,
        daytona_snapshot_count=0,
        oracle_validation=OracleValidation.VERIFIER_EXECUTED,
        intended_use=IntendedUse.TRAIN,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # TaskTrove:tasktrove-math_openreasoning-qa
    AtlasAdmission(
        origin=SourceOrigin.TASKTROVE,
        upstream_dataset="tasktrove-math_openreasoning-qa",
        upstream_name="math_openreasoning",
        release_repo="open-athena/campaign-release-tasktrove-math-openreasoning-qa-45247a4a-oracle-filtered-20261114-v2",
        release_revision="5528a398ba2372d53f803ce3e93c6cca2f4b4aae",
        build_path="Campaign Release",
        family="math-answer",
        upstream_url="https://huggingface.co/datasets/open-thoughts/TaskTrove/tree/02923004846e4e73862c20962f823a6d05100e7a",
        daytona_count_basis=(
            "Complete 42618-TaskSpec census from open-athena/campaign-release-tasktrove-math-openreasoning-qa"
            "-45247a4a-oracle-filtered-20261114-v2@5528a398ba2372d53f803ce3e93c6cca2f4b4aae; native Harbor"
            "task.toml or published runtime allocations; actor only on Daytona, grader roles separate; no"
            "provider calls; s3://marin-us-east-02a/marin/rl-data/atlas/oracle-filter-qa/2026.11.14.2-final-p"
            "rivate/45247a4ae50647945db9/population"
        ),
        notes=(
            "Qualification retained 42618/42618. Typed exclusions: {}. Original source and prior retention"
            "evidence remain archived in the release receipt. remediated: dedup/decon. Retained TaskSpec and"
            "runtime bytes match the original pin; original oversight and quality evidence apply through this"
            "subset lineage: s3://marin-us-east-02a/marin/rl-data/atlas/private-retention/2026.11.13.1/bf2191"
            "4ea9425c3754f8/result.json. Open-reasoning problems with a rigorous"
            "scalar/interval/set/tuple/equation verifier and oracle. 15 rows whose typed references the"
            "production math-verify parser rejects are excluded at admission. QA (RL Data Atlas): blinded"
            "review flagged 2 sampled rows as underdetermined; both were compared against the raw archive and"
            "the converted statements match the upstream instructions verbatim, so the defects are in the"
            "source data, not the conversion. 2 of the 20-case blinded panel; exactly the inclusive 10%"
            "boundary. Private release; conversation-only tasks import canonical task specs. The pinned"
            "cohort revision was downloaded back and every task hash re-validated."
        ),
        tags=("rlvr", "single-turn", "campaign-release", "converted"),
        count=42618,
        daytona_snapshot_count=0,
        oracle_validation=OracleValidation.EXECUTED,
        intended_use=IntendedUse.TRAIN,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # TaskTrove:tasktrove-math_oracle-qa
    AtlasAdmission(
        origin=SourceOrigin.TASKTROVE,
        upstream_dataset="tasktrove-math_oracle-qa",
        upstream_name="math_oracle",
        release_repo="open-athena/campaign-release-tasktrove-math-oracle-qa-7f7f7b29-oracle-filtered-20261114-v2",
        release_revision="6656fc7bfaa524198f8a52ae7db3165c69ab5301",
        build_path="Campaign Release",
        family="math-answer",
        upstream_url="https://huggingface.co/datasets/open-thoughts/TaskTrove/tree/02923004846e4e73862c20962f823a6d05100e7a",
        daytona_count_basis=(
            "Complete 57419-TaskSpec census from open-athena/campaign-release-tasktrove-math-oracle-qa-7f7f7b"
            "29-oracle-filtered-20261114-v2@6656fc7bfaa524198f8a52ae7db3165c69ab5301; native Harbor task.toml"
            "or published runtime allocations; actor only on Daytona, grader roles separate; no provider"
            "calls; s3://marin-us-east-02a/marin/rl-data/atlas/oracle-filter-qa/2026.11.14.2-final-private/7f"
            "7f7b292c811b9e2d3d/population"
        ),
        notes=(
            "Qualification retained 57419/57419. Typed exclusions: {}. Original source and prior retention"
            "evidence remain archived in the release receipt. remediated: dedup/decon. Retained TaskSpec and"
            "runtime bytes match the original pin; original oversight and quality evidence apply through this"
            "subset lineage: s3://marin-us-east-02a/marin/rl-data/atlas/private-retention/2026.11.13.1/940292"
            "d1b2df69913683/result.json. 5,000-scale scalar sympy compare with hidden gold references. 353"
            "rows whose typed references the production math-verify parser rejects are excluded at admission."
            "Private release; conversation-only tasks import canonical task specs. The pinned cohort revision"
            "was downloaded back and every task hash re-validated."
        ),
        tags=("rlvr", "single-turn", "campaign-release", "converted"),
        count=57419,
        daytona_snapshot_count=0,
        oracle_validation=OracleValidation.EXECUTED,
        intended_use=IntendedUse.TRAIN,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # TaskTrove:tasktrove-math_prism-qa
    AtlasAdmission(
        origin=SourceOrigin.TASKTROVE,
        upstream_dataset="tasktrove-math_prism-qa",
        upstream_name="math_prism",
        release_repo="open-athena/campaign-release-tasktrove-math-prism-qa-9433b411-oracle-filtered-20261114-v2",
        release_revision="12332f2b420313ff5cd635d0bd2ecf688b7db6ae",
        build_path="Campaign Release",
        family="math-answer",
        upstream_url="https://huggingface.co/datasets/open-thoughts/TaskTrove/tree/02923004846e4e73862c20962f823a6d05100e7a",
        daytona_count_basis=(
            "Complete 2221-TaskSpec census from open-athena/campaign-release-tasktrove-math-prism-qa-9433b411"
            "-oracle-filtered-20261114-v2@12332f2b420313ff5cd635d0bd2ecf688b7db6ae; native Harbor task.toml"
            "or published runtime allocations; actor only on Daytona, grader roles separate; no provider"
            "calls; s3://marin-us-east-02a/marin/rl-data/atlas/oracle-filter-qa/2026.11.14.2-final-private/94"
            "33b411a517ac95adbd/population"
        ),
        notes=(
            "Qualification retained 2221/2221. Typed exclusions: {}. Original source and prior retention"
            "evidence remain archived in the release receipt. Symbolic exact compare with hidden gold"
            "references. Final v2 content QC reports 286 issues among 2,213 conclusive reviews (12.9236%,"
            "SOME_ISSUES); seven valid uncertain observations and one unreviewed task are excluded from that"
            "denominator. tasktrove-math_prism-2c73de38c5427048ffb943add4fee2759b8f63f8fb172c4a39f28ee94e97e0"
            "c0: QC unreviewed: review exceeded 16,384 tokens. All 2,221 tasks remain in the private release."
            "The 2,220 valid primary reviews are preserved byte for byte. Content coverage remains"
            "incomplete. The pinned cohort revision was downloaded back and every task hash re-validated."
        ),
        tags=("rlvr", "single-turn", "campaign-release", "converted"),
        count=2221,
        daytona_snapshot_count=0,
        oracle_validation=OracleValidation.EXECUTED,
        intended_use=IntendedUse.TRAIN,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # TaskTrove:tasktrove-math_stack-qa
    AtlasAdmission(
        origin=SourceOrigin.TASKTROVE,
        upstream_dataset="tasktrove-math_stack-qa",
        upstream_name="math_stack",
        release_repo="open-athena/campaign-release-tasktrove-math-stack-qa-dd13c545-oracle-filtered-20261114-v2",
        release_revision="5c6818d70670ff2da80ff73fe062bba8c0937be2",
        build_path="Campaign Release",
        family="math-answer",
        upstream_url="https://huggingface.co/datasets/open-thoughts/TaskTrove/tree/02923004846e4e73862c20962f823a6d05100e7a",
        daytona_count_basis=(
            "Complete 110600-TaskSpec census from open-athena/campaign-release-tasktrove-math-stack-qa-dd13c5"
            "45-oracle-filtered-20261114-v2@5c6818d70670ff2da80ff73fe062bba8c0937be2; native Harbor task.toml"
            "or published runtime allocations; actor only on Daytona, grader roles separate; no provider"
            "calls; s3://marin-us-east-02a/marin/rl-data/atlas/oracle-filter-qa/2026.11.14.2-final-private/dd"
            "13c545c945cf0e6664/population"
        ),
        notes=(
            "Qualification retained 110600/110600. Typed exclusions: {}. Original source and prior retention"
            "evidence remain archived in the release receipt. remediated: dedup/decon. Retained TaskSpec and"
            "runtime bytes match the original pin; original oversight and quality evidence apply through this"
            "subset lineage: s3://marin-us-east-02a/marin/rl-data/atlas/private-retention/2026.11.13.1/dc2bb4"
            "ac87716fdb64ed/result.json. Typed sympy comparison with per-task oracle. 125 rows whose typed"
            "references the production math-verify parser rejects are excluded at admission. Private release;"
            "conversation-only tasks import canonical task specs. The pinned cohort revision was downloaded"
            "back and every task hash re-validated."
        ),
        tags=("rlvr", "single-turn", "campaign-release", "converted"),
        count=110600,
        daytona_snapshot_count=0,
        oracle_validation=OracleValidation.EXECUTED,
        intended_use=IntendedUse.TRAIN,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # TaskTrove:tasktrove-puzzles-qa
    AtlasAdmission(
        origin=SourceOrigin.TASKTROVE,
        upstream_dataset="tasktrove-puzzles-qa",
        upstream_name="tasktrove-puzzles-qa",
        release_repo="open-athena/tasktrove-tasktrove-puzzles-qa-ab46b3ff-retained-20261116-v1",
        release_revision="4b557369b6c2db6b0dd00b56c13c8e43ea2fefdb",
        build_path="Campaign Release",
        family="math-answer",
        upstream_url="https://huggingface.co/datasets/open-thoughts/TaskTrove/tree/02923004846e4e73862c20962f823a6d05100e7a",
        daytona_count_basis=(
            "Complete 6921-TaskSpec census from open-athena/tasktrove-tasktrove-puzzles-qa-ab46b3ff-retained-"
            "20261116-v1@4b557369b6c2db6b0dd00b56c13c8e43ea2fefdb; native Harbor task.toml or published"
            "runtime allocations; actor only on Daytona, grader roles separate; no provider calls;"
            "s3://marin-us-east-02a/marin/rl-data/atlas/retained-census/2026.11.16.1-r9/ab46b3ff4a2ac6a14bd7/"
            "population"
        ),
        notes=(
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original"
            "oversight and quality evidence apply through this subset lineage:"
            "s3://marin-us-east-02a/marin/rl-data/atlas/private-retention/2026.11.16.1/ab46b3ff4a2ac6a14bd7/r"
            "esult.json. QA-format repair of TaskTrove:laion__all-puzzles-v2: non-agentic one-message"
            "conversation tasks graded in process by the source's pinned Verifyit exact/math grader against"
            "the reference mechanically extracted from each task's original solution/solve.sh oracle. Grader"
            "mode and instruction text unchanged from the canonical conversion. Task labels carry per-task"
            "lineage (original_task_id, original_solve_sh_sha256, oracle_extraction). All 6,926 rows"
            "converted with zero rejections; 5 exact duplicates collapse under the pipeline dedup key, 6,921"
            "retained. Full-population oracle filter through the production in-process grading path:"
            "6,921/6,921 admitted, zero typed exclusions, empty-submission controls 0.0 on every task."
            "Full-population fidelity against the canonical cohort: 6,921/6,921 content-equal. Lexical"
            "13-gram decontamination: 0 of 6,921 contaminated. SAMPLE content QC recorded 22 issues among 97"
            "conclusive reviews (3 unavailable; 22.68% issue rate, SOME_ISSUES); QC reports issues and never"
            "rejects tasks, and source-wide conclusions wait on FULL coverage."
        ),
        tags=("rlvr", "single-turn", "qa-format-repair", "campaign-release", "converted"),
        count=6921,
        daytona_snapshot_count=0,
        oracle_validation=OracleValidation.VERIFIER_EXECUTED,
        intended_use=IntendedUse.TRAIN,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # TaskTrove:tasktrove-safety-qa
    AtlasAdmission(
        origin=SourceOrigin.TASKTROVE,
        upstream_dataset="tasktrove-safety-qa",
        upstream_name="safety",
        release_repo="open-athena/campaign-release-tasktrove-safety-qa-5f6ab15c-retained-20261116-v1",
        release_revision="9da661b792767f41ed5b224b281035f62e4fba6a",
        build_path="Campaign Release",
        family="llm-judge-freeform",
        upstream_url="https://huggingface.co/datasets/open-thoughts/TaskTrove/tree/02923004846e4e73862c20962f823a6d05100e7a",
        daytona_count_basis=(
            "Complete 44939-TaskSpec census from open-athena/campaign-release-tasktrove-safety-qa-5f6ab15c-re"
            "tained-20261116-v1@9da661b792767f41ed5b224b281035f62e4fba6a; native Harbor task.toml or"
            "published runtime allocations; actor only on Daytona, grader roles separate; no provider calls;"
            "s3://marin-us-east-02a/marin/rl-data/atlas/retained-census/2026.11.16.1-r1/5f6ab15c19f34603c89a/"
            "population"
        ),
        notes=(
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original"
            "oversight and quality evidence apply through this subset lineage:"
            "s3://marin-us-east-02a/marin/rl-data/atlas/private-retention/2026.11.16.1/5f6ab15c19f34603c89a/r"
            "esult.json. Judge-only against a per-prompt two-line safety principle. No endpoint is hard-coded"
            "in any task spec; the MEDIUM JUDGE_TIERS definition pins model and completion settings. Private"
            "release; conversation-only tasks import canonical task specs. The pinned cohort revision was"
            "downloaded back and every task hash re-validated."
        ),
        tags=("rlvr", "single-turn", "campaign-release", "converted"),
        count=44939,
        daytona_snapshot_count=0,
        oracle_validation=OracleValidation.VERIFIER_EXECUTED,
        intended_use=IntendedUse.TRAIN,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # TaskTrove:tasktrove-science-openqa-qa
    AtlasAdmission(
        origin=SourceOrigin.TASKTROVE,
        upstream_dataset="tasktrove-science-openqa-qa",
        upstream_name="science-openqa",
        release_repo="open-athena/campaign-release-tasktrove-science-openqa-qa-3c4aa6a7-retained-20261116-v1",
        release_revision="0494a8d2c57f5ccd8d4ae7ef7307c44305df1209",
        build_path="Campaign Release",
        family="llm-judge-freeform",
        upstream_url="https://huggingface.co/datasets/open-thoughts/TaskTrove/tree/02923004846e4e73862c20962f823a6d05100e7a",
        daytona_count_basis=(
            "Complete 150644-TaskSpec census from open-athena/campaign-release-tasktrove-science-openqa-qa-3c"
            "4aa6a7-retained-20261116-v1@0494a8d2c57f5ccd8d4ae7ef7307c44305df1209; native Harbor task.toml or"
            "published runtime allocations; actor only on Daytona, grader roles separate; no provider calls;"
            "s3://marin-us-east-02a/marin/rl-data/atlas/retained-census/2026.11.16.1-r1/3c4aa6a78a63218fb919/"
            "population"
        ),
        notes=(
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original"
            "oversight and quality evidence apply through this subset lineage:"
            "s3://marin-us-east-02a/marin/rl-data/atlas/private-retention/2026.11.16.1/3c4aa6a78a63218fb919/r"
            "esult.json. Reference answer plus scientific-equivalence judge. The tier is chosen for the"
            "source (the archive demonstrably contains operator-algebra equivalence items a MEDIUM judge"
            "would mis-grade). No endpoint is hard-coded in any task spec; the HEAVY JUDGE_TIERS definition"
            "pins model and completion settings. Private release; conversation-only tasks import canonical"
            "task specs. The pinned cohort revision was downloaded back and every task hash re-validated."
        ),
        tags=("rlvr", "single-turn", "campaign-release", "converted"),
        count=150644,
        daytona_snapshot_count=0,
        oracle_validation=OracleValidation.VERIFIER_EXECUTED,
        intended_use=IntendedUse.TRAIN,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # TaskTrove:tasktrove-taco-built-image
    AtlasAdmission(
        origin=SourceOrigin.TASKTROVE,
        upstream_dataset="tasktrove-taco-built-image",
        upstream_name="tasktrove-taco-built-image",
        release_repo="open-athena/taco-built-084a8cc3-oracle-filter-20261114-v1",
        release_revision="8ffd075257ae0dc529ba702387ec64f668d9fe21",
        build_path="Campaign Release",
        family="competitive-programming",
        upstream_url="https://huggingface.co/datasets/open-thoughts/TaskTrove/tree/02923004846e4e73862c20962f823a6d05100e7a",
        daytona_count_basis=(
            "Complete 5065-TaskSpec census from open-athena/taco-built-084a8cc3-oracle-filter-20261114-v1@8ff"
            "d075257ae0dc529ba702387ec64f668d9fe21; native Harbor task.toml or published runtime allocations;"
            "actor only on Daytona, grader roles separate; no provider calls;"
            "s3://marin-us-east-02a/marin/rl-data/atlas/oracle-filter-code/2026.11.15.222-taco-built-final-pi"
            "n-population/084a8cc32013b7b78c04/population"
        ),
        notes=(
            "Qualification retained 5065/5116. Typed exclusions: {'oracle_failed': 51}. Original source and"
            "prior retention evidence remain archived in the release receipt. Excluded: the pinned payload"
            "reads only numeric rewards and can hide verifier failures. Republish with converter version 4"
            "and robust SAMPLE/FULL controls before admission. 5116 tasks retain the exact built output of"
            "their archived environment for both actor and grader. Registry manifests were read back and"
            "pulled image IDs matched the local builds. The private release includes setup fidelity, stdio"
            "parity, sampled liveness, retained exclusions and frozen build contexts. No oracle solution or"
            "model-quality assessment ran"
        ),
        tags=("agentic", "language:python", "campaign-release", "converted"),
        count=5065,
        daytona_snapshot_count=1,
        oracle_validation=OracleValidation.EXECUTED,
        intended_use=IntendedUse.TRAIN,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # TerminalWorld:TerminalWorld
    AtlasAdmission(
        origin=SourceOrigin.TERMINALWORLD,
        upstream_dataset="TerminalWorld",
        upstream_name="TerminalWorld full",
        release_repo="open-athena/terminalworld-terminalworld-a9818dbb-retained-20261124-b-final-tw-seven",
        release_revision="7a6cfbda8afd30a91a60c27e0a2877141b0c611d",
        build_path="Native Harbor files with root-delivery transport and current-pin oracle filtration",
        family="terminal",
        upstream_url="https://huggingface.co/datasets/terminalworld/TerminalWorld/tree/364549435c0fc130eaf0ccfc047a8a698c72643c",
        daytona_count_basis=(
            "Complete 2-TaskSpec census from open-athena/terminalworld-terminalworld-a9818dbb-retained-202611"
            "24-b-final-tw-seven@7a6cfbda8afd30a91a60c27e0a2877141b0c611d; native Harbor task.toml or"
            "published runtime allocations; actor only on Daytona, grader roles separate; no provider calls;"
            "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/litecoder-terminalworld/2026.11.22.1/termina"
            "lworld-seven-release-sample-v1/census/population"
        ),
        notes=(
            "Qualification retained 2/4. Typed exclusions: {'oracle_failed': 1,"
            "'resource_calibration:infrastructure': 1}. Original source and prior retention evidence remain"
            "archived in the release receipt. TRAIN per user ruling2026-11-22; upstream split=test is source"
            "naming. This four-task native execution scope preserves original task files, oracle, grader,"
            "allocations and RootDeliveryContract quiescence. Canonical filtration retains2; one oracle"
            "failure and one infrastructure hold remain uncounted, alongside the other incomplete DATA and"
            "LOCAL20GiB rows. Seed-zero SAMPLE reviews both retained tasks: BAD with one issue and zero"
            "missing. The git-nomad invocation-order/purge concern remains recorded verbatim; BAD alone never"
            "rejects under task-curation.md499-503/624. Ten nearest-source pairs are distinct. No source"
            "repair, budget increase or new oracle trial is claimed."
        ),
        count=2,
        daytona_snapshot_count=2,
        oracle_validation=OracleValidation.EXECUTED,
        intended_use=IntendedUse.TRAIN,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # marin-community:mcp-atlas-easy
    AtlasAdmission(
        origin=SourceOrigin.MARIN_COMMUNITY,
        upstream_dataset="mcp-atlas-easy",
        upstream_name="MCP-Atlas Easy",
        release_repo="open-athena/mcp-atlas-easy-eval-20261123-b-v1",
        release_revision="2ddca029a76af5b509105f9910066f822c711b98",
        build_path="Raw-completion JSON tool-call evaluation with host-recorded natural-stop status",
        family="tool-call",
        upstream_url="https://huggingface.co/datasets/marin-community/mcp-atlas-easy",
        daytona_count_basis=(
            "Complete 220-TaskSpec census from open-athena/mcp-atlas-easy-eval-20261123-b-v1@2ddca029a76af5b5"
            "09105f9910066f822c711b98; native Harbor task.toml or published runtime allocations; actor only"
            "on Daytona, grader roles separate; no provider calls; s3://marin-us-east-02a/marin/rl-data/atlas"
            "/ingest-b/mcp-atlas/2026.11.23.1/easy-eval-private-publication-census-v1/final-census/population"
        ),
        notes=(
            "EVAL-only completion population: all 220 tasks retained, including measured overlap. Feed the"
            "source prompt without a chat template; bind tokenizer/model EOS and host-recorded ReplyStatus,"
            "and grade one JSON tool call without live tools. The upstream card records inferred schemas for"
            "103 tasks. SAMPLE reviewed 100/100: GOOD, with six strict single-witness matching issues"
            "preserved. Nine packaged-grader/status fixtures on DATA establish checker execution, not"
            "original-oracle or candidate-model accuracy. Source resource budgets remain undeclared; grader"
            "qualification uses canonical deployment defaults. The 500-row live-service MCP-Atlas parent is a"
            "separate population."
        ),
        tags=("eval", "tool-use", "single-turn", "answer-graded"),
        count=220,
        daytona_snapshot_count=0,
        oracle_validation=OracleValidation.VERIFIER_EXECUTED,
        intended_use=IntendedUse.EVAL,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
    # pafitis:hardmath@rhs
    AtlasAdmission(
        origin=SourceOrigin.PAFITIS,
        upstream_dataset="hardmath",
        upstream_name="hardmath",
        release_repo="open-athena/pafitis-hardmath-rhs-83371b10-retained-2026.11.18.1",
        release_revision="00d51df256e06f1a9c791312a1e59dd18ea7a99f",
        build_path="MarinSkyRL",
        family="math-answer",
        variant="rhs",
        upstream_url=(
            "https://huggingface.co/datasets/pafitis/HARDMath_processed_training/tree/937e9f10356e31e854f6efb"
            "9a2507f1e200c8b25"
        ),
        daytona_count_basis=(
            "Complete 684-TaskSpec census from open-athena/pafitis-hardmath-rhs-83371b10-retained-2026.11.18."
            "1@00d51df256e06f1a9c791312a1e59dd18ea7a99f; native Harbor task.toml or published runtime"
            "allocations; actor only on Daytona, grader roles separate; no provider calls;"
            "s3://marin-us-east-02a/marin/rl-data/atlas/retained-census/2026.11.18.1-rhs/83371b101966e9a593c0"
            "/population"
        ),
        notes=(
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original"
            "oversight and quality evidence apply through this subset lineage:"
            "https://huggingface.co/datasets/open-athena/pafitis-hardmath-rhs-83371b10-retained-2026.11.18.1/"
            "raw/a07d71608486d91c032919b2c4be4c4d181950b9/release-manifest.json. "
        ),
        tags=("rlvr", "single-turn", "private", "remediated", "gym/aime"),
        count=684,
        daytona_snapshot_count=0,
        intended_use=IntendedUse.TRAIN,
        task_format=TaskFormat.TASKCOMPENDIUM,
    ),
)


def sources() -> list[RlDataSource]:
    """The Atlas-admitted records with their bound license receipts."""
    receipts = {
        source_id: LicenseReceipt.model_validate(row) for source_id, row in json.loads(RECEIPTS_PATH.read_text()).items()
    }
    records = []
    for record in RECORDS:
        generated = source_identifier(record.origin, record.upstream_dataset, record.subset, record.variant)
        receipt = receipts.get(generated)
        if receipt is None:
            raise ValueError(f"No stage-0 license receipt for the Atlas-admitted source: {generated}")
        records.append(record.source(receipt))
    return records
