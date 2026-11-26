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
from taskcompendium.pipeline.models import IntendedUse

import experiments.post_training.task_curation.datasets.atlas_retained_pins as atlas_retained_pins
from experiments.post_training.task_curation.retained_release import import_retained
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, TaskFormat


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

    def bound(self, pin: atlas_retained_pins.RetainedPin) -> RlDataSource:
        """Bind the record to its sealed retained pin and the retained reader.

        The record's admitted pin must be the pin the receipt sealed: two
        independent records of the same admission (the deferred record and the
        resolved receipt) have to agree before the declaration lands.
        """
        if (self.release_repo, self.release_revision) != (pin.release_repo, pin.release_revision):
            raise ValueError(f"The record's admitted pin differs from the sealed receipt: {self.upstream_dataset}")
        if self.count != pin.retained_count:
            raise ValueError(f"The record's count differs from the retained population: {self.upstream_dataset}")
        if self.intended_use is not pin.intended_use:
            raise ValueError(f"The record's declared use differs from the sealed lexical use: {self.upstream_dataset}")
        config = pin.cohort(name=pin.original_pipeline.name)
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
            count=pin.retained_count,
            upstream_url=self.upstream_url,
            dataset=config.dataset,
            current_release=pin.release,
            license_receipt=pin.license,
            lexical_contamination=pin.lexical,
            daytona_snapshot_count=pin.retained_census.snapshot_count,
            daytona_count_basis=pin.retained_census.count_basis,
            daytona_execution_failures=pin.retained_census.execution_failures,
            oracle_validation=self.oracle_validation,
            notes=(
                "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; "
                f"original oversight and quality evidence apply through this subset lineage: {pin.evidence_url}. "
                + self.notes
            ),
        )
        return RlDataSource(info=info, config=config, pipeline=import_retained)


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
            "ined-20261116-b-v1@1b4076f63af29e34aee43e16781784890b1e9835; native Harbor task.toml or "
            "published runtime allocations; actor only on Daytona, grader roles separate; no provider calls; "
            "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/retained-census/ifbench/2026.11.16.1/7bcdaba"
            "f57071c23bcdc/population"
        ),
        notes=(
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original "
            "oversight and quality evidence apply through this subset lineage: "
            "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/retained-publications/ifbench/2026.11.16.1/7"
            "bcdabaf57071c23bcdc/result.json. Original IFEvalG scorer 28c1816664491493242d453c2c7280ee56e6c20"
            "a; no released oracle; pending final-pin QA. Final-pin QA (2026-11-26, #12321 comment "
            "7189616682): the embedded grader mis-grades 29 of 53 probed instruction types - 4.5 percent of "
            "rows fail their plainest compliant reply, 49 percent pay a minimal violation; verifyit on main "
            "grades nearly all probes right. The declaration stands (the Atlas admitted with the weakness "
            "documented); grade these rows in process only after #11985-style check coverage and the two "
            "verifyit defects are fixed."
        ),
        tags=("license:odc-by",),
        count=95357,
        daytona_snapshot_count=0,
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
            "-retained-20261116-b-v1@7ac886010b3706345337eed86330b484111cfbf8; native Harbor task.toml or "
            "published runtime allocations; actor only on Daytona, grader roles separate; no provider calls; "
            "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/retained-census/cascade/2026.11.16.1/d941139"
            "928857a09768b/population"
        ),
        notes=(
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original "
            "oversight and quality evidence apply through this subset lineage: "
            "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/retained-publications/cascade/2026.11.16.1/d"
            "941139928857a09768b/result.json. Original NeMo terminal graders "
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
            "-rl-ee15b18a-retained-20261116-b-v1@52522171b690b25e50faa954619ab56362c0d55a; native Harbor "
            "task.toml or published runtime allocations; actor only on Daytona, grader roles separate; no "
            "provider calls; s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/retained-census/cascade/2026"
            ".11.16.1/ee15b18affec7822f712/population"
        ),
        notes=(
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original "
            "oversight and quality evidence apply through this subset lineage: "
            "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/retained-publications/cascade/2026.11.16.1/e"
            "e15b18affec7822f712/result.json. Original NeMo terminal graders "
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
            "ured-o-6abcd1f2-retained-20261124-b-final-nemo@be4cd53e7d59041e5a401b03a0fe4018ca401c29; native "
            "Harbor task.toml or published runtime allocations; actor only on Daytona, grader roles separate; "
            "no provider calls; s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/nemo-gym-remaining/2026.1"
            "1.24.1/structured-release-sample-v1/census/population"
        ),
        notes=(
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original "
            "oversight and quality evidence apply through this subset lineage: "
            "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/nemo-gym-remaining/2026.11.24.1/structured-r"
            "elease-sample-v1/retention-publication/6abcd1f2134e7ef00f0d/result.json. The native TRAIN9437 "
            "population converts9213 TaskSpecs and records224 native schema failures outside selection; "
            "validation512 is archived and not selected. Original prompts/private schema/source native "
            "verifier are unchanged. No per-task original answer exists; no generated oracle or empty "
            "liveness is required under the maintained no-oracle static-retention policy. Final-pin static "
            "retention preserves all9213. SAMPLE99reviewed/100attempted,12issues/1unavailable (within5% "
            "tolerance); every quality finding stays recorded. Six measured cross-source partial overlaps are "
            "below95% containment in both directions and stay directional advisory notes; no display fold or "
            "source task removal. Actor has no environment; separate native script grader declares its pinned "
            "build context; snapshot-census grader unresolved-build notes are retained, not relabelled as "
            "live execution support."
        ),
        count=9213,
        daytona_snapshot_count=0,
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
            "ained-20261121-bootstrap-v1@8d5a48bf4994f861c903cc90f34e1530c25ea052; native Harbor task.toml or "
            "published runtime allocations; actor only on Daytona, grader roles separate; no provider calls; "
            "s3://marin-us-east-02a/marin/rl-data/atlas/consumer/20261121/openresearcher-bootstrap-v2/retaine"
            "d-census/8d7d09e17e8f7692ed07/population"
        ),
        notes=(
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original "
            "oversight and quality evidence apply through this subset lineage: "
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
            "ained-20261116-b-v1@9079512fe3d2d332b446b1b37deb9d2898f5461e; native Harbor task.toml or "
            "published runtime allocations; actor only on Daytona, grader roles separate; no provider calls; "
            "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/polaris/2026.11.16.1/retained-census/2026.11"
            ".16.1/b25e0cb84e99b1f26085/population"
        ),
        notes=(
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original "
            "oversight and quality evidence apply through this subset lineage: "
            "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/polaris/2026.11.16.1/publication/2026.11.16."
            "1/b25e0cb84e99b1f26085/result.json. Original deterministic DeepScaleR reward "
            "eda12fe22dd83aca59ec46a67147240dc483f62b; LLM fallback disabled."
        ),
        tags=("rlvr", "single-turn", "license:apache-2.0"),
        count=53167,
        daytona_snapshot_count=0,
        intended_use=IntendedUse.TRAIN,
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
            "2b5961f4-retained-20261116-v1@9d3fbb17806c98a9d6c4299807ab586f3620cad9; native Harbor task.toml "
            "or published runtime allocations; actor only on Daytona, grader roles separate; no provider "
            "calls; s3://marin-us-east-02a/marin/rl-data/atlas/retained-census/2026.11.16.1-r1/2b5961f4fdc08f"
            "3d4218/population"
        ),
        notes=(
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original "
            "oversight and quality evidence apply through this subset lineage: "
            "s3://marin-us-east-02a/marin/rl-data/atlas/private-retention/2026.11.16.1/2b5961f4fdc08f3d4218/r"
            "esult.json. Exact gate then reference-based judge. No endpoint is hard-coded in any task spec; "
            "the MEDIUM JUDGE_TIERS definition pins model and completion settings. Private release; "
            "conversation-only tasks import canonical task specs. The pinned cohort revision was downloaded "
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
            "etained-2026.11.16.1-mathgym@2ec57f36b9ca674c30ec7a5fa514bebf242cc97d; native Harbor task.toml "
            "or published runtime allocations; actor only on Daytona, grader roles separate; no provider "
            "calls; s3://marin-us-east-02a/marin/rl-data/atlas/retained-census/2026.11.16.1-mathgym/03ef831ab"
            "6bc99825711/population"
        ),
        notes=(
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original "
            "oversight and quality evidence apply through this subset lineage: "
            "s3://marin-us-east-02a/marin/rl-data/atlas/private-retention/2026.11.16.1-mathgym/03ef831ab6bc99"
            "825711/result.json. Strict trailing boxed compare with no oracle. 155 rows whose typed "
            "references the production math-verify parser rejects are excluded at admission. Private release; "
            "conversation-only tasks import canonical task specs. The pinned cohort revision was downloaded "
            "back and every task hash re-validated. Content-quality finding 2026-11-18 (SAMPLE receipt "
            "fb053f6439…, pinned GLM-5.3 v3, seed-0 panel at the r23 pin): 91 of 3,941 current-population "
            "rows (2.3%) carry LaTeX-wrapped expected-answer strings (\\(\\displaystyle …\\)) in grader "
            "parameters.expected with math_type=scalar, which the scalar comparison path cannot parse; the "
            "SAMPLE review flags 1 of its 8 issues naming that symptom (tasktrove-math_gym-bb15c01fe058e37e…)"
            ". Grader and scorer stay unchanged (faithful environment). A private remediated version "
            "normalizing ONLY those expected-answer strings remains available as data repair if ruled "
            "material; it would need its own SAMPLE QA and admission, and this original stays as is."
        ),
        tags=("rlvr", "single-turn", "campaign-release", "converted"),
        count=3922,
        daytona_snapshot_count=0,
        oracle_validation=OracleValidation.VERIFIER_EXECUTED,
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
            "20261116-v1@4b557369b6c2db6b0dd00b56c13c8e43ea2fefdb; native Harbor task.toml or published "
            "runtime allocations; actor only on Daytona, grader roles separate; no provider calls; "
            "s3://marin-us-east-02a/marin/rl-data/atlas/retained-census/2026.11.16.1-r9/ab46b3ff4a2ac6a14bd7/"
            "population"
        ),
        notes=(
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original "
            "oversight and quality evidence apply through this subset lineage: "
            "s3://marin-us-east-02a/marin/rl-data/atlas/private-retention/2026.11.16.1/ab46b3ff4a2ac6a14bd7/r"
            "esult.json. QA-format repair of TaskTrove:laion__all-puzzles-v2: non-agentic one-message "
            "conversation tasks graded in process by the source's pinned Verifyit exact/math grader against "
            "the reference mechanically extracted from each task's original solution/solve.sh oracle. Grader "
            "mode and instruction text unchanged from the canonical conversion. Task labels carry per-task "
            "lineage (original_task_id, original_solve_sh_sha256, oracle_extraction). All 6,926 rows "
            "converted with zero rejections; 5 exact duplicates collapse under the pipeline dedup key, 6,921 "
            "retained. Full-population oracle filter through the production in-process grading path: "
            "6,921/6,921 admitted, zero typed exclusions, empty-submission controls 0.0 on every task. "
            "Full-population fidelity against the canonical cohort: 6,921/6,921 content-equal. Lexical "
            "13-gram decontamination: 0 of 6,921 contaminated. SAMPLE content QC recorded 22 issues among 97 "
            "conclusive reviews (3 unavailable; 22.68% issue rate, SOME_ISSUES); QC reports issues and never "
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
            "tained-20261116-v1@9da661b792767f41ed5b224b281035f62e4fba6a; native Harbor task.toml or "
            "published runtime allocations; actor only on Daytona, grader roles separate; no provider calls; "
            "s3://marin-us-east-02a/marin/rl-data/atlas/retained-census/2026.11.16.1-r1/5f6ab15c19f34603c89a/"
            "population"
        ),
        notes=(
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original "
            "oversight and quality evidence apply through this subset lineage: "
            "s3://marin-us-east-02a/marin/rl-data/atlas/private-retention/2026.11.16.1/5f6ab15c19f34603c89a/r"
            "esult.json. Judge-only against a per-prompt two-line safety principle. No endpoint is hard-coded "
            "in any task spec; the MEDIUM JUDGE_TIERS definition pins model and completion settings. Private "
            "release; conversation-only tasks import canonical task specs. The pinned cohort revision was "
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
            "4aa6a7-retained-20261116-v1@0494a8d2c57f5ccd8d4ae7ef7307c44305df1209; native Harbor task.toml or "
            "published runtime allocations; actor only on Daytona, grader roles separate; no provider calls; "
            "s3://marin-us-east-02a/marin/rl-data/atlas/retained-census/2026.11.16.1-r1/3c4aa6a78a63218fb919/"
            "population"
        ),
        notes=(
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original "
            "oversight and quality evidence apply through this subset lineage: "
            "s3://marin-us-east-02a/marin/rl-data/atlas/private-retention/2026.11.16.1/3c4aa6a78a63218fb919/r"
            "esult.json. Reference answer plus scientific-equivalence judge. The tier is chosen for the "
            "source (the archive demonstrably contains operator-algebra equivalence items a MEDIUM judge "
            "would mis-grade). No endpoint is hard-coded in any task spec; the HEAVY JUDGE_TIERS definition "
            "pins model and completion settings. Private release; conversation-only tasks import canonical "
            "task specs. The pinned cohort revision was downloaded back and every task hash re-validated."
        ),
        tags=("rlvr", "single-turn", "campaign-release", "converted"),
        count=150644,
        daytona_snapshot_count=0,
        oracle_validation=OracleValidation.VERIFIER_EXECUTED,
        intended_use=IntendedUse.TRAIN,
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
            "1@00d51df256e06f1a9c791312a1e59dd18ea7a99f; native Harbor task.toml or published runtime "
            "allocations; actor only on Daytona, grader roles separate; no provider calls; "
            "s3://marin-us-east-02a/marin/rl-data/atlas/retained-census/2026.11.18.1-rhs/83371b101966e9a593c0"
            "/population"
        ),
        notes=(
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original "
            "oversight and quality evidence apply through this subset lineage: "
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


@cache
def _records_with_pins() -> list[RlDataSource] | None:
    """Every record bound to its sealed pin, or None when the table is unfetchable."""
    pins = atlas_retained_pins.atlas_retained_pins()
    if pins is None:
        return None
    bound = []
    for record in RECORDS:
        generated = source_identifier(record.origin, record.upstream_dataset, record.subset, record.variant)
        pin = pins.get(generated)
        if pin is None:
            raise ValueError(f"No sealed retained pin for the Atlas-admitted record: {generated}")
        bound.append(record.bound(pin))
    return bound


def sources() -> list[RlDataSource]:
    """The Atlas-admitted records declared at their sealed retained pins."""
    bound = _records_with_pins()
    return list(bound) if bound is not None else []


if __name__ == "__main__":
    raise SystemExit("regeneration runs through the retained-pin refresh; see atlas_retained_pins")
