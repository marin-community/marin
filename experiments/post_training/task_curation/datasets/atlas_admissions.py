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
        notes=(
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original "
            "oversight and quality evidence apply through this subset lineage: "
            "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/retained-publications/cascade/2026.11.16.1/d"
            "941139928857a09768b/result.json. Original NeMo terminal graders "
            "39d2bde61e426f93e4bc00c0255fc1d839f859e6; unsupported components retain typed rejections."
        ),
        tags=("license:odc-by",),
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
        notes=(
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original "
            "oversight and quality evidence apply through this subset lineage: "
            "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/retained-publications/cascade/2026.11.16.1/e"
            "e15b18affec7822f712/result.json. Original NeMo terminal graders "
            "39d2bde61e426f93e4bc00c0255fc1d839f859e6; unsupported components retain typed rejections."
        ),
        tags=("license:odc-by",),
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
        notes=(
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original "
            "oversight and quality evidence apply through this subset lineage: "
            "s3://marin-us-east-02a/marin/rl-data/atlas/consumer/20261121/openresearcher-bootstrap-v2/publica"
            "tions/8d7d09e17e8f7692ed07/result.json. "
        ),
        tags=("license:mit", "corpus-license:odc-by"),
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
        notes=(
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original "
            "oversight and quality evidence apply through this subset lineage: "
            "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/polaris/2026.11.16.1/publication/2026.11.16."
            "1/b25e0cb84e99b1f26085/result.json. Original deterministic DeepScaleR reward "
            "eda12fe22dd83aca59ec46a67147240dc483f62b; LLM fallback disabled."
        ),
        tags=("rlvr", "single-turn", "license:apache-2.0"),
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
        notes=(
            "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; original "
            "oversight and quality evidence apply through this subset lineage: "
            "https://huggingface.co/datasets/open-athena/pafitis-hardmath-rhs-83371b10-retained-2026.11.18.1/"
            "raw/a07d71608486d91c032919b2c4be4c4d181950b9/release-manifest.json. "
        ),
        tags=("rlvr", "single-turn", "private", "remediated", "gym/aime"),
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
