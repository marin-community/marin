# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Published campaign cohorts and runnable rebuilt releases.

Each entry pins an immutable Hugging Face revision in the RL Data Atlas.
Rebuilt TaskCompendium cohorts use ``prebuilt.import_prebuilt`` to validate and
admit their published tasks without repeating conversion. Their qualification
evidence and oversight levels describe the checks behind admission.
Native Harbor cohorts use ``prebuilt.import_harbor_prebuilt`` and require
exact-package control receipts for admission.

Retired legacy pointers are tracked in retired-inventory-only.json. Their
underlying artifacts and prior evidence remain archived.
"""

from dataclasses import replace

from finestore.rl.curation import (
    IndependentQC,
    OracleValidation,
    RecipeQAHistoricalNotes,
    Verdict,
)
from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.prebuilt import (
    PrebuiltCohort,
    import_harbor_prebuilt,
    import_prebuilt,
    import_taskcompendium_prebuilt,
)
from experiments.post_training.task_curation.source import (
    RlDataSource,
    SourceInfo,
    SourceOrigin,
    TaskFormat,
)

TASKTROVE_UPSTREAM_URL = (
    "https://huggingface.co/datasets/open-thoughts/TaskTrove/tree/" "02923004846e4e73862c20962f823a6d05100e7a"
)

LEGACY_PYTHON_TESTS_RECIPE_REVISION = "datasets/tasktrove/python_tests.py @ e6be97168d"


def _rebuilt_cohort(config: PrebuiltCohort, info: SourceInfo) -> RlDataSource[PrebuiltCohort]:
    return RlDataSource(
        pipeline=import_prebuilt,
        config=config,
        info=replace(info, count=config.expected_rows, dataset=config.dataset),
    )


CAMPAIGN_SUFFIX = "taskcompendium-20261112"


def _qa_cohort(config: PrebuiltCohort, info: SourceInfo) -> RlDataSource[PrebuiltCohort]:
    """A conversation-only QA cohort admitted from its published canonical task specs."""
    return RlDataSource(
        pipeline=import_taskcompendium_prebuilt,
        config=config,
        info=replace(info, count=config.expected_rows, dataset=config.dataset),
    )


def _tasktrove_qa_release(
    name: str,
    family: str,
    revision: str,
    expected_rows: int,
    evidence: str,
    recipe_revision: str,
    notes: str,
    tags: tuple[str, ...] = ("rlvr", "single-turn", "campaign-release", "converted"),
    reviewed: int = 24,
) -> RlDataSource[PrebuiltCohort]:
    """A privately published, QA-qualified TaskTrove cohort pending release.

    ``reviewed`` is the QA sample denominator (the 24-task grader-control panel for
    math sources, the 10-task judge e2e panels elsewhere), not the converted row
    count: the full-population conversion census is recorded in the evidence text.
    """
    config = PrebuiltCohort(
        intended_use=IntendedUse.TRAIN,
        name=f"{name}-qa",
        version="2026.11.12.1",
        repo=f"open-athena/tasktrove-{name}-{CAMPAIGN_SUFFIX}",
        revision=revision,
        expected_rows=expected_rows,
    )
    return _qa_cohort(
        config,
        SourceInfo(
            data_restriction=DataRestriction.OPEN,
            id=f"TaskTrove:tasktrove-{name}-qa",
            upstream_dataset=f"tasktrove-{name}-qa",
            upstream_name=name,
            build_path="Campaign Release",
            origin=SourceOrigin.TASKTROVE,
            upstream_url=TASKTROVE_UPSTREAM_URL,
            family=family,
            tags=tags,
            task_format=TaskFormat.TASKSPEC,
            oracle_validation=OracleValidation.VERIFIER_EXECUTED,
            recipe_qa_history=RecipeQAHistoricalNotes(
                campaign="RL Data Atlas",
                recipe_revision=recipe_revision,
                verification=evidence,
                reviewed=reviewed,
                admitted=expected_rows,
            ),
            notes=notes,
        ),
    )


PUZZLES_QA_REPAIR_REVISION = "44528e65eee00f092ab971a0099f61e6ae4e27ce"
PUZZLES_QA_PIN = PrebuiltCohort(
    intended_use=IntendedUse.TRAIN,
    name="tasktrove-puzzles-qa",
    version="2026.11.14.1",
    repo="open-athena/tasktrove-puzzles-taskcompendium-20261114",
    revision="adfcbd4a8ddefc9e69febc91931c08ba03372bad",
    expected_rows=6921,
)
PUZZLES_QA_NOTES = (
    "QA-format repair of TaskTrove:laion__all-puzzles-v2: non-agentic one-message conversation tasks "
    "graded in process by the source's pinned Verifyit exact/math grader against the reference "
    "mechanically extracted from each task's original solution/solve.sh oracle. Grader mode and "
    "instruction text unchanged from the canonical conversion. Task labels carry per-task lineage "
    "(original_task_id, original_solve_sh_sha256, oracle_extraction). All 6,926 rows converted with "
    "zero rejections; 5 exact duplicates collapse under the pipeline dedup key, 6,921 retained. "
    "Full-population oracle filter through the production in-process grading path: 6,921/6,921 "
    "admitted, zero typed exclusions, empty-submission controls 0.0 on every task. Full-population "
    "fidelity against the canonical cohort: 6,921/6,921 content-equal. Lexical 13-gram "
    "decontamination: 0 of 6,921 contaminated. SAMPLE content QC recorded 22 issues among 97 "
    "conclusive reviews (3 unavailable; 22.68% issue rate, SOME_ISSUES); QC reports issues and never "
    "rejects tasks, and source-wide conclusions wait on FULL coverage."
)
PUZZLES_QA_RECIPE_QA = RecipeQAHistoricalNotes(
    campaign="RL Data Atlas",
    recipe_revision=f"datasets/tasktrove/puzzles_qa.py v1; {PUZZLES_QA_REPAIR_REVISION[:12]}",
    verification=(
        "All 6,926 rows converted through the production recipe with zero rejections; 5 exact "
        "duplicates collapse under the pipeline dedup key, 6,921 retained. Every reference is "
        "mechanically extracted from the task's original solution/solve.sh oracle (single-printf "
        "literal template) and cross-checked member-for-member against the declared tests/gold.json "
        "key. Full-population oracle filter through the production in-process grading path: "
        "6,921/6,921 admitted, zero typed exclusions, empty-submission controls 0.0 on every task. "
        "Full-population fidelity against the canonical cohort: 6,921/6,921 content-equal. Lexical "
        "13-gram decontamination: 0 of 6,921 contaminated. SAMPLE content QC on the seed-0 100-task "
        "panel through the pinned V3 reviewer (GLM-5.3, temperature 0.7): 97/100 reviewed, 3 "
        "unavailable, 22 issues (22.68% issue rate, SOME_ISSUES) — source-content defects recorded "
        "for FULL review; panel coverage is incomplete, so source-wide conclusions wait on FULL."
    ),
    reviewed=100,
    admitted=6921,
)


def _puzzles_qa_release() -> RlDataSource[PrebuiltCohort]:
    """The puzzles QA-format repair cohort over its published private pin."""
    return _qa_cohort(
        PUZZLES_QA_PIN,
        SourceInfo(
            data_restriction=DataRestriction.OPEN,
            id="TaskTrove:tasktrove-puzzles-qa",
            upstream_dataset="tasktrove-puzzles-qa",
            upstream_name="tasktrove-puzzles-qa",
            subset=None,
            build_path="Campaign Release",
            origin=SourceOrigin.TASKTROVE,
            upstream_url=TASKTROVE_UPSTREAM_URL,
            family="math-answer",
            tags=("rlvr", "single-turn", "qa-format-repair", "campaign-release", "converted"),
            task_format=TaskFormat.TASKSPEC,
            oracle_validation=OracleValidation.VERIFIER_EXECUTED,
            recipe_qa_history=PUZZLES_QA_RECIPE_QA,
            notes=PUZZLES_QA_NOTES,
        ),
    )


def sources() -> list[RlDataSource]:
    # The taskspec re-pin (declared-pin ruling 2026-11-25 ~23:35 EST): the cohort's
    # conversion release moves to the taskspec conversion of the same upstream content
    # (per-task byte and grader identity verified against the prior pin's conversion
    # output; the conversion identity harbor-cohort-taskspec-v1). The prior release's
    # qualification evidence stays at its revisions and in this file's history.
    swesmith_repaired = PrebuiltCohort(
        intended_use=IntendedUse.TRAIN,
        name="swesmith-verifier-repaired-v5",
        version="harbor-cohort-taskspec-v1",
        repo="open-athena/swesmith-verifier-repaired-v5-0bfcd894-taskspec-20261124g",
        revision="aab66e36193afed777ceea90bf04e2001f2cf11e",
        expected_rows=7016,
        repo_type="dataset",
    )
    return [
        _rebuilt_cohort(
            config=PrebuiltCohort(
                intended_use=IntendedUse.TRAIN,
                name="forge-main-20261117",
                version="2026.11.21.2",
                repo="open-athena/forge-main-taskcompendium-20261117",
                revision="2f1155cb0278fbb3dd4ee5b0685ae68800d2a143",
                expected_rows=4000,
            ),
            info=SourceInfo(
                origin=SourceOrigin.MARIN,
                upstream_dataset="forge-main-20261117",
                upstream_name="Forge main split, 2026-11-17",
                build_path="Forge / Campaign Release",
                family="synthetic-cross-domain",
                tags=("rlvr", "agentic", "synthetic", "campaign-release"),
                task_format=TaskFormat.TASKSPEC,
                oracle_validation=OracleValidation.SKIPPED,
                recipe_qa_history=RecipeQAHistoricalNotes(
                    campaign="Forge re-validation campaign 2026-11-13",
                    recipe_revision="forge_release.py; forge @ be691e29b0",
                    verification=(
                        "Unverified historical evidence; does not certify execution in the declared environments. "
                        "From the exported records of all 4,000 tasks, in forge's own sandboxes: 95,706 authored "
                        "controls (21,996 correct answers, at least one per task) all graded as expected at "
                        "confirmation; both adversaries finished on the accepted version without a reproduced "
                        "grader flaw; the final review of eight GLM-5.3 trials finished with no defect. Machines "
                        "naming the digest-pinned python:3.12-slim (670 task machines, 3,293 graders) ran under "
                        "forge's local bubblewrap sandbox, not in that image"
                    ),
                    reviewed=4000,
                    admitted=0,
                ),
                notes=(
                    "4,000 synthetic tasks written by forge from a 1,999-skill, 45-subject catalog (1,598 skills "
                    "covered), selected coverage first from 4,378 accepts after dropping 244 that carried a "
                    "reviewer note or missing evidence; the ranking favoured GLM-5.3 solve rates near one half. "
                    "955 in band, 3,021 too easy and 24 too hard for GLM-5.3 over eight trials; 2,978 "
                    "answer-only, 670 Linux and 352 simulated-shell tasks. Machines that forge's local sandbox "
                    "recorded without an image are restored to the image their authors declared. The 2026.11.21.1 "
                    "conversion re-roots the 6,855 workspace resources of 850 tasks at their declared working "
                    "directories, so the runtime's root install lands each file where its author declared it, "
                    "and the 2026.11.21.2 conversion preserves each task's authored golden control as private "
                    "oracle resources on 1,024 tasks (the card's revision history carries both). Not "
                    "re-executed through the Atlas rollout path. The repository is private pending release."
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                intended_use=IntendedUse.TRAIN,
                name="code-contests-built-image",
                version="2026.11.12.1",
                repo="open-athena/tasktrove-code-contests-taskcompendium-20261112",
                revision="5b43c6f4c2322a5808e4dbdfaf4f0052037d3607",
                expected_rows=8222,
            ),
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                id="TaskTrove:tasktrove-code-contests-built-image",
                upstream_dataset="tasktrove-code-contests-built-image",
                upstream_name="tasktrove-code-contests-built-image",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="competitive-programming",
                tags=("agentic", "single-turn", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKSPEC,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "production daytona_identity, Harbor machine_spec and Shellbox snapshot_identity over "
                    "all 8222 specs: one identity across actor and grader, at 2 CPU / 4096 MB / 10240 MB. "
                    "Zero identity failures and execution-requirement refusals; within the ten-identity budget"
                ),
                daytona_execution_failures=0,
                daytona_run_evidence=(
                    "https://huggingface.co/datasets/open-athena/tasktrove-code-contests-taskcompendium-20261112"
                    "/blob/5b43c6f4c2322a5808e4dbdfaf4f0052037d3607/liveness.jsonl"
                ),
                oracle_validation=OracleValidation.SKIPPED,
                recipe_qa_history=RecipeQAHistoricalNotes(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/code.py version 3; frozen VerifyIT @ d8cdda47f4",
                    verification=(
                        "Withdrawn qualification: numeric-only rewards did not preserve unscored verifier errors. "
                        "all 8222 rows match their native archive's built context, including file bytes "
                        "and modes; all rendered stdio configurations equal their original typed graders. "
                        "The declared script invokes the image-installed VerifyIT CLI, preserving the "
                        "candidate interpreter. The seed-0 Daytona RL sample graded 20 empty submissions: "
                        "20 negative verdicts, reward 0.0, no infrastructure errors or timeouts. "
                        "Evidence accompanies tasks.parquet at the pinned private revision"
                    ),
                    reviewed=8222,
                    admitted=8222,
                ),
                notes=(
                    "Excluded: the pinned payload reads only numeric rewards and can hide verifier failures. "
                    "Republish with converter version 4 and robust SAMPLE/FULL controls before admission. "
                    "8222 tasks retain the exact built output of their archived environment for both "
                    "actor and grader. Registry manifests were read back and pulled image IDs matched "
                    "the local builds. The private release includes setup fidelity, stdio parity, "
                    "sampled liveness, retained exclusions and frozen build contexts. "
                    "No oracle solution or model-quality assessment ran"
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                intended_use=IntendedUse.TRAIN,
                name="taco-built-image",
                version="2026.11.12.1",
                repo="open-athena/tasktrove-taco-taskcompendium-20261112",
                revision="c96a95bc94e4824fae425811f246501f64abfadd",
                expected_rows=5116,
            ),
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                id="TaskTrove:tasktrove-taco-built-image",
                upstream_dataset="tasktrove-taco-built-image",
                upstream_name="tasktrove-taco-built-image",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="competitive-programming",
                tags=("agentic", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKSPEC,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "production daytona_identity, Harbor machine_spec and Shellbox snapshot_identity over "
                    "all 5116 specs: one identity across actor and grader, at 2 CPU / 4096 MB / 10240 MB. "
                    "Zero identity failures and execution-requirement refusals; within the ten-identity budget"
                ),
                daytona_execution_failures=0,
                daytona_run_evidence=(
                    "https://huggingface.co/datasets/open-athena/tasktrove-taco-taskcompendium-20261112"
                    "/blob/c96a95bc94e4824fae425811f246501f64abfadd/liveness.jsonl"
                ),
                oracle_validation=OracleValidation.SKIPPED,
                recipe_qa_history=RecipeQAHistoricalNotes(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/code.py version 3; frozen VerifyIT @ d8cdda47f4",
                    verification=(
                        "Withdrawn qualification: numeric-only rewards did not preserve unscored verifier errors. "
                        "all 5116 rows match their native archive's built context, including file bytes "
                        "and modes; all rendered stdio configurations equal their original typed graders. "
                        "The declared script invokes the image-installed VerifyIT CLI, preserving the "
                        "candidate interpreter. The seed-0 Daytona RL sample graded 20 empty submissions: "
                        "20 negative verdicts, reward 0.0, no infrastructure errors or timeouts. "
                        "Evidence accompanies tasks.parquet at the pinned private revision"
                    ),
                    reviewed=5116,
                    admitted=5116,
                ),
                notes=(
                    "Excluded: the pinned payload reads only numeric rewards and can hide verifier failures. "
                    "Republish with converter version 4 and robust SAMPLE/FULL controls before admission. "
                    "5116 tasks retain the exact built output of their archived environment for both "
                    "actor and grader. Registry manifests were read back and pulled image IDs matched "
                    "the local builds. The private release includes setup fidelity, stdio parity, "
                    "sampled liveness, retained exclusions and frozen build contexts. "
                    "No oracle solution or model-quality assessment ran"
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                intended_use=IntendedUse.TRAIN,
                name="calibforge-v1-rebuilt",
                version="2026.11.12.1",
                repo="open-athena/calibforge-v1-rebuilt-taskcompendium-20261111",
                revision="8f00c12884d6a68107b245e7c98711b6bd64f4af",
                expected_rows=1357,
            ),
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                id="TaskTrove:smartie-calibforge-rebuilt-v1",
                upstream_dataset="smartie-calibforge-rebuilt-v1",
                upstream_name="smartie-calibforge-rebuilt-v1",
                subset=None,
                build_path="Snowball Smartie RL / Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="terminal-agent",
                tags=("agentic", "license:cc-by-4.0", "campaign-release", "converted"),
                task_format=TaskFormat.TASKSPEC,
                daytona_snapshot_count=7,
                daytona_count_basis=(
                    "production snapshot_identity with Harbor's machine_spec over the lowered task specs "
                    "at the pinned revision: 1,357 specs, seven distinct identities, zero failures"
                ),
                daytona_execution_failures=0,
                oracle_validation=OracleValidation.SKIPPED,
                recipe_qa_history=RecipeQAHistoricalNotes(
                    campaign="Snowball Smartie RL",
                    recipe_revision="calibforge_rebuild.py @ 817f7dbd00",
                    verification=(
                        "Unverified historical reward-only sample; does not certify test execution. "
                        "setup fidelity executed on all 1,357 tasks (1,357 pass) and verifier liveness on "
                        "76 sampled tasks drawn across all seven bases (76 negative verdicts, reward 0.0, "
                        "no infrastructure errors)"
                    ),
                    reviewed=1357,
                    admitted=1357,
                ),
                notes=(
                    "Excluded pending file-delivery repair and qualification with verifier execution receipts. "
                    "1,357 CalibForge V1 tasks rebuilt onto seven digest-pinned base images, with fixtures "
                    "staged through task resources and environment_requirements.setup_commands. Upstream: "
                    "smartie-continuation-20261016/calibforge @ 6a6429ccba098e11ea9f6297a70cb43a34670c0f, "
                    "minus the curated decontamination list. 732 tasks deferred because their recipe needs a "
                    "step the lowering cannot express; one excluded because its own setup segfaults. "
                    "Withdrawn reward-only claim: 1,357 of 1,357 reproduce the native copies at their "
                    "destinations, and 76 of 76 sampled verifiers return a negative verdict on the "
                    "untouched workspace. The repository is private pending release."
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                intended_use=IntendedUse.TRAIN,
                name="nl2bash-v1-rebuilt",
                version="2026.11.12.1",
                repo="open-athena/nl2bash-v1-rebuilt-taskcompendium-20261111",
                revision="1be6be13a89ab3ef30c7390ca2f29d7b9eeb40dd",
                expected_rows=1498,
            ),
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                id="TaskTrove:tasktrove-nl2bash-rebuilt-v1",
                upstream_dataset="tasktrove-nl2bash-rebuilt-v1",
                upstream_name="tasktrove-nl2bash-rebuilt-v1",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="terminal-agent",
                tags=("agentic", "language:bash", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKSPEC,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "Production snapshot_identity over every pinned published TaskSpec: 1 distinct agent identities. "
                    "Offline Daytona factory validation refuses all 1,498 declared package-lock graders. The declared "
                    "grader environments require local execution; this cohort is Daytona-unsafe."
                ),
                daytona_execution_failures=1498,
                oracle_validation=OracleValidation.SKIPPED,
                recipe_qa_history=RecipeQAHistoricalNotes(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/nl2bash.py @ e6be97168d",
                    verification=(
                        "Unverified historical reward-only sample; does not certify test execution. "
                        "setup fidelity executed on all 1,498 converted specs at this revision (all reproduce "
                        "their archive's lowered base, working directory, steps and environment variables) and "
                        "verifier liveness on 20 sampled tasks drawn across the cohort's single environment "
                        "(20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=1498,
                    admitted=1498,
                ),
                notes=(
                    "1,498 nl2bash tasks converted to run in the environment their archives specify: the "
                    "ubuntu:24.04 recipe each task ships is lowered onto the digest-pinned base and its RUN "
                    "steps replay as setup commands, one distinct environment across the cohort. Measured "
                    "at the pinned upstream revision "
                    "DCAgent2__nl2bash-tasks-cleaned-oracle-v2 @ 02923004846e4e73862c20962f823a6d05100e7a: "
                    "zero refusals, zero substitutions. Withdrawn reward-only claim: setup fidelity matches on "
                    "all 1,498 rows and 20 of 20 sampled verifiers return a negative verdict on an empty "
                    "submission at this revision. The repository is private pending release."
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                intended_use=IntendedUse.TRAIN,
                name="codeforces-v1-rebuilt",
                version="2026.11.16.lock-repair.1",
                repo="open-athena/tasktrove-codeforces-lock-repair-20261116-b-v2",
                revision="c11e504b01247cb68dd1e1aebc586ac48c72829e",
                expected_rows=9690,
            ),
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                id="TaskTrove:tasktrove-codeforces-rebuilt-v1",
                upstream_dataset="tasktrove-codeforces-rebuilt-v1",
                upstream_name="tasktrove-codeforces-rebuilt-v1",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="competitive-programming",
                tags=("agentic", "language:python", "campaign-release", "converted", "single-turn"),
                task_format=TaskFormat.TASKSPEC,
                oracle_validation=OracleValidation.SKIPPED,
                recipe_qa_history=RecipeQAHistoricalNotes(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/code.py @ e6be97168d",
                    verification=(
                        "Unverified historical reward-only sample; does not certify test execution. "
                        "setup fidelity executed on all 9,745 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks drawn across the cohort's "
                        "single environment (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=9745,
                    admitted=9745,
                ),
                notes=(
                    "Repair of the prior 9,690-task Codeforces release changes only its grader's package-lock "
                    "reference. The requirements lock was reconstructed from the pinned environment recipe; the "
                    "historical lock bytes were unavailable. The original grader implementation, task payloads and "
                    "stored runtime configurations remain unchanged. Lexical decontamination excluded 17 tasks, leaving "
                    "9,673. Complete actor, grading and derived-contract censuses cover all retained tasks. Static "
                    "checks for missing artifacts and unsupported contract features excluded none. The pinned "
                    "policy for sources without original oracle solutions suspends per-task executability proof; "
                    "execution remains unmeasured. Environment provenance: "
                    "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/codeforces-readmission/"
                    "2026.11.16.5/lock-repair-v2/publication.json."
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                intended_use=IntendedUse.TRAIN,
                name="competitive-coding-v1-rebuilt",
                version="2026.11.13.1",
                repo="open-athena/campaign-release-tasktrove-competitive-coding-rebuilt-v-6ce7836f-retained-20261113-v1",
                revision="93e1cc49e06e2efba6fd5cf1cb611596644f7f9d",
                expected_rows=13969,
            ),
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                id="TaskTrove:tasktrove-competitive-coding-rebuilt-v1",
                upstream_dataset="tasktrove-competitive-coding-rebuilt-v1",
                upstream_name="tasktrove-competitive-coding-rebuilt-v1",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="competitive-programming",
                tags=("agentic", "single-turn", "language:python", "campaign-release", "converted"),
                task_format=TaskFormat.TASKSPEC,
                oracle_validation=OracleValidation.SKIPPED,
                recipe_qa_history=RecipeQAHistoricalNotes(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/code.py @ e6be97168d",
                    verification=(
                        "Unverified historical reward-only sample; does not certify test execution. "
                        "setup fidelity executed on all 13,973 converted specs at the original conversion revision "
                        "(all reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks drawn across the cohort's "
                        "single environment (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=13973,
                    admitted=13973,
                ),
                notes=(
                    "13,969 competitive-coding tasks: the 2026.11.13.1 retained release published from the "
                    "tasktrove-competitive-coding-v1-taskcompendium-20261112 conversion (13,973 converted; the "
                    "release's exact-duplicate selection kept 13,969), measured at the pinned upstream revision "
                    "laion__nemotron-gym-competitive-coding-v2 @ 02923004846e4e73862c20962f823a6d05100e7a "
                    "with zero substitutions. The 2026.11.17.1 census repair re-measured the release pin with "
                    "every TaskSpec byte identical to the original conversion (content-identity receipts), and "
                    "the 2026.11.18.1-r1 retained re-publication selects this pin after one lexical-flag drop. "
                    "1,739 rows were refused for gold leaked into the instruction and one as a reviewed defect; "
                    "refused rows are the conversion's own reviewed outcome classes. Withdrawn reward-only "
                    "claim: setup fidelity matches on every converted row and 20 of 20 sampled verifiers return "
                    "a negative verdict on an empty submission. The repository is private pending release."
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                intended_use=IntendedUse.TRAIN,
                name="curriculum-easy-v1-rebuilt",
                version="2026.11.12.1",
                repo="open-athena/tasktrove-curriculum-easy-v1-taskcompendium-20261112",
                revision="ca799448d14f9b72ac2846884f5f4c03025f8ea0",
                expected_rows=505,
            ),
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                id="TaskTrove:tasktrove-curriculum-easy-rebuilt-v1",
                upstream_dataset="tasktrove-curriculum-easy-rebuilt-v1",
                upstream_name="tasktrove-curriculum-easy-rebuilt-v1",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="python-tests",
                tags=("agentic", "language:python", "campaign-release", "converted"),
                task_format=TaskFormat.TASKSPEC,
                daytona_snapshot_count=2,
                daytona_count_basis=(
                    "Production snapshot_identity over every pinned published TaskSpec: 2 distinct agent identities. "
                    "Offline Daytona factory validation refuses all 505 declared package-lock graders. The declared "
                    "grader environments require local execution; this cohort is Daytona-unsafe."
                ),
                daytona_execution_failures=505,
                oracle_validation=OracleValidation.SKIPPED,
                recipe_qa_history=RecipeQAHistoricalNotes(
                    campaign="RL Data Atlas",
                    recipe_revision=LEGACY_PYTHON_TESTS_RECIPE_REVISION,
                    verification=(
                        "Unverified historical reward-only sample; does not certify test execution. "
                        "setup fidelity executed on all 505 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks across the cohort's "
                        "environments (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=505,
                    admitted=505,
                ),
                notes=(
                    "505 curriculum-easy tasks converted to run in their archived environment, lowered onto "
                    "the digest-pinned base, measured at the pinned upstream revision "
                    "DCAgent__exp_rpt_curriculum-easy @ "
                    "02923004846e4e73862c20962f823a6d05100e7a with zero substitutions. Qualified without "
                    "oracles: per-row setup fidelity matches and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                intended_use=IntendedUse.TRAIN,
                name="curriculum-medium-v1-rebuilt",
                version="2026.11.12.1",
                repo="open-athena/tasktrove-curriculum-medium-v1-taskcompendium-20261112",
                revision="7f6df7d532912fe1b589e727bbd07f03743891f5",
                expected_rows=488,
            ),
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                id="TaskTrove:tasktrove-curriculum-medium-rebuilt-v1",
                upstream_dataset="tasktrove-curriculum-medium-rebuilt-v1",
                upstream_name="tasktrove-curriculum-medium-rebuilt-v1",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="python-tests",
                tags=("agentic", "language:python", "campaign-release", "converted"),
                task_format=TaskFormat.TASKSPEC,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "Production snapshot_identity over every pinned published TaskSpec: 1 distinct agent identities. "
                    "Offline Daytona factory validation refuses all 488 declared package-lock graders. The declared "
                    "grader environments require local execution; this cohort is Daytona-unsafe."
                ),
                daytona_execution_failures=488,
                oracle_validation=OracleValidation.SKIPPED,
                recipe_qa_history=RecipeQAHistoricalNotes(
                    campaign="RL Data Atlas",
                    recipe_revision=LEGACY_PYTHON_TESTS_RECIPE_REVISION,
                    verification=(
                        "Unverified historical reward-only sample; does not certify test execution. "
                        "setup fidelity executed on all 488 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks across the cohort's "
                        "environments (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=488,
                    admitted=488,
                ),
                notes=(
                    "488 curriculum-medium tasks converted to run in their archived environment, lowered onto "
                    "the digest-pinned base, measured at the pinned upstream revision "
                    "DCAgent__exp_rpt_curriculum-medium-v2 @ "
                    "02923004846e4e73862c20962f823a6d05100e7a with zero substitutions. Qualified without "
                    "oracles: per-row setup fidelity matches and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                intended_use=IntendedUse.TRAIN,
                name="e2egit-v2-rebuilt",
                version="2026.11.12.1",
                repo="open-athena/tasktrove-e2egit-v2-taskcompendium-20261112",
                revision="e0c56a19c77cbab15526b681110088ccd9ef0eb3",
                expected_rows=477,
            ),
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                id="TaskTrove:tasktrove-e2egit-v2-rebuilt",
                upstream_dataset="tasktrove-e2egit-v2-rebuilt",
                upstream_name="tasktrove-e2egit-v2-rebuilt",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="python-tests",
                tags=("agentic", "language:python", "campaign-release", "converted"),
                task_format=TaskFormat.TASKSPEC,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "Production snapshot_identity over every pinned published TaskSpec: 1 distinct agent identities. "
                    "Offline Daytona factory validation refuses all 477 declared package-lock graders. The declared "
                    "grader environments require local execution; this cohort is Daytona-unsafe."
                ),
                daytona_execution_failures=477,
                oracle_validation=OracleValidation.SKIPPED,
                recipe_qa_history=RecipeQAHistoricalNotes(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/python_tests.py version 8; verifyit @ f0155c1143",
                    verification=(
                        "Unverified historical reward-only sample; does not certify test execution. "
                        "setup fidelity compared on all 477 converted specs against their native archives; "
                        "the sole shared setup recipe executed successfully with real package installation. "
                        "Production grading with COMPILER_GRADER_PACKAGES, build identity "
                        "42bb937b102dd758b4d1210769750d7e259a036169ef1458d11821809118888b, "
                        "returned 20 negative verdicts (reward 0.0, no infrastructure errors) on the seed-0 sample. "
                        "Evidence is published beside tasks.parquet at the pinned revision"
                    ),
                    reviewed=477,
                    admitted=477,
                ),
                notes=(
                    "477 e2egit-v2 tasks preserve their archived environment, lowered onto its digest-pinned base. "
                    "The upstream is DCAgent__exp_rpt_e2egit-v2 at 02923004846e4e73862c20962f823a6d05100e7a; "
                    "ten reviewed source defects remain excluded. The private release includes setup and liveness "
                    "evidence, retained rejections, and an explicit durable compiler grader lock. "
                    "Qualification uses an empty submission; no oracle or model-quality run is claimed. "
                    "Daytona execution remains untested"
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                intended_use=IntendedUse.TRAIN,
                name="e2egit-large-v1-rebuilt",
                version="2026.11.12.1",
                repo="open-athena/tasktrove-e2egit-large-v1-taskcompendium-20261112",
                revision="223300482ef034a5014aed9bbb1750f5d9cd89b2",
                expected_rows=4993,
            ),
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                id="TaskTrove:tasktrove-e2egit-large-rebuilt-v1",
                upstream_dataset="tasktrove-e2egit-large-rebuilt-v1",
                upstream_name="tasktrove-e2egit-large-rebuilt-v1",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="python-tests",
                tags=("agentic", "language:python", "campaign-release", "converted"),
                task_format=TaskFormat.TASKSPEC,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "Production snapshot_identity over every pinned published TaskSpec: 1 distinct agent identities. "
                    "Offline Daytona factory validation refuses all 4,993 declared package-lock graders. The declared "
                    "grader environments require local execution; this cohort is Daytona-unsafe."
                ),
                daytona_execution_failures=4993,
                oracle_validation=OracleValidation.SKIPPED,
                recipe_qa_history=RecipeQAHistoricalNotes(
                    campaign="RL Data Atlas",
                    recipe_revision=LEGACY_PYTHON_TESTS_RECIPE_REVISION,
                    verification=(
                        "Unverified historical reward-only sample; does not certify test execution. "
                        "setup fidelity executed on all 4,993 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks across the cohort's "
                        "environments (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=4993,
                    admitted=4993,
                ),
                notes=(
                    "4,993 e2egit-large tasks converted to run in their archived environment, lowered onto "
                    "the digest-pinned base, measured at the pinned upstream revision DCAgent__exp_rpt_e2egit-large @ "
                    "02923004846e4e73862c20962f823a6d05100e7a with zero substitutions. Qualified without "
                    "oracles: per-row setup fidelity matches and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                intended_use=IntendedUse.TRAIN,
                name="multifile-v1-rebuilt",
                version="2026.11.12.1",
                repo="open-athena/tasktrove-multifile-v1-taskcompendium-20261112",
                revision="422b336d6056c3770864e6ef26825731290b2c57",
                expected_rows=4842,
            ),
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                id="TaskTrove:tasktrove-multifile-rebuilt-v1",
                upstream_dataset="tasktrove-multifile-rebuilt-v1",
                upstream_name="tasktrove-multifile-rebuilt-v1",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="python-tests",
                tags=("agentic", "language:python", "campaign-release", "converted"),
                task_format=TaskFormat.TASKSPEC,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "Production snapshot_identity over every pinned published TaskSpec: 1 distinct agent identities. "
                    "Offline Daytona factory validation refuses all 4,842 declared package-lock graders. The declared "
                    "grader environments require local execution; this cohort is Daytona-unsafe."
                ),
                daytona_execution_failures=4842,
                oracle_validation=OracleValidation.SKIPPED,
                recipe_qa_history=RecipeQAHistoricalNotes(
                    campaign="RL Data Atlas",
                    recipe_revision=LEGACY_PYTHON_TESTS_RECIPE_REVISION,
                    verification=(
                        "Unverified historical reward-only sample; does not certify test execution. "
                        "setup fidelity executed on all 4,842 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks across the cohort's "
                        "environments (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=4842,
                    admitted=4842,
                ),
                notes=(
                    "4,842 multifile tasks converted to run in their archived environment, lowered onto "
                    "the digest-pinned base, measured at the pinned upstream revision DCAgent__exp_rpt_multifile-v3 @ "
                    "02923004846e4e73862c20962f823a6d05100e7a with zero substitutions. Qualified without "
                    "oracles: per-row setup fidelity matches and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                intended_use=IntendedUse.TRAIN,
                name="pymethods2test-v1-rebuilt",
                version="2026.11.12.1",
                repo="open-athena/tasktrove-pymethods2test-v1-taskcompendium-20261112",
                revision="f6a20442763e152ec662483dc0ee1476caa6d541",
                expected_rows=500,
            ),
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                id="TaskTrove:tasktrove-pymethods2test-rebuilt-v1",
                upstream_dataset="tasktrove-pymethods2test-rebuilt-v1",
                upstream_name="tasktrove-pymethods2test-rebuilt-v1",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="python-tests",
                tags=("agentic", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKSPEC,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "Production snapshot_identity over every pinned published TaskSpec: 1 distinct agent identities. "
                    "Offline Daytona factory validation refuses all 500 declared package-lock graders. The declared "
                    "grader environments require local execution; this cohort is Daytona-unsafe."
                ),
                daytona_execution_failures=500,
                oracle_validation=OracleValidation.SKIPPED,
                recipe_qa_history=RecipeQAHistoricalNotes(
                    campaign="RL Data Atlas",
                    recipe_revision=LEGACY_PYTHON_TESTS_RECIPE_REVISION,
                    verification=(
                        "Unverified historical reward-only sample; does not certify test execution. "
                        "setup fidelity executed on all 500 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks across the cohort's "
                        "environments (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=500,
                    admitted=500,
                ),
                notes=(
                    "500 pymethods2test tasks converted to run in their archived environment, lowered onto "
                    "the digest-pinned base, measured at the pinned upstream revision "
                    "DCAgent__exp_rpt_pymethods2test-v3 @ "
                    "02923004846e4e73862c20962f823a6d05100e7a with zero substitutions. Qualified without "
                    "oracles: per-row setup fidelity matches and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                intended_use=IntendedUse.TRAIN,
                name="pymethods2test-large-v1-rebuilt",
                version="2026.11.12.1",
                repo="open-athena/tasktrove-pymethods2test-large-v1-taskcompendium-20261112",
                revision="a9ed5c72ade6670853fc28af8bf1b59541d3827c",
                expected_rows=4990,
            ),
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                id="TaskTrove:tasktrove-pymethods2test-large-rebuilt-v1",
                upstream_dataset="tasktrove-pymethods2test-large-rebuilt-v1",
                upstream_name="tasktrove-pymethods2test-large-rebuilt-v1",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="python-tests",
                tags=("agentic", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKSPEC,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "Production snapshot_identity over every pinned published TaskSpec: 1 distinct agent identities. "
                    "Offline Daytona factory validation refuses all 4,990 declared package-lock graders. The declared "
                    "grader environments require local execution; this cohort is Daytona-unsafe."
                ),
                daytona_execution_failures=4990,
                oracle_validation=OracleValidation.SKIPPED,
                recipe_qa_history=RecipeQAHistoricalNotes(
                    campaign="RL Data Atlas",
                    recipe_revision=LEGACY_PYTHON_TESTS_RECIPE_REVISION,
                    verification=(
                        "Unverified historical reward-only sample; does not certify test execution. "
                        "setup fidelity executed on all 4,990 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks across the cohort's "
                        "environments (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=4990,
                    admitted=4990,
                ),
                notes=(
                    "4,990 pymethods2test-large tasks converted to run in their archived environment, lowered onto "
                    "the digest-pinned base, measured at the pinned upstream revision "
                    "DCAgent__exp_rpt_pymethods2test-large-v2 @ "
                    "02923004846e4e73862c20962f823a6d05100e7a with zero substitutions. Qualified without "
                    "oracles: per-row setup fidelity matches and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                intended_use=IntendedUse.TRAIN,
                name="stack-pytest-v1-rebuilt",
                version="2026.11.12.1",
                repo="open-athena/tasktrove-stack-pytest-v1-taskcompendium-20261112",
                revision="861bb03356b5267feba839df7bfc45c14589577c",
                expected_rows=492,
            ),
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                id="TaskTrove:tasktrove-stack-pytest-rebuilt-v1",
                upstream_dataset="tasktrove-stack-pytest-rebuilt-v1",
                upstream_name="tasktrove-stack-pytest-rebuilt-v1",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="python-tests",
                tags=("agentic", "language:python", "campaign-release", "converted"),
                task_format=TaskFormat.TASKSPEC,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "Production snapshot_identity over every pinned published TaskSpec: 1 distinct agent identities. "
                    "Offline Daytona factory validation refuses all 492 declared package-lock graders. The declared "
                    "grader environments require local execution; this cohort is Daytona-unsafe."
                ),
                daytona_execution_failures=492,
                oracle_validation=OracleValidation.SKIPPED,
                recipe_qa_history=RecipeQAHistoricalNotes(
                    campaign="RL Data Atlas",
                    recipe_revision=LEGACY_PYTHON_TESTS_RECIPE_REVISION,
                    verification=(
                        "Unverified historical reward-only sample; does not certify test execution. "
                        "setup fidelity executed on all 492 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks across the cohort's "
                        "environments (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=492,
                    admitted=492,
                ),
                notes=(
                    "492 stack-pytest tasks converted to run in their archived environment, lowered onto "
                    "the digest-pinned base, measured at the pinned upstream revision "
                    "DCAgent__exp_rpt_stack-pytest-v2 @ "
                    "02923004846e4e73862c20962f823a6d05100e7a with zero substitutions. Qualified without "
                    "oracles: per-row setup fidelity matches and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                intended_use=IntendedUse.TRAIN,
                name="unitsyn-python-v1-rebuilt",
                version="2026.11.12.1",
                repo="open-athena/tasktrove-unitsyn-python-v1-taskcompendium-20261112",
                revision="3ed40d26a51c8ca0cf26c15cc84316409534a937",
                expected_rows=4991,
            ),
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                id="TaskTrove:tasktrove-unitsyn-python-rebuilt-v1",
                upstream_dataset="tasktrove-unitsyn-python-rebuilt-v1",
                upstream_name="tasktrove-unitsyn-python-rebuilt-v1",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="python-tests",
                tags=("agentic", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKSPEC,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "Production snapshot_identity over every pinned published TaskSpec: 1 distinct agent identities. "
                    "Offline Daytona factory validation refuses all 4,991 declared package-lock graders. The declared "
                    "grader environments require local execution; this cohort is Daytona-unsafe."
                ),
                daytona_execution_failures=4991,
                oracle_validation=OracleValidation.SKIPPED,
                recipe_qa_history=RecipeQAHistoricalNotes(
                    campaign="RL Data Atlas",
                    recipe_revision=LEGACY_PYTHON_TESTS_RECIPE_REVISION,
                    verification=(
                        "Unverified historical reward-only sample; does not certify test execution. "
                        "setup fidelity executed on all 4,991 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks across the cohort's "
                        "environments (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=4991,
                    admitted=4991,
                ),
                notes=(
                    "Excluded: this release contains the large Unitsyn population rather than the 491-row small cohort. "
                    "4,991 tasks converted to run in their archived environment, lowered onto "
                    "the digest-pinned base, measured at the pinned upstream revision "
                    "DCAgent__exp_rpt_unitsyn-python-v4 @ "
                    "02923004846e4e73862c20962f823a6d05100e7a with zero substitutions. Qualified without "
                    "oracles: per-row setup fidelity matches and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                intended_use=IntendedUse.TRAIN,
                name="unitsyn-python-large-v1-rebuilt",
                version="2026.11.12.1",
                repo="open-athena/tasktrove-unitsyn-python-large-v1-taskcompendium-20261112",
                revision="3a64ac3ed258bd184c24c24e7c7578f64142c180",
                expected_rows=4991,
            ),
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                id="TaskTrove:tasktrove-unitsyn-python-large-rebuilt-v1",
                upstream_dataset="tasktrove-unitsyn-python-large-rebuilt-v1",
                upstream_name="tasktrove-unitsyn-python-large-rebuilt-v1",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="python-tests",
                tags=("agentic", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKSPEC,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "Production snapshot_identity over every pinned published TaskSpec: 1 distinct agent identities. "
                    "Offline Daytona factory validation refuses all 4,991 declared package-lock graders. The declared "
                    "grader environments require local execution; this cohort is Daytona-unsafe."
                ),
                daytona_execution_failures=4991,
                oracle_validation=OracleValidation.SKIPPED,
                recipe_qa_history=RecipeQAHistoricalNotes(
                    campaign="RL Data Atlas",
                    recipe_revision=LEGACY_PYTHON_TESTS_RECIPE_REVISION,
                    verification=(
                        "Unverified historical reward-only sample; does not certify test execution. "
                        "setup fidelity executed on all 4,991 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks across the cohort's "
                        "environments (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=4991,
                    admitted=4991,
                ),
                notes=(
                    "4,991 unitsyn-python-large tasks converted to run in their archived environment, lowered onto "
                    "the digest-pinned base, measured at the pinned upstream revision "
                    "DCAgent__exp_rpt_unitsyn-python-large-v2 @ "
                    "02923004846e4e73862c20962f823a6d05100e7a with zero substitutions. Qualified without "
                    "oracles: per-row setup fidelity matches and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
                ),
            ),
        ),
        RlDataSource(
            pipeline=import_harbor_prebuilt,
            config=swesmith_repaired,
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                id="TaskTrove:swesmith-verifier-repaired-v5",
                upstream_dataset="swesmith-verifier-repaired-v5",
                upstream_name="swesmith-verifier-repaired-v5",
                subset=None,
                build_path="Snowball Smartie RL / Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="swe-repo",
                tags=("agentic", "language:python", "campaign-release", "verified"),
                task_format=TaskFormat.HARBOR,
                count=swesmith_repaired.expected_rows,
                dataset=swesmith_repaired.dataset,
                daytona_snapshot_count=0,
                daytona_execution_failures=0,
                daytona_count_basis=(
                    "Coordinator ruling 2026-11-25 13:00 EST: declarative build contexts are acceptable "
                    "for admission with execution evidence. The cohort's passing acceptance run "
                    "(acceptance-swesmith-verifier-repaired-v5) built and executed its contexts through "
                    "Daytona's declarative path without named snapshots; declared demands measured from "
                    "the staged population artifact without provider calls: 7016 machine slots, zero "
                    "execution failures. Conversion identity harbor-cohort-taskspec-v1 @ "
                    "a94870c78d23d1c569987f4146d379fad9e4b2ec. Evidence: "
                    "s3://marin-us-east-02a/marin/rl-data/staged-census-20261126/"
                    "swesmith-verifier-repaired-v5/"
                ),
                oracle_validation=OracleValidation.EXECUTED,
                recipe_qa_history=RecipeQAHistoricalNotes(
                    campaign="SWE-smith verifier repair v5; taskspec conversion",
                    recipe_revision="a94870c78d23d1c569987f4146d379fad9e4b2ec",
                    verification=(
                        "The taskspec conversion converted every input row with zero rejections "
                        "(converted_rows 7016 == input_rows 7016); the cohort's passing acceptance run "
                        "(acceptance-swesmith-verifier-repaired-v5) replayed the recorded bug controls "
                        "through Daytona against the converted graders per the 2026-11-24 ruling, and "
                        "that passing run is the declarative execution evidence. The per-task grading "
                        "boundary census assessed all 7016 tasks (required separate: 7016; status "
                        "matched: 7016). The prior release's control evidence stays at its revisions."
                    ),
                    reviewed=7016,
                    admitted=7016,
                ),
                independent_qc=IndependentQC(
                    verdict=Verdict.GOOD,
                    method=(
                        "per-task grading-boundary census over the complete staged population plus the "
                        "passing acceptance replay; the conversion receipts account for every row"
                    ),
                    reviewed=7016,
                    admitted=7016,
                    evidence_url="s3://marin-us-east-02a/marin/rl-data/grading-census-taskspec-20261126/",
                ),
                notes=(
                    "7,016 SWE-smith tasks re-pinned to the taskspec conversion release "
                    "(declared-pin ruling 2026-11-25 ~23:35 EST): the same upstream content converted "
                    "by harbor-cohort-taskspec-v1 with built executable graders; per-task byte and "
                    "grader identity verified against the prior conversion output. The prior release's "
                    "qualification evidence (5,628 admitted v4 packages, the control receipts, the "
                    "independent review) stays at its revisions and in this file's history. The "
                    "repair-campaign repositories are private."
                ),
            ),
        ),
        # RL Data Atlas QA campaigns: eight TaskTrove cohorts privately published after
        # conversion-fidelity, grader-soundness and blinded-reference review. Each
        # pinned revision's build.json records the measured rejections, dedup collapse
        # and row counts; the evidence strings below carry the measured results.
        # Conversation-only tasks import their canonical task specs without a runtime
        # envelope; sample controls ran on the archive-wide seeded samples, not the
        # full populations.
        _tasktrove_qa_release(
            "math_oracle",
            "math-answer",
            revision="1ac3735fcbd85bb059053b37d1c5891622e4fc5a",
            expected_rows=57424,
            recipe_revision="datasets/tasktrove/math.py v3; TaskTrove @ 02923004846e",
            evidence=(
                "All 57,777 rows re-converted and compared row-by-row against the archive with zero "
                "discrepancies; 353 unparsable-reference rows rejected at conversion. Contained grader "
                "controls on the 24-task seeded sample: golden 23/24 (the failing task is the "
                "unparseable-reference row, now rejected), empty 24/24, wrong 24/24. Blinded review: "
                "19/19 solved cases agree."
            ),
            notes=(
                "5,000-scale scalar sympy compare with hidden gold references. 353 rows whose typed "
                "references the production math-verify parser rejects are excluded at admission. "
                "Private release; conversation-only tasks import canonical task specs. The pinned "
                "cohort revision was downloaded back and every task hash re-validated."
            ),
        ),
        _tasktrove_qa_release(
            "math_stack",
            "math-answer",
            revision="2cef971f0e6de40011aa5b5a423aca97be27cac7",
            expected_rows=110604,
            recipe_revision="datasets/tasktrove/math.py v3; TaskTrove @ 02923004846e",
            evidence=(
                "All 110,730 rows re-converted and compared row-by-row against the archive (the single "
                "pre-existing reviewed defect unchanged); 125 unparsable-reference rows rejected at "
                "conversion. Contained grader controls on the 24-task seeded sample: 72/72. Blinded "
                "review: 20/20 solved cases agree."
            ),
            notes=(
                "Typed sympy comparison with per-task oracle. 125 rows whose typed references the "
                "production math-verify parser rejects are excluded at admission. Private release; "
                "conversation-only tasks import canonical task specs. The pinned cohort revision was "
                "downloaded back and every task hash re-validated."
            ),
        ),
        _tasktrove_qa_release(
            "math_openreasoning",
            "math-answer",
            revision="af81284226c98852a60e0946ec084867e22875e1",
            expected_rows=42621,
            recipe_revision="datasets/tasktrove/math.py v3; TaskTrove @ 02923004846e",
            evidence=(
                "All 42,636 rows re-converted and compared row-by-row against the archive with zero "
                "discrepancies; 15 unparsable-reference rows rejected at conversion. Contained grader "
                "controls on the 24-task seeded sample: 72/72. Blinded review: 16/16 addressed cases "
                "agree; 2 sampled rows carry suspected conversion defects (statements unanswerable as "
                "converted), recorded in the audit evidence."
            ),
            notes=(
                "Open-reasoning problems with a rigorous scalar/interval/set/tuple/equation verifier "
                "and oracle. 15 rows whose typed references the production math-verify parser rejects "
                "are excluded at admission. QA (RL Data Atlas): blinded review flagged 2 sampled rows "
                "as underdetermined; both were compared against the raw archive and the converted "
                "statements match the upstream instructions verbatim, so the defects are in the source "
                "data, not the conversion. 2 of the 20-case blinded panel; exactly the inclusive 10% boundary. "
                "Private release; conversation-only tasks import canonical task specs. The pinned cohort "
                "revision was downloaded back and every task hash re-validated."
            ),
        ),
        _tasktrove_qa_release(
            "math_gym",
            "math-answer",
            revision="8b9af73ebed3b62f5a73dacf6c238cf2d82391ef",
            expected_rows=3941,
            recipe_revision="datasets/tasktrove/math.py v3; TaskTrove @ 02923004846e",
            evidence=(
                "All 4,096 rows re-converted and compared row-by-row against the archive with zero "
                "discrepancies; 155 unparsable-reference rows rejected at conversion. Contained grader "
                "controls on the 24-task seeded sample: 72/72. Blinded review: 7/7 solved cases agree "
                "(35% of cases solvable within the review's budget; the rest excluded as "
                "competition-difficulty with coverage reported)."
            ),
            notes=(
                "Strict trailing boxed compare with no oracle. 155 rows whose typed references the "
                "production math-verify parser rejects are excluded at admission. Private release; "
                "conversation-only tasks import canonical task specs. The pinned cohort revision was "
                "downloaded back and every task hash re-validated. "
                "Content-quality finding 2026-11-18 (SAMPLE receipt fb053f6439…, pinned GLM-5.3 v3, "
                "seed-0 panel at the r23 pin): 91 of 3,941 current-population rows (2.3%) carry "
                "LaTeX-wrapped expected-answer strings (\\(\\displaystyle …\\)) in grader "
                "parameters.expected with math_type=scalar, which the scalar comparison path cannot "
                "parse; the SAMPLE review flags 1 of its 8 issues naming that symptom "
                "(tasktrove-math_gym-bb15c01fe058e37e…). Grader and scorer stay unchanged (faithful "
                "environment). A private remediated version normalizing ONLY those expected-answer "
                "strings remains available as data repair if ruled material; it would need its own "
                "SAMPLE QA and admission, and this original stays as is."
            ),
        ),
        _tasktrove_qa_release(
            "math_prism",
            "math-answer",
            revision="ddb6f6cc8dbb09638d496739ab204b3ba5c6c37e",
            expected_rows=2221,
            recipe_revision="datasets/tasktrove/math.py v3; TaskTrove @ 02923004846e",
            evidence=(
                "All 2,404 rows re-converted and compared row-by-row against the archive with zero "
                "discrepancies; 2 unparsable-reference rows rejected at conversion; 181 exact duplicates "
                "collapse under the pipeline dedup key, 2,221 retained. Contained grader controls on the "
                "24-task seeded sample: 72/72. Blinded review of 20 seeded cases: 8 solvable blind; after "
                "adjudication all 8 references verified correct (3 solver errors on the reviewer's side, "
                "each confirmed by independent recomputation), 12 cases unsolved blind so their references "
                "remain unverified by this method. An earlier provisional impossible-reference claim was "
                "retracted: the connected 20-vertex/21-edge graph is bicyclic and theta(7,7,7) attains the "
                "recorded 147 spanning trees, confirmed by two independent Matrix-Tree determinant checks."
            ),
            notes=(
                "Symbolic exact compare with hidden gold references. Final v2 content QC reports 286 issues "
                "among 2,213 conclusive reviews (12.9236%, SOME_ISSUES); seven valid uncertain observations "
                "and one unreviewed task are excluded from that denominator. "
                "tasktrove-math_prism-2c73de38c5427048ffb943add4fee2759b8f63f8fb172c4a39f28ee94e97e0c0: "
                "QC unreviewed: review exceeded 16,384 tokens. All 2,221 tasks remain in the private release. "
                "The 2,220 valid primary reviews are preserved byte for byte. Content coverage remains incomplete. "
                "The pinned cohort revision was downloaded back and every task hash re-validated."
            ),
        ),
        _tasktrove_qa_release(
            "safety",
            "llm-judge-freeform",
            revision="b4f15e07796b5bea1f311892583ecce129efe0fa",
            expected_rows=44939,
            reviewed=10,
            recipe_revision="datasets/tasktrove/judged.py v3; MEDIUM tier under RUBRIC_CHECKLIST",
            evidence=(
                "All 89,066 rows re-converted and compared row-by-row against the archive with zero "
                "discrepancies; 44,127 exact duplicates collapse under the pipeline dedup key, 44,939 "
                "retained. MEDIUM-tier checklist judge e2e on the 10-task seeded sample through the "
                "production grading boundary at temperature 0.7: 29/30 as expected (one conservative "
                "0.5 on a wrong submission expected to score 0.0). Self-judged verdicts recorded "
                "before any judge run."
            ),
            notes=(
                "Judge-only against a per-prompt two-line safety principle. No endpoint is hard-coded "
                "in any task spec; the MEDIUM JUDGE_TIERS definition pins model and completion settings. "
                "Private release; conversation-only tasks import canonical task specs. The pinned cohort "
                "revision was downloaded back and every task hash re-validated."
            ),
        ),
        _tasktrove_qa_release(
            "knowledge-openqa",
            "qa-short-answer",
            revision="f8f83ffcc9779f8314f381a095de2d29525e48dc",
            expected_rows=122314,
            reviewed=10,
            recipe_revision="datasets/tasktrove/qa.py v5; MEDIUM JUDGE_TIERS pin; TaskTrove @ 02923004846e",
            evidence=(
                "All 122,357 rows re-converted and compared row-by-row against the archive with zero "
                "discrepancies; 43 exact duplicates collapse under the pipeline dedup key, 122,314 "
                "retained. MEDIUM-tier reference judge e2e on the 10-task seeded sample through the "
                "production grading boundary at temperature 0.7: 30/30 as expected."
            ),
            notes=(
                "Exact gate then reference-based judge. No endpoint is hard-coded in any task spec; "
                "the MEDIUM JUDGE_TIERS definition pins model and completion settings. Private release; "
                "conversation-only tasks import canonical task specs. The pinned cohort revision was "
                "downloaded back and every task hash re-validated."
            ),
        ),
        _tasktrove_qa_release(
            "science-openqa",
            "llm-judge-freeform",
            revision="e3b624988578da13c523fcc3e39e65f752a781ff",
            expected_rows=150644,
            reviewed=10,
            recipe_revision="datasets/tasktrove/qa.py v5; HEAVY JUDGE_TIERS pin; TaskTrove @ 02923004846e",
            evidence=(
                "All 150,644 rows re-converted and compared row-by-row against the archive with zero "
                "discrepancies; no dedup collisions. HEAVY-tier reference judge e2e on the 10-task "
                "seeded sample through the production grading boundary against the pinned GLM relay at "
                "temperature 0.7: 30/30 as expected, including the small-angle leniency trap flagged "
                "before the run."
            ),
            notes=(
                "Reference answer plus scientific-equivalence judge. The tier is chosen for the source "
                "(the archive demonstrably contains operator-algebra equivalence items a MEDIUM judge "
                "would mis-grade). No endpoint is hard-coded in any task spec; the HEAVY JUDGE_TIERS "
                "definition pins model and completion settings. Private release; conversation-only "
                "tasks import canonical task specs. The pinned cohort revision was downloaded back and "
                "every task hash re-validated."
            ),
        ),
        _puzzles_qa_release(),
    ]
