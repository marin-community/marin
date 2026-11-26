# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed manual exclusion claims for sources excluded by policy rather than admission state.

Bare ``excluded`` tags and migrated ``deprecated`` tags carry a ruling the
admission checks cannot derive. This registry records that ruling as an
``ExclusionReason`` member with typed evidence. Every source that is Excluded
because of a tag must appear here or carry an admission-derived claim; sources
outside this registry get their reasons from the admission checks alone.

Evidence here quotes the source's recorded notes. The notes remain the fuller
record on the declaration; the claim keeps the deciding sentence.
"""

from finestore.rl.status_taxonomy import ExclusionClaim, ExclusionEvidence, ExclusionEvidenceKind, ExclusionReason


def _note(message: str) -> ExclusionEvidence:
    return ExclusionEvidence(kind=ExclusionEvidenceKind.NOTE, message=message)


MANUAL_EXCLUSION_CLAIMS: dict[str, tuple[ExclusionClaim, ...]] = {
    # The 2026-11-26 15:20 EST user rule (nothing stays Pending indefinitely): intake
    # rows whose owner shows no live work driving them to admission by the atlas-
    # 2026.11.27 release are Excluded with their typed reason, the clearing path and
    # their owner. They keep their measured receipts and return to intake when the
    # path clears.
    "TaskTrove:laion__glaive-code-assistant-sandboxes-verified-v2": (
        ExclusionClaim(
            reason=ExclusionReason.GRADER_JUDGE_UNAVAILABLE,
            evidence=_note(
                "Staged intake with no live driver (the 2026-11-26 15:20 EST user rule): the declared "
                "judge grader (MEDIUM MiniMax-M3, binary full-rubric, no reference) has no runtime "
                "qualification — CANDIDATES parks it for the coordinator-directed fresh three-source "
                "rotation and ingest B's live groups (QA-Abstention, Formatting, Edge) do not include "
                "it. Judge-only over code that is never run; graders fail closed. Clears when the "
                "rotation claims it and the native-judge runtime qualification (oracle/no-op controls "
                "through the declared judge) seals."
            ),
        ),
    ),
    "TaskTrove:laion__stackexchange-tezos-sandboxes-verified-v2": (
        ExclusionClaim(
            reason=ExclusionReason.GRADER_JUDGE_UNAVAILABLE,
            evidence=_note(
                "Staged intake with no live driver (the 2026-11-26 15:20 EST user rule): the declared "
                "judge grader (MEDIUM MiniMax-M3, binary full-rubric, no reference) has no runtime "
                "qualification — CANDIDATES parks it for the coordinator-directed fresh three-source "
                "rotation and ingest B's live groups (QA-Abstention, Formatting, Edge) do not include "
                "it. Judge-only Tezos answers; the _copyN duplicates fall to dedup. Clears when the "
                "rotation claims it and the native-judge runtime qualification seals."
            ),
        ),
    ),
    "TaskTrove:tasktrove-unitsyn-python-rebuilt-v1": (
        ExclusionClaim(
            reason=ExclusionReason.RETAINED_POPULATION_EMPTY,
            evidence=_note(
                "The declared 491-row small cohort was never published (the 2026-11-26 15:20 EST user "
                "rule): the pinned release carries the LARGE Unitsyn population (4,991 rows) instead of "
                "the 491-row small cohort the declaration names, so the declared retained population is "
                "empty at this pin. Ingest A owns the republish; no republish job is live. Clears when "
                "the small cohort is republished at its declared pin and the chain re-measures."
            ),
        ),
    ),
    "Anthropic:hh_rlhf/harmless-base": (
        ExclusionClaim(
            reason=ExclusionReason.GRADER_MISSING,
            evidence=_note(
                "Not a valid RL source as declared (ingest C decision, brief-delegated 2026-11-26): every "
                "converted task carries NoGrader — the preference converter's typed reason is 'Preference "
                "labels rank source candidates; scoring a new reply needs a reward model' and the Atlas has "
                "no reward-model grader, so no task can produce a reward; main declares the preference "
                "sources 'kept for review only'. Sealed evidence at the pin: the sibling subsets' grading "
                "censuses (converter 1509fc8638f6) report 100% 'No executable grader is declared' "
                "(helpful-base 43,699/43,699, helpful-online 21,989, helpful-rejection-sampled 52,382) and "
                "this subset's 42,378-task census shares that converter and declaration. MIT license gate "
                "PASSES (registry ingest_c_intake_20261126_a_v1) — the rows are untrainable as RL, not "
                "banned. Clears only with an executable or judge grader and a zero-missing grading census."
            ),
        ),
    ),
    "Anthropic:hh_rlhf/helpful-base": (
        ExclusionClaim(
            reason=ExclusionReason.GRADER_MISSING,
            evidence=_note(
                "Not a valid RL source as declared (ingest C decision, brief-delegated 2026-11-26): the "
                "sealed grading census at this pin reports 43,699/43,699 tasks 'No executable grader is "
                "declared' (s3://marin-us-east-02a/marin/rl-data/atlas/grading-census/2026.11.13.1/results/"
                "d587beb02be5706914cc/receipt.json) — the preference converter emits NoGrader ('Preference "
                "labels rank source candidates; scoring a new reply needs a reward model'). MIT license gate "
                "PASSES (registry ingest_c_intake_20261126_a_v1) — untrainable as RL, not banned. Clears "
                "only with an executable or judge grader and a zero-missing grading census."
            ),
        ),
    ),
    "Anthropic:hh_rlhf/helpful-online": (
        ExclusionClaim(
            reason=ExclusionReason.GRADER_MISSING,
            evidence=_note(
                "Not a valid RL source as declared (ingest C decision, brief-delegated 2026-11-26): the "
                "sealed grading census at this pin reports 21,989/21,989 tasks 'No executable grader is "
                "declared' — the preference converter emits NoGrader ('Preference labels rank source "
                "candidates; scoring a new reply needs a reward model'). MIT license gate PASSES (registry "
                "ingest_c_intake_20261126_a_v1) — untrainable as RL, not banned. Clears only with an "
                "executable or judge grader and a zero-missing grading census."
            ),
        ),
    ),
    "Anthropic:hh_rlhf/helpful-rejection-sampled": (
        ExclusionClaim(
            reason=ExclusionReason.GRADER_MISSING,
            evidence=_note(
                "Not a valid RL source as declared (ingest C decision, brief-delegated 2026-11-26): the "
                "sealed grading census at this pin reports 52,382/52,382 tasks 'No executable grader is "
                "declared' — the preference converter emits NoGrader ('Preference labels rank source "
                "candidates; scoring a new reply needs a reward model'). MIT license gate PASSES (registry "
                "ingest_c_intake_20261126_a_v1) — untrainable as RL, not banned. Clears only with an "
                "executable or judge grader and a zero-missing grading census."
            ),
        ),
    ),
    "Nemotron:nemotron_ultra_mopd/hs3_en": (
        ExclusionClaim(
            reason=ExclusionReason.GRADER_MISSING,
            evidence=_note(
                "Excluded per the user correction (2026-11-26 3:20 PM EST; ingest C): no executable grader is"
                " declared on any task — the blend converter emits NoGrader text tasks, and the sibling hs3"
                " grading censuses at this pin report 100% 'No executable grader is declared' (hs3_multi 962,"
                " hs3_multiturn 1,243). cc-by-4.0 license gate PASSES (registry ingest_c_intake_20261126_a_v1)"
                " — re-admissible with an executable or reward-model grader plus a zero-missing grading"
                " census; not banned."
            ),
        ),
    ),
    "Nemotron:nemotron_ultra_mopd/hs3_multi": (
        ExclusionClaim(
            reason=ExclusionReason.GRADER_MISSING,
            evidence=_note(
                "Excluded per the user correction (2026-11-26 3:20 PM EST; ingest C): the sealed grading"
                " census at this pin reports 962/962 tasks 'No executable grader is declared' (converter"
                " 1509fc8638f6, s3://marin-us-east-02a/marin/rl-data/atlas/grading-census/2026.11.13.1/)."
                " cc-by-4.0 license gate PASSES — re-admissible with an executable or reward-model grader"
                " plus a zero-missing grading census; not banned."
            ),
        ),
    ),
    "Nemotron:nemotron_ultra_mopd/hs3_multiturn": (
        ExclusionClaim(
            reason=ExclusionReason.GRADER_MISSING,
            evidence=_note(
                "Excluded per the user correction (2026-11-26 3:20 PM EST; ingest C): the sealed grading"
                " census at this pin reports 1,243/1,243 tasks 'No executable grader is declared' (converter"
                " 1509fc8638f6). cc-by-4.0 license gate PASSES — re-admissible with an executable or"
                " reward-model grader plus a zero-missing grading census; not banned."
            ),
        ),
    ),
    "Nemotron:nemotron_ultra_mopd/makeshn_ultra_v3_ipi_train": (
        ExclusionClaim(
            reason=ExclusionReason.GRADER_MISSING,
            evidence=_note(
                "Excluded per the user correction (2026-11-26 3:20 PM EST; ingest C): the sealed grading"
                " census at this pin reports 2,000/2,000 tasks 'No executable grader is declared', and the"
                " actor census reports every task blocked on 'agent: no declared environment' (execution"
                " requirement). cc-by-4.0 license gate PASSES — re-admissible with an executable or"
                " reward-model grader plus a zero-missing grading census; not banned."
            ),
        ),
    ),
    "Nemotron:nemotron_ultra_mopd/safety_en": (
        ExclusionClaim(
            reason=ExclusionReason.GRADER_MISSING,
            evidence=_note(
                "Excluded per the user correction (2026-11-26 3:20 PM EST; ingest C): the sealed grading"
                " census at this pin reports 629/629 tasks 'No executable grader is declared' (converter"
                " 1509fc8638f6). cc-by-4.0 license gate PASSES — re-admissible with an executable or"
                " reward-model grader plus a zero-missing grading census; not banned."
            ),
        ),
    ),
    "Nemotron:nemotron_ultra_mopd/swe_pivot_len40k/SWE-Gym/SWE-Gym": (
        ExclusionClaim(
            reason=ExclusionReason.GRADER_MISSING,
            evidence=_note(
                "Excluded per the user correction (2026-11-26 3:20 PM EST; ingest C): the sealed grading"
                " census at this pin reports 860/860 tasks 'No executable grader is declared' (converter"
                " 1509fc8638f6, s3://marin-us-east-02a/marin/rl-data/atlas/grading-census/2026.11.13.1/"
                "results/449f65753d610f85f4c8/receipt.json). cc-by-4.0 license gate PASSES — re-admissible"
                " with an executable or reward-model grader plus a zero-missing grading census; not banned."
            ),
        ),
    ),
    "Nemotron:nemotron_ultra_mopd/swe_pivot_len40k/nebius/SWE-rebench-V2": (
        ExclusionClaim(
            reason=ExclusionReason.GRADER_MISSING,
            evidence=_note(
                "Excluded per the user correction (2026-11-26 3:20 PM EST; ingest C): the sealed grading"
                " census at this pin reports 3,903/3,903 tasks 'No executable grader is declared' (converter"
                " 1509fc8638f6). cc-by-4.0 license gate PASSES — re-admissible with an executable or"
                " reward-model grader plus a zero-missing grading census; not banned."
            ),
        ),
    ),
    "Nemotron:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/SWE-Gym/SWE-Gym": (
        ExclusionClaim(
            reason=ExclusionReason.GRADER_MISSING,
            evidence=_note(
                "Excluded per the user correction (2026-11-26 3:20 PM EST; ingest C): the sealed grading"
                " census at this pin reports 246/246 tasks 'No executable grader is declared' (converter"
                " 1509fc8638f6). cc-by-4.0 license gate PASSES — re-admissible with an executable or"
                " reward-model grader plus a zero-missing grading census; not banned."
            ),
        ),
    ),
    "Nemotron:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/nebius/SWE-rebench-V2": (
        ExclusionClaim(
            reason=ExclusionReason.GRADER_MISSING,
            evidence=_note(
                "Excluded per the user correction (2026-11-26 3:20 PM EST; ingest C): the sealed grading"
                " census at this pin reports 932/932 tasks 'No executable grader is declared' (converter"
                " 1509fc8638f6). cc-by-4.0 license gate PASSES — re-admissible with an executable or"
                " reward-model grader plus a zero-missing grading census; not banned."
            ),
        ),
    ),
    "TaskTrove:DCAgent__mix_h4_binary_easy": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (1996 tasks) stays intact "
                "pending a pinned rebuild."
            ),
        ),
    ),
    "TaskTrove:XiaomiMiMo__MiMo-V2.6-RL-oss__general": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (925 tasks) stays intact pending "
                "a pinned rebuild."
            ),
        ),
    ),
    "TaskTrove:XiaomiMiMo__MiMo-V2.6-RL-oss__music": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (1000 tasks) stays intact "
                "pending a pinned rebuild."
            ),
        ),
    ),
    "TaskTrove:laion__exp_rpt_bugsinpy-v4": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (479 tasks) stays intact pending "
                "a pinned rebuild."
            ),
        ),
    ),
    "TaskTrove:laion__exp_rpt_nemotron-junit-v6": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (447 tasks) stays intact pending "
                "a pinned rebuild."
            ),
        ),
    ),
    "TaskTrove:laion__exp_rpt_stack-php-large-v9": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (462 tasks) stays intact pending "
                "a pinned rebuild."
            ),
        ),
    ),
    "TaskTrove:laion__mix_h11_single_skill_only-v2": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (2859 tasks) stays intact "
                "pending a pinned rebuild."
            ),
        ),
    ),
    "TaskTrove:laion__mix_h8_original_tests-v2": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (2848 tasks) stays intact "
                "pending a pinned rebuild."
            ),
        ),
    ),
    "TaskTrove:laion__nemotron-gym-agentic-indirect-prompt-injection-v3": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (1272 tasks) stays intact "
                "pending a pinned rebuild."
            ),
        ),
    ),
    "TaskTrove:laion__nemotron-gym-agentic-swe-pivot-v4": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (1541 tasks) stays intact "
                "pending a pinned rebuild."
            ),
        ),
    ),
    "TaskTrove:laion__nemotron-gym-cfbench-v4": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (468 tasks) stays intact pending "
                "a pinned rebuild."
            ),
        ),
    ),
    "TaskTrove:laion__nemotron-gym-identity-following-v4": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (21660 tasks) stays intact "
                "pending a pinned rebuild."
            ),
        ),
    ),
    "TaskTrove:laion__nemotron-gym-instruction-following-adversarial-v5": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (1000 tasks) stays intact "
                "pending a pinned rebuild."
            ),
        ),
    ),
    "TaskTrove:laion__nemotron-gym-instruction-following-multiturnchat-v4": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (1982 tasks) stays intact "
                "pending a pinned rebuild."
            ),
        ),
    ),
    "TaskTrove:laion__nemotron-gym-inverse-ifeval-v4": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (1000 tasks) stays intact "
                "pending a pinned rebuild."
            ),
        ),
    ),
    "TaskTrove:laion__nemotron-gym-knowledge-web-search-mcqa-v2": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (2915 tasks) stays intact "
                "pending a pinned rebuild."
            ),
        ),
    ),
    "TaskTrove:laion__nemotron-gym-qa-abstention-v4": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (3150 tasks) stays intact "
                "pending a pinned rebuild."
            ),
        ),
    ),
    "TaskTrove:laion__nemotron-gym-sysbench-v4": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (1010 tasks) stays intact "
                "pending a pinned rebuild."
            ),
        ),
    ),
    "TaskTrove:DCAgent__code-contests-noblock": (
        ExclusionClaim(
            reason=ExclusionReason.GRADER_DEFECT,
            evidence=_note(
                "Excluded pending SAMPLE/FULL QA of converter version 4. Hidden test_data.json, "
                "only the first case is shown. Exact-line compare; fine."
            ),
        ),
    ),
    "TaskTrove:tasktrove-code-contests-built-image": (
        ExclusionClaim(
            reason=ExclusionReason.GRADER_DEFECT,
            evidence=_note(
                "Excluded: the pinned payload reads only numeric rewards and can hide verifier "
                "failures. Republish with converter version 4 and robust SAMPLE/FULL controls "
                "before admission."
            ),
        ),
    ),
    "TaskTrove:laion__exp_rpt_codenet-python-v4": (
        ExclusionClaim(
            reason=ExclusionReason.GRADER_DEFECT,
            evidence=_note(
                "Only 3 hidden cases and whitespace-collapsing compare. Oracle present; regenerate 20+ cases per task."
            ),
        ),
    ),
    "TaskTrove:laion__exp_rpt_crosscodeeval-csharp-v4": (
        ExclusionClaim(
            reason=ExclusionReason.GRADER_DEFECT,
            evidence=_note("0.25 reward for any identifier-shaped output; instruction coaches the hack."),
        ),
    ),
    "TaskTrove:laion__exp_rpt_crosscodeeval-java-v3": (
        ExclusionClaim(
            reason=ExclusionReason.GRADER_DEFECT,
            evidence=_note("Exact string match on a single line completion. Not agentic, no execution."),
        ),
    ),
    "TaskTrove:laion__exp_rpt_crosscodeeval-typescript-v2": (
        ExclusionClaim(
            reason=ExclusionReason.GRADER_DEFECT,
            evidence=_note("Re-skin of the Python variant with the same free 0.25 tier; metadata still says python."),
        ),
    ),
    "TaskTrove:laion__exp_rpt_methods2test-large-v4": (
        ExclusionClaim(
            reason=ExclusionReason.GRADER_DEFECT,
            evidence=_note(
                "All 10 sampled shipped Java oracles failed because Maven could not resolve its "
                "plugins offline. Recovering the source requires rebuilding the Java grading "
                "environment."
            ),
        ),
    ),
    "TaskTrove:laion__openswe-tasks-patched-v7-oracle-success": (
        ExclusionClaim(
            reason=ExclusionReason.GRADER_DEFECT,
            evidence=_note(
                "No FAIL_TO_PASS ids: the v7 verifier scores whichever tests its custom pytest "
                "guard plugin saw execute, and the repository is cloned by a root-level setup "
                "script at agent time. Needs its own converter."
            ),
        ),
    ),
    "TaskTrove:laion__r2egym-patched-full-oracle-v3": (
        ExclusionClaim(
            reason=ExclusionReason.GRADER_DEFECT,
            evidence=_note(
                "Grades by overlap between test_info.json and expected_output_json rather than "
                "by pytest node id; not the trusted-paths shape the swe converters handle."
            ),
        ),
    ),
    "TaskTrove:laion__swegym-tasks-patched-validated-v5": (
        ExclusionClaim(
            reason=ExclusionReason.GRADER_DEFECT,
            evidence=_note(
                "Image ships no repository: the instruction clones it and runs make init at "
                "agent time, and the old grader pip-installed requirements again at grading "
                "time. Sampled oracles fail on missing dependencies and the empty check cannot "
                "start."
            ),
        ),
    ),
    "TaskTrove:laion__toolscale-v4": (
        ExclusionClaim(
            reason=ExclusionReason.GRADER_DEFECT,
            evidence=_note(
                "Good design (offline tool service) but the CLI script embeds the gold calls and "
                "answer, and the prompt states the conclusion. Move the fixture behind a server "
                "and strip the success criteria."
            ),
        ),
    ),
    "TaskTrove:laion__tulu3-sft-personas-math-sandboxes-verified-v3": (
        ExclusionClaim(
            reason=ExclusionReason.GRADER_DEFECT,
            evidence=_note("Easy SFT persona math, gold in plaintext."),
        ),
        ExclusionClaim(
            reason=ExclusionReason.CONTAMINATION_RISK,
            evidence=_note("Carries the terminal-bench canary."),
        ),
    ),
    "TaskTrove:SankalpKJ__nemotron-code-oracle-filtered": (
        ExclusionClaim(
            reason=ExclusionReason.GRADER_DEFECT,
            evidence=_note(
                "Only test is the example shown in the prompt. Oracle solutions exist, so "
                "generate hidden cases by fuzzing inputs through the oracle."
            ),
        ),
    ),
    "TaskTrove:DCAgent__selfinstruct-naive-sandboxes-2-verified-v3": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "BackendSurvey (v21 preview catalog): every surveyed backend reports unsupported "
                "with per-task validation errors 'agent: Unresolved Docker build context must be "
                "built and pinned before execution'; DifficultyGap.TOOL_PROVIDER_REQUIRED "
                "(2026-11-16 exclude ruling)."
            ),
        ),
    ),
    "TaskTrove:laion__magicoder-v4": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "BackendSurvey (v21 preview catalog): every surveyed backend reports unsupported "
                "with per-task validation errors 'agent: Unresolved Docker build context must be "
                "built and pinned before execution'; DifficultyGap.TOOL_PROVIDER_REQUIRED "
                "(2026-11-16 exclude ruling)."
            ),
        ),
    ),
}

NEEDS_RULING_SOURCES: frozenset[str] = frozenset()
"""Sources whose legacy exclusion reason stays unknown while they wait for readmission.

The user ruled (2026-11-16) that the tagged sources join the readmission list as
P1s and stay visibly Excluded until an ingest agent's admission procedure
clears every check; only then does the bare tag (and this entry) go away.
Empty since 2026-11-17: codeforces and competitive-coding both cleared the
procedure and readmitted.
"""


def manual_exclusion_claims(source_id: str) -> tuple[ExclusionClaim, ...]:
    """Return the recorded manual exclusion claims for a source, empty when none."""
    return MANUAL_EXCLUSION_CLAIMS.get(source_id, ())
