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

from experiments.post_training.task_curation.source_policy import CROSSCODE_EVAL_REASON

_EVAL_ONLY_EVIDENCE = ExclusionEvidence(kind=ExclusionEvidenceKind.NOTE, message=CROSSCODE_EVAL_REASON)


def _note(message: str) -> ExclusionEvidence:
    return ExclusionEvidence(kind=ExclusionEvidenceKind.NOTE, message=message)


MANUAL_EXCLUSION_CLAIMS: dict[str, tuple[ExclusionClaim, ...]] = {
    "TaskTrove:swesmith-verifier-repaired-v5": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Machine execution requires "
                "explicit semantics and native dependencies. The dev-only backends (docker, docker-gvisor) never "
                "count for admission, and the declared TaskSpec population (7016 tasks) stays intact pending a "
                "pinned rebuild. Surfaced by the 908 grading-metadata adoption (grading-metadata-proof-v5)."
            ),
        ),
    ),
    "TaskTrove:DCAgent__exp_rle_adversarial-v6": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (2726 tasks) stays intact "
                "pending a pinned rebuild."
            ),
        ),
    ),
    "TaskTrove:DCAgent__exp_rpt_nemotron-cpp": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (4196 tasks) stays intact "
                "pending a pinned rebuild."
            ),
        ),
    ),
    "TaskTrove:DCAgent__inferredbugs-sandboxes-verifier": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (9659 tasks) stays intact "
                "pending a pinned rebuild."
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
    "TaskTrove:laion__exp_rpt_ghactions-v3": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (9930 tasks) stays intact "
                "pending a pinned rebuild."
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
    "TaskTrove:laion__exp_rpt_scaffold-v3": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (3121 tasks) stays intact "
                "pending a pinned rebuild."
            ),
        ),
    ),
    "TaskTrove:laion__exp_rpt_stack-cpp-v4": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (7878 tasks) stays intact "
                "pending a pinned rebuild."
            ),
        ),
    ),
    "TaskTrove:laion__exp_rpt_stack-dockerfile-gpt5mini-v7": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (587 tasks) stays intact pending "
                "a pinned rebuild."
            ),
        ),
    ),
    "TaskTrove:laion__exp_rpt_stack-go-v5": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (2275 tasks) stays intact "
                "pending a pinned rebuild."
            ),
        ),
    ),
    "TaskTrove:laion__exp_rpt_stack-jest-v5": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (424 tasks) stays intact pending "
                "a pinned rebuild."
            ),
        ),
    ),
    "TaskTrove:laion__exp_rpt_stack-junit-v6": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (843 tasks) stays intact pending "
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
    "TaskTrove:laion__exp_rpt_stack-pytest-large-v3": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (1782 tasks) stays intact "
                "pending a pinned rebuild."
            ),
        ),
    ),
    "TaskTrove:laion__exp_rpt_stack-rspec-v4": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (8860 tasks) stays intact "
                "pending a pinned rebuild."
            ),
        ),
    ),
    "TaskTrove:laion__mix_h10_reward_proportional-v2": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (2858 tasks) stays intact "
                "pending a pinned rebuild."
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
    "TaskTrove:laion__nemotron-gym-agentic-function-calling-pivot-v3": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (9579 tasks) stays intact "
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
    "TaskTrove:laion__nemotron-gym-instruction-following-citation-v2": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (9033 tasks) stays intact "
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
    "TaskTrove:laion__nemotron-gym-litmus-bench-v2": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (5232 tasks) stays intact "
                "pending a pinned rebuild."
            ),
        ),
    ),
    "TaskTrove:laion__nemotron-gym-math-advanced-calculations-v4": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (5291 tasks) stays intact "
                "pending a pinned rebuild."
            ),
        ),
    ),
    "TaskTrove:laion__nemotron-gym-multichallenge-vanilla-v3": (
        ExclusionClaim(
            reason=ExclusionReason.NO_SUPPORTED_BACKEND,
            evidence=_note(
                "Excluded per the live-qualification ruling (2026-11-17): the fresh regional BackendSurvey "
                "(4ef1967b12, backend-map-v2 in the r22 preview rebuild) reports every production backend "
                "(daytona/iris/local) UNSUPPORTED with measured per-role cause: agent: Unresolved Docker build "
                "context must be built and pinned before execution. The dev-only backends (docker, docker-gvisor) "
                "never count for admission, and the declared TaskSpec population (1050 tasks) stays intact "
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
    "TaskTrove:laion__swesmith-oracle-filtered-v2": (
        ExclusionClaim(
            reason=ExclusionReason.SUPERSEDED_BY_REPLACEMENT,
            evidence=_note(
                "Deprecated: this upstream SWE-smith conversion predates the hardened "
                "bug-zero/reference-one admission and retains unresolved Docker contexts. "
                "The long-loop repair campaign found scratch, generated-metadata and verifier "
                "infrastructure defects in the derived Smartie cohort. Use its pinned repaired "
                "release with exact per-task control receipts."
            ),
        ),
    ),
    "TaskTrove:laion__exp_rpt_codenet-python-v4": (
        ExclusionClaim(
            reason=ExclusionReason.GRADER_DEFECT,
            evidence=_note(
                "Only 3 hidden cases and whitespace-collapsing compare. Oracle present; "
                "regenerate 20+ cases per task."
            ),
        ),
    ),
    "TaskTrove:laion__exp_rpt_crosscodeeval-csharp-v4": (
        ExclusionClaim(reason=ExclusionReason.EVAL_ONLY_BENCHMARK, evidence=_EVAL_ONLY_EVIDENCE),
        ExclusionClaim(
            reason=ExclusionReason.GRADER_DEFECT,
            evidence=_note("0.25 reward for any identifier-shaped output; instruction coaches the hack."),
        ),
    ),
    "TaskTrove:laion__exp_rpt_crosscodeeval-java-v3": (
        ExclusionClaim(reason=ExclusionReason.EVAL_ONLY_BENCHMARK, evidence=_EVAL_ONLY_EVIDENCE),
        ExclusionClaim(
            reason=ExclusionReason.GRADER_DEFECT,
            evidence=_note("Exact string match on a single line completion. Not agentic, no execution."),
        ),
    ),
    "TaskTrove:laion__exp_rpt_crosscodeeval-typescript-v2": (
        ExclusionClaim(reason=ExclusionReason.EVAL_ONLY_BENCHMARK, evidence=_EVAL_ONLY_EVIDENCE),
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
    "TaskTrove:smartie-swesmith-admitted": (
        ExclusionClaim(
            reason=ExclusionReason.SUPERSEDED_BY_REPLACEMENT,
            evidence=_note(
                "Deprecated: the original 8,045-task admission did not enforce the hardened "
                "bug-zero/reference-one controls. Re-control excluded 2,417 tasks, including "
                "scratch permissions, generated metadata, verifier infrastructure failures, "
                "timeouts, degenerate references and passing bug commits. Use the separately "
                "pinned repaired cohort; its v4-admitted task packages retain their tested bytes."
            ),
        ),
    ),
    "TaskTrove:smartie-calibforge-shared-v2": (
        ExclusionClaim(
            reason=ExclusionReason.SNAPSHOT_BUDGET_EXCEEDED,
            evidence=_note(
                "Deprecated pending the snapshot-safe CalibForge rebuild: the campaign's "
                "provisioning records (snapshot-capacity-check) report the cohort needing 23 "
                "image/resource snapshot variants (19 on the retained training slice), over the "
                "ten-snapshot budget."
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
    "TaskTrove:laion__codeelo-v2": (
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
    "TaskTrove:laion__nemotron-gym-instruction-following-freeform-v2": (
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
