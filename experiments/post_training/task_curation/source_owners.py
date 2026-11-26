# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The owning lane for every unresolved source, rendered beside its status reasons.

The atlas-2026.11.27 release renders an owner beside every unresolved Pending or
Excluded reason (coordinator directive 2026-11-26): a burndown list without an owner
is not actionable. Each entry names the lane and its session; a row whose blocker is
closed by ruling carries that ruling instead of a lane. Maintain this registry when a
row changes hands - `require_typed_rows` fails the status-extras build when an
unresolved row lacks its owner, so a stale registry cannot ship silently.
"""

from dataclasses import dataclass

from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id


@dataclass(frozen=True)
class SourceOwner:
    """One unresolved source's owning lane and its current work item.

    ``active`` is the release-time driver check (user rule 2026-11-26 15:20 EST):
    a Pending row whose owner shows no live jobs or landings driving it since the
    previous release fails the status-extras build. Maintain the flag beside the
    note that carries the activity evidence.
    """

    owner: str
    note: str
    active: bool = True


SOURCE_OWNERS: dict[str, SourceOwner] = {
    "ASDiv:asdiv": SourceOwner(
        owner="no lane — license-ban by ruling #11497; lifts only by user ruling",
        note="excluded",
    ),
    "AllenAI:Dolci-Think-SFT-Clean-v2/precise-if": SourceOwner(
        owner="ingest B · Codex 01a1e039",
        note="admitted R41 intake; surfaces at the load",
    ),
    "AllenAI:rlvr_math": SourceOwner(
        owner="no lane — license-ban by ruling #11497; lifts only by user ruling",
        note="excluded",
    ),
    "Anthropic:hh_rlhf/harmless-base": SourceOwner(
        owner="ingest C · ses_de69f162affeNDxM7uXXruALFm",
        note="decided 2026-11-26 — not a valid RL source as declared: grader-missing"
        " (NoGrader preference pairs; MIT license PASS)",
    ),
    "Anthropic:hh_rlhf/helpful-base": SourceOwner(
        owner="ingest C · ses_de69f162affeNDxM7uXXruALFm",
        note="decided 2026-11-26 — not a valid RL source as declared: grader-missing"
        " (NoGrader preference pairs; MIT license PASS)",
    ),
    "Anthropic:hh_rlhf/helpful-online": SourceOwner(
        owner="ingest C · ses_de69f162affeNDxM7uXXruALFm",
        note="decided 2026-11-26 — not a valid RL source as declared: grader-missing"
        " (NoGrader preference pairs; MIT license PASS)",
    ),
    "Anthropic:hh_rlhf/helpful-rejection-sampled": SourceOwner(
        owner="ingest C · ses_de69f162affeNDxM7uXXruALFm",
        note="decided 2026-11-26 — not a valid RL source as declared: grader-missing"
        " (NoGrader preference pairs; MIT license PASS)",
    ),
    "CyBench:andyzorigin-cybench": SourceOwner(
        owner="by design — ban registry / rulings; nothing to burn down",
        note="metadata exclusion",
    ),
    "Evalchemy:aime24": SourceOwner(
        owner="no lane — license-ban by ruling #11497; lifts only by user ruling",
        note="excluded",
    ),
    "Evalchemy:aime25": SourceOwner(
        owner="no lane — license-ban by ruling #11497; lifts only by user ruling",
        note="excluded",
    ),
    "Evalchemy:amc23": SourceOwner(
        owner="no lane — license-ban by ruling #11497; lifts only by user ruling",
        note="excluded",
    ),
    "Evalchemy:boolq": SourceOwner(
        owner="no lane — license-ban by ruling #11497; lifts only by user ruling",
        note="excluded",
    ),
    "Evalchemy:financebench": SourceOwner(
        owner="no lane — license-ban by ruling #11497; lifts only by user ruling",
        note="excluded",
    ),
    "Evalchemy:hellaswag": SourceOwner(
        owner="no lane — license-ban by ruling #11497; lifts only by user ruling",
        note="excluded",
    ),
    "Evalchemy:math500": SourceOwner(
        owner="no lane — license-ban by ruling #11497; lifts only by user ruling",
        note="excluded",
    ),
    "Evalchemy:olympiadbench": SourceOwner(
        owner="no lane — license-ban by ruling #11497; lifts only by user ruling",
        note="excluded",
    ),
    "Evalchemy:olympiadbenchdeterministic": SourceOwner(
        owner="no lane — license-ban by ruling #11497; lifts only by user ruling",
        note="excluded",
    ),
    "Evalchemy:openbookqa": SourceOwner(
        owner="no lane — license-ban by ruling #11497; lifts only by user ruling",
        note="excluded",
    ),
    "Evalchemy:piqa": SourceOwner(
        owner="no lane — license-ban by ruling #11497; lifts only by user ruling",
        note="excluded",
    ),
    "Evalchemy:triviaqa": SourceOwner(
        owner="no lane — license-ban by ruling #11497; lifts only by user ruling",
        note="excluded",
    ),
    "Evalchemy:winogrande": SourceOwner(
        owner="no lane — license-ban by ruling #11497; lifts only by user ruling",
        note="excluded",
    ),
    "FACET-Terminal:FACET-Terminal/FACET-Terminal-Tasks-6k": SourceOwner(
        owner="facet-endless-admission · ses_df1063379ffel",
        note="intake",
    ),
    "FWeindel:validated-tasks/endless": SourceOwner(
        owner="facet-endless-admission · ses_df1063379ffel",
        note="intake",
    ),
    "FWeindel:validated-tasks/facet_v2": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "FWeindel:validated-tasks/facet_v3": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "FWeindel:validated-tasks/fineenvs_v2": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "FWeindel:validated-tasks/fineenvs_v3": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "FWeindel:validated-tasks/mimo_v1": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "FWeindel:validated-tasks/pymethods2test_v1": SourceOwner(
        owner="ingest A · ses_de826fccdffe",
        note="intake; controls/chain running",
    ),
    "GPUMODE:kernelbot-data": SourceOwner(
        owner="by design — ban registry / rulings; nothing to burn down",
        note="metadata exclusion",
    ),
    "HAL:princeton-pli-hal-harness": SourceOwner(
        owner="by design — ban registry / rulings; nothing to burn down",
        note="metadata exclusion",
    ),
    "Harbor Hub:skill2env/skill2env": SourceOwner(
        owner="no lane — license-ban by ruling #11497; lifts only by user ruling",
        note="excluded",
    ),
    "HuggingFaceH4:aime24": SourceOwner(
        owner="no lane — license-ban by ruling #11497; lifts only by user ruling",
        note="excluded",
    ),
    "HuggingFaceH4:math500": SourceOwner(
        owner="no lane — license-ban by ruling #11497; lifts only by user ruling",
        note="excluded",
    ),
    "Lego-X:Terminal-Lego-15k": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "Lite-Coder:LiteCoder-Terminal-RL-preview": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "MAI-Gym:tasks": SourceOwner(
        owner="by design — ban registry / rulings; nothing to burn down",
        note="metadata exclusion",
    ),
    "Marin:forge-main-20261117": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "Marin:thoughtcrime": SourceOwner(
        owner="atlas-pending-burndown · ses_de7b345a3ffe",
        note="grading census re-measured 328/328; readmits at the next load",
    ),
    "MiroVerse:miromind-ai-MiroVerse-v0.1": SourceOwner(
        owner="by design — ban registry / rulings; nothing to burn down",
        note="metadata exclusion",
    ),
    "Nemotron:Nemotron-Cascade-2-RL-data/IF-RL": SourceOwner(
        owner="ingest B · Codex 01a1e039",
        note="intake queued; parent admitted",
    ),
    "Nemotron:Nemotron-Cascade-2-RL-data/SWE-RL": SourceOwner(
        owner="ingest B · Codex 01a1e039",
        note="intake queued; parent admitted",
    ),
    "Nemotron:Nemotron-RL-instruction_following-structured_outputs": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "Nemotron:Nemotron-RL-knowledge-mcqa": SourceOwner(
        owner="ingest B · Codex 01a1e039",
        note="admitted R41 intake; surfaces at the load",
    ),
    "Nemotron:Nemotron-RL-knowledge-web_search-mcqa": SourceOwner(
        owner="by design — ban registry / rulings; nothing to burn down",
        note="metadata exclusion",
    ),
    "Nemotron:nemotron_ultra_mopd/hs3_en": SourceOwner(
        owner="ingest C · ses_de69f162affeNDxM7uXXruALFm",
        note="excluded 2026-11-26 — grader-missing (100% NoGrader; cc-by-4.0 license PASS)",
    ),
    "Nemotron:nemotron_ultra_mopd/hs3_multi": SourceOwner(
        owner="ingest C · ses_de69f162affeNDxM7uXXruALFm",
        note="excluded 2026-11-26 — grader-missing (962/962 NoGrader; cc-by-4.0 license PASS)",
    ),
    "Nemotron:nemotron_ultra_mopd/hs3_multiturn": SourceOwner(
        owner="ingest C · ses_de69f162affeNDxM7uXXruALFm",
        note="excluded 2026-11-26 — grader-missing (1,243/1,243 NoGrader; cc-by-4.0 license PASS)",
    ),
    "Nemotron:nemotron_ultra_mopd/makeshn_ultra_v3_ipi_train": SourceOwner(
        owner="ingest C · ses_de69f162affeNDxM7uXXruALFm",
        note="excluded 2026-11-26 — grader-missing (2,000/2,000 NoGrader;"
        " agent: no declared environment; cc-by-4.0 PASS)",
    ),
    "Nemotron:nemotron_ultra_mopd/safety_en": SourceOwner(
        owner="ingest C · ses_de69f162affeNDxM7uXXruALFm",
        note="excluded 2026-11-26 — grader-missing (629/629 NoGrader; cc-by-4.0 license PASS)",
    ),
    "Nemotron:nemotron_ultra_mopd/swe_pivot_len40k/SWE-Gym/SWE-Gym": SourceOwner(
        owner="ingest C · ses_de69f162affeNDxM7uXXruALFm",
        note="excluded 2026-11-26 — grader-missing (860/860 NoGrader; cc-by-4.0 license PASS)",
    ),
    "Nemotron:nemotron_ultra_mopd/swe_pivot_len40k/nebius/SWE-rebench-V2": SourceOwner(
        owner="ingest C · ses_de69f162affeNDxM7uXXruALFm",
        note="excluded 2026-11-26 — grader-missing (3,903/3,903 NoGrader; cc-by-4.0 license PASS)",
    ),
    "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_lean": SourceOwner(
        owner="ingest C · ses_de69f162affeNDxM7uXXruALFm",
        note="intake; wave 1 — license gate then conversion chain",
    ),
    "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_math_tir": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_rdkit": SourceOwner(
        owner="ingest C · ses_de69f162affeNDxM7uXXruALFm",
        note="intake; wave 1 — license gate then conversion chain",
    ),
    "Nemotron:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/SWE-Gym/SWE-Gym": SourceOwner(
        owner="ingest C · ses_de69f162affeNDxM7uXXruALFm",
        note="excluded 2026-11-26 — grader-missing (246/246 NoGrader; cc-by-4.0 license PASS)",
    ),
    "Nemotron:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/nebius/SWE-rebench-V2": SourceOwner(
        owner="ingest C · ses_de69f162affeNDxM7uXXruALFm",
        note="excluded 2026-11-26 — grader-missing (932/932 NoGrader; cc-by-4.0 license PASS)",
    ),
    "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_lean": SourceOwner(
        owner="ingest C · ses_de69f162affeNDxM7uXXruALFm",
        note="intake; wave 2 — same conversion path as the mopd pair",
    ),
    "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_math_tir": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_lean": SourceOwner(
        owner="ingest C · ses_de69f162affeNDxM7uXXruALFm",
        note="intake; wave 2 — same conversion path as the mopd pair",
    ),
    "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_math_tir": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_rdkit": SourceOwner(
        owner="ingest C · ses_de69f162affeNDxM7uXXruALFm",
        note="intake; wave 2 — same conversion path as the mopd pair",
    ),
    "Nvidia:Nemotron-RL-InverseIFEval-v1": SourceOwner(
        owner="ingest B · Codex 01a1e039",
        note="admitted R41 intake; surfaces at the load",
    ),
    "Nvidia:Nemotron-RL-QA-Abstention-v1": SourceOwner(
        owner="ingest B · OpenCode ses_de6edbb82ffeg8kF9ytv2hToj2",
        note="admitted R41 intake; surfaces at the load",
    ),
    "Yatin:Edge-Agent-Reasoning-WebSearch-260K": SourceOwner(
        owner="ingest B · OpenCode ses_de6edbb82ffeg8kF9ytv2hToj2",
        note="admitted R41 intake; surfaces at the load",
    ),
    "ScienceOne:S1-DeepResearch-15k": SourceOwner(
        owner="ingest B · OpenCode ses_de6edbb82ffeg8kF9ytv2hToj2",
        note="admitted R41 intake; surfaces at the load",
    ),
    "Nvidia:Nemotron-RL-Instruction-Following-Free-Form-Formatting-v1": SourceOwner(
        owner="ingest B · OpenCode ses_de6edbb82ffeg8kF9ytv2hToj2",
        note="admitted R41 intake; surfaces at the load",
    ),
    "Nvidia:Nemotron-RL-Instruction-Following-Citation-Formatting-v1": SourceOwner(
        owner="ingest B · OpenCode ses_de6edbb82ffeg8kF9ytv2hToj2",
        note="admitted R41 intake; surfaces at the load",
    ),
    "Nvidia:Nemotron-RL-agent-workplace_assistant": SourceOwner(
        owner="ingest B · Codex 01a1e039",
        note="admitted R41 intake; surfaces at the load",
    ),
    "Nvidia:Nemotron-System-User-Instruction-Hierarchy-v1": SourceOwner(
        owner="ingest B · Codex 01a1e039",
        note="admitted R41 intake; surfaces at the load",
    ),
    "Open-R1:verifiable_code": SourceOwner(
        owner="no lane — license-ban by ruling #11497; lifts only by user ruling",
        note="excluded",
    ),
    "OpenGCUA:trajectories": SourceOwner(
        owner="by design — ban registry / rulings; nothing to burn down",
        note="metadata exclusion",
    ),
    "OpenResearcher:OpenResearcher-Dataset": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "PrimeIntellect:PrimeIntellect/Multi-SWE-RL-Verified": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "PrimeIntellect:PrimeIntellect/SWE-Lego-Real-Data-Verified": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "PrimeIntellect:PrimeIntellect/Scale-SWE-Verified": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "RST:laion/rts-1k-rootless-apptainer": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "SWE-rebench:ibragim-bad/SWE-rebench-V2.1@repaired-20261119": SourceOwner(
        owner="swe-repair · ses_e15d1c017ffe",
        note="excluded",
    ),
    "TRL:kto_mix/argilla/distilabel-capybara-dpo-7k-binarized": SourceOwner(
        owner="by design — ban registry / rulings; nothing to burn down",
        note="metadata exclusion",
    ),
    "TRL:kto_mix/argilla/distilabel-intel-orca-dpo-pairs": SourceOwner(
        owner="by design — ban registry / rulings; nothing to burn down",
        note="metadata exclusion",
    ),
    "TRL:kto_mix/argilla/ultrafeedback-binarized-preferences-cleaned": SourceOwner(
        owner="by design — ban registry / rulings; nothing to burn down",
        note="metadata exclusion",
    ),
    "TaskTrove:DCAgent2__nl2bash-tasks-cleaned-oracle-v2@repaired-20261118": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "TaskTrove:DCAgent__code-contests-noblock": SourceOwner(
        owner="atlas-pending-burndown · ses_de7b345a3ffe",
        note="converter-v4 republish + SAMPLE/FULL QA (coordinator-assigned)",
    ),
    "TaskTrove:DCAgent__exp_rle_adversarial-v6": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:DCAgent__exp_rpt_nemotron-cpp": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:DCAgent__exp_rpt_pymethods2test-large-v2@backfilled-20261122": SourceOwner(
        owner="requirement-backfill · awaits the next preview publication",
        note="excluded",
    ),
    "TaskTrove:DCAgent__exp_rpt_pymethods2test-large-v2@repaired-20261118": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "TaskTrove:DCAgent__exp_rpt_unitsyn-python-large-v2@backfilled-20261122": SourceOwner(
        owner="requirement-backfill · awaits the next preview publication",
        note="excluded",
    ),
    "TaskTrove:DCAgent__inferredbugs-sandboxes-verifier": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:DCAgent__mix_h4_binary_easy": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:DCAgent__selfinstruct-naive-sandboxes-2-verified-v3": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:DCAgent__swe_rebench_v2_patched_oracle-v2": SourceOwner(
        owner="swe-repair · ses_e15d1c017ffe",
        note="superseded by V2.1@repaired per ruling 2026-11-26; Deprecated-with-link once it admits",
    ),
    "TaskTrove:GAIR__OpenSWE__openswe_oss": SourceOwner(
        owner="SWE zero-task · ses_de7ba44ecffe",
        note="conversions sealed; acceptance/census/lexical/SAMPLE chain running",
    ),
    "TaskTrove:GAIR__OpenSWE__openswe_other": SourceOwner(
        owner="no lane — license-ban by ruling #11497; lifts only by user ruling",
        note="excluded",
    ),
    "TaskTrove:R2E-Gym__R2E-Gym-V1": SourceOwner(
        owner="SWE zero-task · ses_de7ba44ecffe",
        note="conversions sealed; acceptance/census/lexical/SAMPLE chain running",
    ),
    "TaskTrove:SWE-Gym__SWE-Gym": SourceOwner(
        owner="SWE zero-task · ses_de7ba44ecffe",
        note="conversions sealed; acceptance/census/lexical/SAMPLE chain running",
    ),
    "TaskTrove:SankalpKJ__nemotron-code-oracle-filtered": SourceOwner(
        owner="atlas-pending-burndown · ses_de7b345a3ffe",
        note="zero-task converting (basepin wave 3)",
    ),
    "TaskTrove:XiaomiMiMo__MiMo-V2.6-RL-oss__general": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:XiaomiMiMo__MiMo-V2.6-RL-oss__music": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__codeelo-v2": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__exp_rpt_bugsinpy-v4": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__exp_rpt_codenet-python-v4": SourceOwner(
        owner="atlas-pending-burndown · ses_de7b345a3ffe",
        note="zero-task converting (basepin wave 3)",
    ),
    "TaskTrove:laion__exp_rpt_crosscodeeval-csharp-v4": SourceOwner(
        owner="atlas-pending-burndown · ses_de7b345a3ffe",
        note="zero-task converting (basepin wave 3)",
    ),
    "TaskTrove:laion__exp_rpt_crosscodeeval-java-v3": SourceOwner(
        owner="atlas-pending-burndown · ses_de7b345a3ffe",
        note="zero-task converting (basepin wave 3)",
    ),
    "TaskTrove:laion__exp_rpt_crosscodeeval-typescript-v2": SourceOwner(
        owner="atlas-pending-burndown · ses_de7b345a3ffe",
        note="zero-task converting (basepin wave 3)",
    ),
    "TaskTrove:laion__exp_rpt_ghactions-v3": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__exp_rpt_methods2test-large-v4": SourceOwner(
        owner="atlas-pending-burndown · ses_de7b345a3ffe",
        note="zero-task converting (basepin wave 3)",
    ),
    "TaskTrove:laion__exp_rpt_nemotron-junit-v6": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__exp_rpt_scaffold-v3": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__exp_rpt_stack-cpp-v4": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__exp_rpt_stack-dockerfile-gpt5mini-v7": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__exp_rpt_stack-go-v5": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__exp_rpt_stack-jest-v5": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__exp_rpt_stack-junit-v6": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__exp_rpt_stack-php-large-v9": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__exp_rpt_stack-pytest-large-v3": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__exp_rpt_stack-rspec-v4": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__glaive-code-assistant-sandboxes-verified-v2": SourceOwner(
        owner="ingest B · Codex 01a1e039",
        note="parked for the coordinator-directed rotation; needs native-judge runtime qualification",
    ),
    "TaskTrove:laion__magicoder-v4": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__mix_h10_reward_proportional-v2": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__mix_h11_single_skill_only-v2": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__mix_h8_original_tests-v2": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__nemotron-gym-agent-calendar-v2": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "TaskTrove:laion__nemotron-gym-agentic-function-calling-pivot-v3": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__nemotron-gym-agentic-indirect-prompt-injection-v3": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__nemotron-gym-agentic-swe-pivot-v4": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__nemotron-gym-arc-agi-python-inductive-v2": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "TaskTrove:laion__nemotron-gym-arc-agi-transductive-v3": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "TaskTrove:laion__nemotron-gym-cfbench-v4": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__nemotron-gym-identity-following-v4": SourceOwner(
        owner="identity-following-repair · repair sealed, registers at the R41 load",
        note="excluded",
    ),
    "TaskTrove:laion__nemotron-gym-identity-following-v4@repaired-20261123": SourceOwner(
        owner="identity-following-repair · repair sealed, registers at the R41 load",
        note="Measurements pending",
    ),
    "TaskTrove:laion__nemotron-gym-instruction-following-adversarial-v5": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__nemotron-gym-instruction-following-calendar-v3": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "TaskTrove:laion__nemotron-gym-instruction-following-citation-v2": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__nemotron-gym-instruction-following-freeform-v2": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__nemotron-gym-instruction-following-multiturnchat-v4": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__nemotron-gym-inverse-ifeval-v4": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__nemotron-gym-knowledge-openqa-v4": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "TaskTrove:laion__nemotron-gym-knowledge-web-search-mcqa-v2": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__nemotron-gym-litmus-bench-v2": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__nemotron-gym-math-advanced-calculations-v4": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__nemotron-gym-math-openmathreasoning-v2": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "TaskTrove:laion__nemotron-gym-math-stack-overflow-v3": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "TaskTrove:laion__nemotron-gym-multichallenge-advanced-v4": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "TaskTrove:laion__nemotron-gym-multichallenge-vanilla-v3": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__nemotron-gym-qa-abstention-v4": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__nemotron-gym-reasoning-gym-v2": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "TaskTrove:laion__nemotron-gym-safety-v3": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "TaskTrove:laion__nemotron-gym-science-so-openq-v3": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "TaskTrove:laion__nemotron-gym-structured-outputs-v4": SourceOwner(
        owner="ingest B · Codex 01a1e039",
        note="admitted R41 intake; surfaces at the load",
    ),
    "TaskTrove:laion__nemotron-gym-sysbench-v4": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__openswe-tasks-patched-v7-oracle-success": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:laion__r2egym-patched-full-oracle-v3": SourceOwner(
        owner="SWE zero-task · ses_de7ba44ecffe",
        note="conversions sealed; acceptance/census/lexical/SAMPLE chain running",
    ),
    "TaskTrove:laion__stackexchange-codereview-sandboxes-verified-v2": SourceOwner(
        owner="ingest B · Codex 01a1e039",
        note="Formatting group: lexical + SAMPLE QC",
    ),
    "TaskTrove:laion__stackexchange-overflow-sandboxes-verified-v2": SourceOwner(
        owner="ingest B · Codex 01a1e039",
        note="Formatting group: lexical + SAMPLE QC",
    ),
    "TaskTrove:laion__stackexchange-superuser-sandboxes-verified-v2": SourceOwner(
        owner="ingest B · Codex 01a1e039",
        note="Formatting group: lexical + SAMPLE QC",
    ),
    "TaskTrove:laion__stackexchange-tezos-sandboxes-verified-v2": SourceOwner(
        owner="ingest B · Codex 01a1e039",
        note="parked for the coordinator-directed rotation; needs native-judge runtime qualification",
    ),
    "TaskTrove:laion__stackexchange-unix-sandboxes-verified-v2": SourceOwner(
        owner="ingest B · Codex 01a1e039",
        note="Formatting group: lexical + SAMPLE QC",
    ),
    "TaskTrove:laion__swegym-tasks-patched-validated-v5": SourceOwner(
        owner="SWE zero-task · ses_de7ba44ecffe",
        note="conversions sealed; acceptance/census/lexical/SAMPLE chain running",
    ),
    "TaskTrove:laion__toolscale-v4": SourceOwner(
        owner="atlas-pending-burndown · ses_de7b345a3ffe",
        note="zero-task converting (basepin wave 3)",
    ),
    "TaskTrove:laion__tulu3-sft-personas-math-sandboxes-verified-v3": SourceOwner(
        owner="atlas-pending-burndown · ses_de7b345a3ffe",
        note="zero-task converting (basepin wave 3)",
    ),
    "TaskTrove:laion__wizardlm-orca-v4": SourceOwner(
        owner="ingest B · Codex 01a1e039",
        note="Formatting group: lexical + SAMPLE QC",
    ),
    "TaskTrove:smartie-calibforge-rebuilt-v1": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "TaskTrove:swesmith-verifier-repaired-v5": SourceOwner(
        owner="TaskSpec re-admission · ses_df2a1f590ffe",
        note="excluded — SAMPLE QC v4 running",
    ),
    "TaskTrove:tasktrove-code-contests-built-image": SourceOwner(
        owner="atlas-pending-burndown · ses_de7b345a3ffe",
        note="converter-v4 republish + QA before admission",
    ),
    "TaskTrove:tasktrove-codeforces-rebuilt-v1": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "TaskTrove:tasktrove-competitive-coding-rebuilt-v1": SourceOwner(
        owner="ingest B · Codex 01a1e039",
        note="admitted R41 intake; surfaces at the load",
    ),
    "TaskTrove:tasktrove-curriculum-easy-rebuilt-v1": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "TaskTrove:tasktrove-curriculum-medium-rebuilt-v1": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "TaskTrove:tasktrove-e2egit-large-rebuilt-v1": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "TaskTrove:tasktrove-e2egit-v2-rebuilt": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "TaskTrove:tasktrove-multifile-rebuilt-v1": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "TaskTrove:tasktrove-nl2bash-rebuilt-v1": SourceOwner(
        owner="facet-endless-admission · ses_df1063379ffel",
        note="intake",
    ),
    "TaskTrove:tasktrove-pymethods2test-large-rebuilt-v1": SourceOwner(
        owner="ingest A · ses_de826fccdffe",
        note="intake; controls/chain running",
    ),
    "TaskTrove:tasktrove-pymethods2test-rebuilt-v1": SourceOwner(
        owner="ingest A · ses_de826fccdffe",
        note="intake; controls/chain running",
    ),
    "TaskTrove:tasktrove-stack-pytest-rebuilt-v1": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "TaskTrove:tasktrove-taco-built-image": SourceOwner(
        owner="atlas-pending-burndown · ses_de7b345a3ffe",
        note="converter-v4 republish + QA before admission",
    ),
    "TaskTrove:tasktrove-unitsyn-python-large-rebuilt-v1": SourceOwner(
        owner="ingest A · ses_de826fccdffe",
        note="intake; controls/chain running",
    ),
    "TaskTrove:tasktrove-unitsyn-python-rebuilt-v1": SourceOwner(
        owner="ingest A · ses_de826fccdffe",
        note="491-row small-cohort republish first",
    ),
    "TerminalWorld:TerminalWorld": SourceOwner(
        owner="R41 self-executing · ses_deccb722",
        note="release readmission from sealed receipts / disk_root / host-image routes",
    ),
    "marcusm117:ProofNet-Verified": SourceOwner(
        owner="atlas-source-coverage · RLVR declarations registry",
        note="admitted at the R41 cut; surfaces at the load",
    ),
    "pafitis:hardmath": SourceOwner(
        owner="no lane — license-ban by ruling #11497; lifts only by user ruling",
        note="excluded",
    ),
    "pafitis:hardmath@rhs": SourceOwner(
        owner="ingest A · ses_de826fccdffe (+ atlas-grader-reseal for the grader)",
        note="P0 republish of the repaired variant; grader reseal in flight",
    ),
    "project-numina:NuminaMath-LEAN": SourceOwner(
        owner="ingest C · ses_de69f162affeNDxM7uXXruALFm",
        note="intake; wave 1 — license gate then conversion chain",
    ),
}


def source_owner(source_id: str) -> SourceOwner | None:
    """The registry entry for one source, canonicalized; None outside the registry."""
    return SOURCE_OWNERS.get(canonical_source_id(source_id))


def owner_labels() -> dict[str, str]:
    """The rendered owner label per source, for the status page's owners overlay."""
    return {source_id: entry.owner for source_id, entry in SOURCE_OWNERS.items()}
