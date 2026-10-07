# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned RL source declarations and Atlas reporting metadata."""

import json
from dataclasses import replace
from pathlib import Path

import experiments.post_training.task_curation.datasets.nemotron_ultra.mopd.abstention as nemotron_ultra_mopd_abstention
import experiments.post_training.task_curation.datasets.nemotron_ultra.mopd.hs3_en as nemotron_ultra_mopd_hs3_en
import experiments.post_training.task_curation.datasets.nemotron_ultra.mopd.hs3_multi as nemotron_ultra_mopd_hs3_multi
import experiments.post_training.task_curation.datasets.nemotron_ultra.mopd.jailbreak as nemotron_ultra_mopd_jailbreak
import experiments.post_training.task_curation.datasets.nemotron_ultra.mopd.lean as nemotron_ultra_mopd_lean
import experiments.post_training.task_curation.datasets.nemotron_ultra.mopd.math_cot as nemotron_ultra_mopd_math_cot
import experiments.post_training.task_curation.datasets.nemotron_ultra.mopd.math_tir as nemotron_ultra_mopd_math_tir
import experiments.post_training.task_curation.datasets.nemotron_ultra.mopd.rdkit as nemotron_ultra_mopd_rdkit
import experiments.post_training.task_curation.datasets.nemotron_ultra.mopd.safety_en as nemotron_ultra_mopd_safety_en
import experiments.post_training.task_curation.datasets.nemotron_ultra.mopd.stem_mcqa as nemotron_ultra_mopd_stem_mcqa
import experiments.post_training.task_curation.datasets.nemotron_ultra.mopd.swe_gym as nemotron_ultra_mopd_swe_gym
import experiments.post_training.task_curation.datasets.nemotron_ultra.mopd.swe_nebius as nemotron_ultra_mopd_swe_nebius
import experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr1.jailbreak as nemotron_ultra_rlvr1_jailbreak
import experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr1.lean as nemotron_ultra_rlvr1_lean
import experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr1.math_cot as nemotron_ultra_rlvr1_math_cot
import experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr1.math_tir as nemotron_ultra_rlvr1_math_tir
import experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr1.stem_mcqa as nemotron_ultra_rlvr1_stem_mcqa
import experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr1.swe_gym as nemotron_ultra_rlvr1_swe_gym
import experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr1.tau_pivot as nemotron_ultra_rlvr1_tau_pivot
import experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr2.jailbreak as nemotron_ultra_rlvr2_jailbreak
import experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr2.lean as nemotron_ultra_rlvr2_lean
import experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr2.math_cot as nemotron_ultra_rlvr2_math_cot
import experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr2.math_tir as nemotron_ultra_rlvr2_math_tir
import experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr2.rdkit as nemotron_ultra_rlvr2_rdkit
import experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr2.stem_mcqa as nemotron_ultra_rlvr2_stem_mcqa
import experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr2.swe_gym as nemotron_ultra_rlvr2_swe_gym
import experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr2.tau_pivot as nemotron_ultra_rlvr2_tau_pivot
import experiments.post_training.task_curation.datasets.skyrl.aime24 as skyrl_aime24
import experiments.post_training.task_curation.datasets.skyrl.aime_1983_2024 as skyrl_aime_1983_2024
import experiments.post_training.task_curation.datasets.skyrl.apps as skyrl_apps
import experiments.post_training.task_curation.datasets.skyrl.asdiv as skyrl_asdiv
import experiments.post_training.task_curation.datasets.skyrl.dapo_math as skyrl_dapo_math
import experiments.post_training.task_curation.datasets.skyrl.deepscaler as skyrl_deepscaler
import experiments.post_training.task_curation.datasets.skyrl.eurus2_code as skyrl_eurus2_code
import experiments.post_training.task_curation.datasets.skyrl.gpqa as skyrl_gpqa
import experiments.post_training.task_curation.datasets.skyrl.gretel_text_to_sql as skyrl_gretel_text_to_sql
import experiments.post_training.task_curation.datasets.skyrl.gsm8k as skyrl_gsm8k
import experiments.post_training.task_curation.datasets.skyrl.hardmath as skyrl_hardmath
import experiments.post_training.task_curation.datasets.skyrl.hendrycks_math as skyrl_hendrycks_math
import experiments.post_training.task_curation.datasets.skyrl.hh_rlhf.harmless_base as skyrl_hh_rlhf_harmless_base
import experiments.post_training.task_curation.datasets.skyrl.hh_rlhf.helpful_base as skyrl_hh_rlhf_helpful_base
import experiments.post_training.task_curation.datasets.skyrl.hh_rlhf.helpful_online as skyrl_hh_rlhf_helpful_online
import experiments.post_training.task_curation.datasets.skyrl.kto_mix.capybara as skyrl_kto_mix_capybara
import experiments.post_training.task_curation.datasets.skyrl.kto_mix.intel_orca as skyrl_kto_mix_intel_orca
import experiments.post_training.task_curation.datasets.skyrl.kto_mix.ultrafeedback as skyrl_kto_mix_ultrafeedback
import experiments.post_training.task_curation.datasets.skyrl.math500 as skyrl_math500
import experiments.post_training.task_curation.datasets.skyrl.nemotron_if as skyrl_nemotron_if
import experiments.post_training.task_curation.datasets.skyrl.numina_math as skyrl_numina_math
import experiments.post_training.task_curation.datasets.skyrl.openscience as skyrl_openscience
import experiments.post_training.task_curation.datasets.skyrl.reasoning_gym_generated as skyrl_reasoning_gym_generated
import experiments.post_training.task_curation.datasets.skyrl.rlvr_ifeval as skyrl_rlvr_ifeval
import experiments.post_training.task_curation.datasets.skyrl.rlvr_math as skyrl_rlvr_math
import experiments.post_training.task_curation.datasets.skyrl.svamp as skyrl_svamp
import experiments.post_training.task_curation.datasets.skyrl.verifiable_code as skyrl_verifiable_code
import experiments.post_training.task_curation.datasets.tasktrove.all_puzzles as tasktrove_all_puzzles
import experiments.post_training.task_curation.datasets.tasktrove.arc_inductive as tasktrove_arc_inductive
import experiments.post_training.task_curation.datasets.tasktrove.arc_transductive as tasktrove_arc_transductive
import experiments.post_training.task_curation.datasets.tasktrove.calendar as tasktrove_calendar
import experiments.post_training.task_curation.datasets.tasktrove.calibforge_native as tasktrove_calibforge_native
import experiments.post_training.task_curation.datasets.tasktrove.code_contests as tasktrove_code_contests
import experiments.post_training.task_curation.datasets.tasktrove.codeforces as tasktrove_codeforces
import experiments.post_training.task_curation.datasets.tasktrove.codereview as tasktrove_codereview
import experiments.post_training.task_curation.datasets.tasktrove.competitive_coding as tasktrove_competitive_coding
import experiments.post_training.task_curation.datasets.tasktrove.curriculum_easy as tasktrove_curriculum_easy
import experiments.post_training.task_curation.datasets.tasktrove.curriculum_medium as tasktrove_curriculum_medium
import experiments.post_training.task_curation.datasets.tasktrove.e2egit as tasktrove_e2egit
import experiments.post_training.task_curation.datasets.tasktrove.e2egit_large as tasktrove_e2egit_large
import experiments.post_training.task_curation.datasets.tasktrove.glaive_code as tasktrove_glaive_code
import experiments.post_training.task_curation.datasets.tasktrove.if_calendar as tasktrove_if_calendar
import experiments.post_training.task_curation.datasets.tasktrove.knowledge_mcqa as tasktrove_knowledge_mcqa
import experiments.post_training.task_curation.datasets.tasktrove.knowledge_openqa as tasktrove_knowledge_openqa
import experiments.post_training.task_curation.datasets.tasktrove.math_gym as tasktrove_math_gym
import experiments.post_training.task_curation.datasets.tasktrove.math_openreasoning as tasktrove_math_openreasoning
import experiments.post_training.task_curation.datasets.tasktrove.math_oracle as tasktrove_math_oracle
import experiments.post_training.task_curation.datasets.tasktrove.math_prism as tasktrove_math_prism
import experiments.post_training.task_curation.datasets.tasktrove.math_stack as tasktrove_math_stack
import experiments.post_training.task_curation.datasets.tasktrove.mimo_code_native as tasktrove_mimo_code_native
import experiments.post_training.task_curation.datasets.tasktrove.mimo_music_native as tasktrove_mimo_music_native
import experiments.post_training.task_curation.datasets.tasktrove.multichallenge as tasktrove_multichallenge
import experiments.post_training.task_curation.datasets.tasktrove.multifile as tasktrove_multifile
import experiments.post_training.task_curation.datasets.tasktrove.nl2bash as tasktrove_nl2bash
import experiments.post_training.task_curation.datasets.tasktrove.openswe_oss_native as tasktrove_openswe_oss_native
import experiments.post_training.task_curation.datasets.tasktrove.openswe_other_native as tasktrove_openswe_other_native
import experiments.post_training.task_curation.datasets.tasktrove.pymethods as tasktrove_pymethods
import experiments.post_training.task_curation.datasets.tasktrove.pymethods_large as tasktrove_pymethods_large
import experiments.post_training.task_curation.datasets.tasktrove.r2egym_native as tasktrove_r2egym_native
import experiments.post_training.task_curation.datasets.tasktrove.reasoning_gym as tasktrove_reasoning_gym
import experiments.post_training.task_curation.datasets.tasktrove.safety as tasktrove_safety
import experiments.post_training.task_curation.datasets.tasktrove.science_openqa as tasktrove_science_openqa
import experiments.post_training.task_curation.datasets.tasktrove.stack_overflow as tasktrove_stack_overflow
import experiments.post_training.task_curation.datasets.tasktrove.stack_pytest as tasktrove_stack_pytest
import experiments.post_training.task_curation.datasets.tasktrove.structured_output as tasktrove_structured_output
import experiments.post_training.task_curation.datasets.tasktrove.structured_outputs as tasktrove_structured_outputs
import experiments.post_training.task_curation.datasets.tasktrove.superuser as tasktrove_superuser
import experiments.post_training.task_curation.datasets.tasktrove.swe_rebench as tasktrove_swe_rebench
import experiments.post_training.task_curation.datasets.tasktrove.swegym_native as tasktrove_swegym_native
import experiments.post_training.task_curation.datasets.tasktrove.swesmith as tasktrove_swesmith
import experiments.post_training.task_curation.datasets.tasktrove.taco as tasktrove_taco
import experiments.post_training.task_curation.datasets.tasktrove.tezos as tasktrove_tezos
import experiments.post_training.task_curation.datasets.tasktrove.unitsyn as tasktrove_unitsyn
import experiments.post_training.task_curation.datasets.tasktrove.unitsyn_large as tasktrove_unitsyn_large
import experiments.post_training.task_curation.datasets.tasktrove.unix as tasktrove_unix
import experiments.post_training.task_curation.datasets.tasktrove.wizard_orca as tasktrove_wizard_orca
from experiments.post_training.task_curation.datasets.nemotron_ultra.mopd import (
    calendar_v2 as nemotron_ultra_mopd_calendar_v2,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.mopd import (
    comp_coding as nemotron_ultra_mopd_comp_coding,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.mopd import (
    hs3_multiturn as nemotron_ultra_mopd_hs3_multiturn,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.mopd import (
    instruction_following as nemotron_ultra_mopd_instruction_following,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.mopd import (
    makeshn_ultra_v3_ipi_train as nemotron_ultra_mopd_makeshn_ultra_v3_ipi_train,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.mopd import (
    multichallenge_len40k as nemotron_ultra_mopd_multichallenge_len40k,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.mopd import (
    nvarc_inductive as nemotron_ultra_mopd_nvarc_inductive,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.mopd import (
    nvarc_transductive as nemotron_ultra_mopd_nvarc_transductive,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.mopd import (
    reasoning_gym as nemotron_ultra_mopd_reasoning_gym,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.mopd import (
    stem_mcqa_cot_rima_new as nemotron_ultra_mopd_stem_mcqa_cot_rima_new,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.mopd import (
    structured_outputs_v2 as nemotron_ultra_mopd_structured_outputs_v2,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.mopd import (
    toolcall_schema as nemotron_ultra_mopd_toolcall_schema,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.mopd.agentic import (
    citation_format_v2 as nemotron_ultra_mopd_agentic_citation_format_v2,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.mopd.agentic import (
    freeform_text_v2 as nemotron_ultra_mopd_agentic_freeform_text_v2,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.mopd.agentic import (
    structured_outputs_v2 as nemotron_ultra_mopd_agentic_structured_outputs_v2,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.mopd.agentic import (
    swe_gym as nemotron_ultra_mopd_agentic_swe_gym,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.mopd.agentic import (
    swe_nebius as nemotron_ultra_mopd_agentic_swe_nebius,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.mopd.next_action import (
    nebius as nemotron_ultra_mopd_next_action_nebius,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.mopd.next_action import (
    swe_gym as nemotron_ultra_mopd_next_action_swe_gym,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr1 import (
    abstention as nemotron_ultra_rlvr1_abstention,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr1 import (
    calendar_v2 as nemotron_ultra_rlvr1_calendar_v2,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr1 import (
    comp_coding as nemotron_ultra_rlvr1_comp_coding,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr1 import (
    instruction_following as nemotron_ultra_rlvr1_instruction_following,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr1 import (
    language_mixing as nemotron_ultra_rlvr1_language_mixing,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr1 import (
    multichallenge_len40k as nemotron_ultra_rlvr1_multichallenge_len40k,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr1 import (
    nvarc_inductive as nemotron_ultra_rlvr1_nvarc_inductive,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr1 import (
    nvarc_transductive as nemotron_ultra_rlvr1_nvarc_transductive,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr1 import (
    reasoning_gym as nemotron_ultra_rlvr1_reasoning_gym,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr1 import (
    stem_mcqa_cot_rima_new as nemotron_ultra_rlvr1_stem_mcqa_cot_rima_new,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr1 import (
    structured_outputs_v2 as nemotron_ultra_rlvr1_structured_outputs_v2,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr1 import (
    swe_nebius as nemotron_ultra_rlvr1_swe_nebius,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr1 import (
    toolcall_schema as nemotron_ultra_rlvr1_toolcall_schema,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr2 import (
    abstention as nemotron_ultra_rlvr2_abstention,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr2 import (
    calendar_v2 as nemotron_ultra_rlvr2_calendar_v2,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr2 import (
    comp_coding as nemotron_ultra_rlvr2_comp_coding,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr2 import (
    ds2_freeform as nemotron_ultra_rlvr2_ds2_freeform,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr2 import (
    ds3_citation as nemotron_ultra_rlvr2_ds3_citation,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr2 import (
    instruction_following as nemotron_ultra_rlvr2_instruction_following,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr2 import (
    language_mixing as nemotron_ultra_rlvr2_language_mixing,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr2 import (
    multichallenge_len40k as nemotron_ultra_rlvr2_multichallenge_len40k,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr2 import (
    nvarc_inductive as nemotron_ultra_rlvr2_nvarc_inductive,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr2 import (
    nvarc_transductive as nemotron_ultra_rlvr2_nvarc_transductive,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr2 import (
    reasoning_gym as nemotron_ultra_rlvr2_reasoning_gym,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr2 import (
    stem_mcqa_cot_rima_new as nemotron_ultra_rlvr2_stem_mcqa_cot_rima_new,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr2 import (
    structured_outputs_v2 as nemotron_ultra_rlvr2_structured_outputs_v2,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr2 import (
    structured_outputs_v3 as nemotron_ultra_rlvr2_structured_outputs_v3,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr2 import (
    swe_nebius as nemotron_ultra_rlvr2_swe_nebius,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.rlvr2 import (
    toolcall_schema as nemotron_ultra_rlvr2_toolcall_schema,
)
from experiments.post_training.task_curation.datasets.skyrl.hh_rlhf import (
    helpful_rejection_sampled as skyrl_hh_rlhf_helpful_rejection_sampled,
)
from experiments.post_training.task_curation.datasets.tasktrove import (
    instruction_following as tasktrove_instruction_following,
)
from experiments.post_training.task_curation.pipeline import AtlasSource, AtlasStatus, RlDataPipeline


def atlas_metadata() -> dict[str, AtlasSource]:
    """Read Atlas lineage and eligibility attached to the source declarations."""
    metadata = json.loads(Path(__file__).with_name("atlas_catalog.json").read_text())
    sources = {}
    for row in metadata["listings"]:
        historical = row.get("historical_inspection")
        sources[row["id"]] = AtlasSource(
            id=row["id"],
            name=row["name"],
            origin=row["origin"],
            family=row["family"],
            status=AtlasStatus(row["atlas_status"]),
            exclusion_reason=row["atlas_exclusion_reason"],
            dataset_id=row["dataset_id"],
            dataset_revision=row["dataset_revision"],
            archive_revision=row["revision"] if row["origin"] == "Task Trove" else None,
            verifier_revision=row["verifier_revision"],
            historical_disposition=historical["disposition"] if historical else None,
            historical_contract_changed=(
                any(row[key] != historical[key] for key in ("dataset_revision", "revision", "verifier_revision"))
                if historical
                else None
            ),
        )
    return sources


def rl_data_pipelines() -> dict[str, RlDataPipeline]:
    """Select dataset pipelines by their declared availability."""
    declarations = {
        "MarinSkyRL:aime_1983_2024": skyrl_aime_1983_2024.pipeline(),
        "MarinSkyRL:aime24": skyrl_aime24.pipeline(),
        "MarinSkyRL:apps": skyrl_apps.pipeline(),
        "MarinSkyRL:asdiv": skyrl_asdiv.pipeline(),
        "MarinSkyRL:dapo_math": skyrl_dapo_math.pipeline(),
        "MarinSkyRL:deepscaler": skyrl_deepscaler.pipeline(),
        "MarinSkyRL:eurus2_code": skyrl_eurus2_code.pipeline(),
        "MarinSkyRL:gpqa": skyrl_gpqa.pipeline(),
        "MarinSkyRL:gretel_text_to_sql": skyrl_gretel_text_to_sql.pipeline(),
        "MarinSkyRL:gsm8k": skyrl_gsm8k.pipeline(),
        "MarinSkyRL:hardmath": skyrl_hardmath.pipeline(),
        "MarinSkyRL:hendrycks_math": skyrl_hendrycks_math.pipeline(),
        "MarinSkyRL:hh_rlhf/harmless-base": skyrl_hh_rlhf_harmless_base.pipeline(),
        "MarinSkyRL:hh_rlhf/helpful-base": skyrl_hh_rlhf_helpful_base.pipeline(),
        "MarinSkyRL:hh_rlhf/helpful-online": skyrl_hh_rlhf_helpful_online.pipeline(),
        "MarinSkyRL:hh_rlhf/helpful-rejection-sampled": skyrl_hh_rlhf_helpful_rejection_sampled.pipeline(),
        "MarinSkyRL:kto_mix/argilla/distilabel-capybara-dpo-7k-binarized": skyrl_kto_mix_capybara.pipeline(),
        "MarinSkyRL:kto_mix/argilla/distilabel-intel-orca-dpo-pairs": skyrl_kto_mix_intel_orca.pipeline(),
        "MarinSkyRL:kto_mix/argilla/ultrafeedback-binarized-preferences-cleaned": skyrl_kto_mix_ultrafeedback.pipeline(),
        "MarinSkyRL:math500": skyrl_math500.pipeline(),
        "MarinSkyRL:nemotron_if": skyrl_nemotron_if.pipeline(),
        (
            "MarinSkyRL:nemotron_ultra_mopd/"
            "agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent/nebius/SWE-rebench-V2"
        ): nemotron_ultra_mopd_next_action_nebius.pipeline(),
        (
            "MarinSkyRL:nemotron_ultra_mopd/"
            "agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent/SWE-Gym/SWE-Gym"
        ): nemotron_ultra_mopd_next_action_swe_gym.pipeline(),
        "MarinSkyRL:nemotron_ultra_mopd/hs3_en": nemotron_ultra_mopd_hs3_en.pipeline(),
        "MarinSkyRL:nemotron_ultra_mopd/hs3_multi": nemotron_ultra_mopd_hs3_multi.pipeline(),
        "MarinSkyRL:nemotron_ultra_mopd/hs3_multiturn": nemotron_ultra_mopd_hs3_multiturn.pipeline(),
        "MarinSkyRL:nemotron_ultra_mopd/makeshn_ultra_v3_ipi_train": (
            nemotron_ultra_mopd_makeshn_ultra_v3_ipi_train.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_mopd/safety_en": nemotron_ultra_mopd_safety_en.pipeline(),
        "MarinSkyRL:nemotron_ultra_mopd/swe_pivot_len40k/nebius/SWE-rebench-V2": (
            nemotron_ultra_mopd_swe_nebius.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_mopd/swe_pivot_len40k/SWE-Gym/SWE-Gym": nemotron_ultra_mopd_swe_gym.pipeline(),
        "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_abstention": nemotron_ultra_mopd_abstention.pipeline(),
        "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_calendar_v2": nemotron_ultra_mopd_calendar_v2.pipeline(),
        "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_comp_coding": nemotron_ultra_mopd_comp_coding.pipeline(),
        "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_instruction_following": (
            nemotron_ultra_mopd_instruction_following.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_jailbreak": nemotron_ultra_mopd_jailbreak.pipeline(),
        "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_lean": nemotron_ultra_mopd_lean.pipeline(),
        "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_math_cot": nemotron_ultra_mopd_math_cot.pipeline(),
        "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_math_tir": nemotron_ultra_mopd_math_tir.pipeline(),
        "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_multichallenge_len40k": (
            nemotron_ultra_mopd_multichallenge_len40k.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_nvarc_inductive": (
            nemotron_ultra_mopd_nvarc_inductive.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_nvarc_transductive": (
            nemotron_ultra_mopd_nvarc_transductive.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_rdkit": nemotron_ultra_mopd_rdkit.pipeline(),
        "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_reasoning_gym": nemotron_ultra_mopd_reasoning_gym.pipeline(),
        "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_stem_mcqa": nemotron_ultra_mopd_stem_mcqa.pipeline(),
        "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_stem_mcqa_cot_rima_new": (
            nemotron_ultra_mopd_stem_mcqa_cot_rima_new.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_structured_outputs_v2": (
            nemotron_ultra_mopd_structured_outputs_v2.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_toolcall_schema": (
            nemotron_ultra_mopd_toolcall_schema.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_citation_format_v2": (
            nemotron_ultra_mopd_agentic_citation_format_v2.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_freeform_text_v2": (
            nemotron_ultra_mopd_agentic_freeform_text_v2.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_structured_outputs_v2": (
            nemotron_ultra_mopd_agentic_structured_outputs_v2.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/nebius/SWE-rebench-V2": (
            nemotron_ultra_mopd_agentic_swe_nebius.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/SWE-Gym/SWE-Gym": (
            nemotron_ultra_mopd_agentic_swe_gym.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_rlvr1/language_mixing_hs3_ultra_genrm_fmt": (
            nemotron_ultra_rlvr1_language_mixing.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_abstention": nemotron_ultra_rlvr1_abstention.pipeline(),
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_calendar_v2": nemotron_ultra_rlvr1_calendar_v2.pipeline(),
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_comp_coding": nemotron_ultra_rlvr1_comp_coding.pipeline(),
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_instruction_following": (
            nemotron_ultra_rlvr1_instruction_following.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_jailbreak": nemotron_ultra_rlvr1_jailbreak.pipeline(),
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_lean": nemotron_ultra_rlvr1_lean.pipeline(),
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_math_cot": nemotron_ultra_rlvr1_math_cot.pipeline(),
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_math_tir": nemotron_ultra_rlvr1_math_tir.pipeline(),
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_multichallenge_len40k": (
            nemotron_ultra_rlvr1_multichallenge_len40k.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_nvarc_inductive": (
            nemotron_ultra_rlvr1_nvarc_inductive.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_nvarc_transductive": (
            nemotron_ultra_rlvr1_nvarc_transductive.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_reasoning_gym": (
            nemotron_ultra_rlvr1_reasoning_gym.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_stem_mcqa": nemotron_ultra_rlvr1_stem_mcqa.pipeline(),
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_stem_mcqa_cot_rima_new": (
            nemotron_ultra_rlvr1_stem_mcqa_cot_rima_new.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_structured_outputs_v2": (
            nemotron_ultra_rlvr1_structured_outputs_v2.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_swe_pivot_len40k/nebius/SWE-rebench-V2": (
            nemotron_ultra_rlvr1_swe_nebius.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_swe_pivot_len40k/SWE-Gym/SWE-Gym": (
            nemotron_ultra_rlvr1_swe_gym.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_tau_pivot": nemotron_ultra_rlvr1_tau_pivot.pipeline(),
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_toolcall_schema": (
            nemotron_ultra_rlvr1_toolcall_schema.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_rlvr2/language_mixing_hs3_ultra_genrm_fmt": (
            nemotron_ultra_rlvr2_language_mixing.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_abstention": nemotron_ultra_rlvr2_abstention.pipeline(),
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_calendar_v2": nemotron_ultra_rlvr2_calendar_v2.pipeline(),
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_comp_coding": nemotron_ultra_rlvr2_comp_coding.pipeline(),
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_ds2_freeform": nemotron_ultra_rlvr2_ds2_freeform.pipeline(),
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_ds3_citation": nemotron_ultra_rlvr2_ds3_citation.pipeline(),
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_instruction_following": (
            nemotron_ultra_rlvr2_instruction_following.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_jailbreak": nemotron_ultra_rlvr2_jailbreak.pipeline(),
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_lean": nemotron_ultra_rlvr2_lean.pipeline(),
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_math_cot": nemotron_ultra_rlvr2_math_cot.pipeline(),
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_math_tir": nemotron_ultra_rlvr2_math_tir.pipeline(),
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_multichallenge_len40k": (
            nemotron_ultra_rlvr2_multichallenge_len40k.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_nvarc_inductive": (
            nemotron_ultra_rlvr2_nvarc_inductive.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_nvarc_transductive": (
            nemotron_ultra_rlvr2_nvarc_transductive.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_rdkit": nemotron_ultra_rlvr2_rdkit.pipeline(),
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_reasoning_gym": (
            nemotron_ultra_rlvr2_reasoning_gym.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_stem_mcqa": nemotron_ultra_rlvr2_stem_mcqa.pipeline(),
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_stem_mcqa_cot_rima_new": (
            nemotron_ultra_rlvr2_stem_mcqa_cot_rima_new.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_structured_outputs_v2": (
            nemotron_ultra_rlvr2_structured_outputs_v2.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_structured_outputs_v3": (
            nemotron_ultra_rlvr2_structured_outputs_v3.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_swe_pivot_len40k/nebius/SWE-rebench-V2": (
            nemotron_ultra_rlvr2_swe_nebius.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_swe_pivot_len40k/SWE-Gym/SWE-Gym": (
            nemotron_ultra_rlvr2_swe_gym.pipeline()
        ),
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_tau_pivot": nemotron_ultra_rlvr2_tau_pivot.pipeline(),
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_toolcall_schema": (
            nemotron_ultra_rlvr2_toolcall_schema.pipeline()
        ),
        "MarinSkyRL:numina_math": skyrl_numina_math.pipeline(),
        "MarinSkyRL:openscience": skyrl_openscience.pipeline(),
        "MarinSkyRL:reasoning_gym": skyrl_reasoning_gym_generated.pipeline(),
        "MarinSkyRL:rlvr_ifeval": skyrl_rlvr_ifeval.pipeline(),
        "MarinSkyRL:rlvr_math": skyrl_rlvr_math.pipeline(),
        "MarinSkyRL:svamp": skyrl_svamp.pipeline(),
        "MarinSkyRL:verifiable_code": skyrl_verifiable_code.pipeline(),
        "Task Trove:AweAI-Team__CalibForge": tasktrove_calibforge_native.pipeline(),
        "Task Trove:DCAgent2__nl2bash-tasks-cleaned-oracle-v2": tasktrove_nl2bash.pipeline(),
        "Task Trove:DCAgent__code-contests-noblock": tasktrove_code_contests.pipeline(),
        "Task Trove:DCAgent__exp_rpt_curriculum-easy": tasktrove_curriculum_easy.pipeline(),
        "Task Trove:DCAgent__exp_rpt_curriculum-medium-v2": tasktrove_curriculum_medium.pipeline(),
        "Task Trove:DCAgent__exp_rpt_e2egit-large": tasktrove_e2egit_large.pipeline(),
        "Task Trove:DCAgent__exp_rpt_e2egit-v2": tasktrove_e2egit.pipeline(),
        "Task Trove:DCAgent__exp_rpt_multifile-v3": tasktrove_multifile.pipeline(),
        "Task Trove:DCAgent__exp_rpt_pymethods2test-large-v2": tasktrove_pymethods_large.pipeline(),
        "Task Trove:DCAgent__exp_rpt_pymethods2test-v3": tasktrove_pymethods.pipeline(),
        "Task Trove:DCAgent__exp_rpt_stack-pytest-v2": tasktrove_stack_pytest.pipeline(),
        "Task Trove:DCAgent__exp_rpt_unitsyn-python-large-v2": tasktrove_unitsyn_large.pipeline(),
        "Task Trove:DCAgent__exp_rpt_unitsyn-python-v4": tasktrove_unitsyn.pipeline(),
        "Task Trove:DCAgent__swe_rebench_v2_patched_oracle-v2": tasktrove_swe_rebench.pipeline(),
        "Task Trove:GAIR__OpenSWE__openswe_oss": tasktrove_openswe_oss_native.pipeline(),
        "Task Trove:GAIR__OpenSWE__openswe_other": tasktrove_openswe_other_native.pipeline(),
        "Task Trove:laion__all-puzzles-v2": tasktrove_all_puzzles.pipeline(),
        "Task Trove:laion__codeforces-v3": tasktrove_codeforces.pipeline(),
        "Task Trove:laion__exp_rpt_taco-v2": tasktrove_taco.pipeline(),
        "Task Trove:laion__glaive-code-assistant-sandboxes-verified-v2": tasktrove_glaive_code.pipeline(),
        "Task Trove:laion__nemo-prism-math-v3": tasktrove_math_prism.pipeline(),
        "Task Trove:laion__nemotron-gym-agent-calendar-v2": tasktrove_calendar.pipeline(),
        "Task Trove:laion__nemotron-gym-arc-agi-python-inductive-v2": tasktrove_arc_inductive.pipeline(),
        "Task Trove:laion__nemotron-gym-arc-agi-transductive-v3": tasktrove_arc_transductive.pipeline(),
        "Task Trove:laion__nemotron-gym-competitive-coding-v2": tasktrove_competitive_coding.pipeline(),
        "Task Trove:laion__nemotron-gym-instruction-following-calendar-v3": tasktrove_if_calendar.pipeline(),
        "Task Trove:laion__nemotron-gym-instruction-following-structured-v3": tasktrove_structured_output.pipeline(),
        "Task Trove:laion__nemotron-gym-instruction-following-v3": tasktrove_instruction_following.pipeline(),
        "Task Trove:laion__nemotron-gym-knowledge-mcqa-v2": tasktrove_knowledge_mcqa.pipeline(),
        "Task Trove:laion__nemotron-gym-knowledge-openqa-v4": tasktrove_knowledge_openqa.pipeline(),
        "Task Trove:laion__nemotron-gym-math-openmathreasoning-v2": tasktrove_math_openreasoning.pipeline(),
        "Task Trove:laion__nemotron-gym-math-stack-overflow-v3": tasktrove_math_stack.pipeline(),
        "Task Trove:laion__nemotron-gym-math-v5": tasktrove_math_gym.pipeline(),
        "Task Trove:laion__nemotron-gym-multichallenge-advanced-v4": tasktrove_multichallenge.pipeline(),
        "Task Trove:laion__nemotron-gym-reasoning-gym-v2": tasktrove_reasoning_gym.pipeline(),
        "Task Trove:laion__nemotron-gym-safety-v3": tasktrove_safety.pipeline(),
        "Task Trove:laion__nemotron-gym-science-so-openq-v3": tasktrove_science_openqa.pipeline(),
        "Task Trove:laion__nemotron-gym-structured-outputs-v4": tasktrove_structured_outputs.pipeline(),
        "Task Trove:laion__stackexchange-codereview-sandboxes-verified-v2": tasktrove_codereview.pipeline(),
        "Task Trove:laion__stackexchange-overflow-sandboxes-verified-v2": tasktrove_stack_overflow.pipeline(),
        "Task Trove:laion__stackexchange-superuser-sandboxes-verified-v2": tasktrove_superuser.pipeline(),
        "Task Trove:laion__stackexchange-tezos-sandboxes-verified-v2": tasktrove_tezos.pipeline(),
        "Task Trove:laion__stackexchange-unix-sandboxes-verified-v2": tasktrove_unix.pipeline(),
        "Task Trove:laion__swesmith-oracle-filtered-v2": tasktrove_swesmith.pipeline(),
        "Task Trove:laion__wizardlm-orca-v4": tasktrove_wizard_orca.pipeline(),
        "Task Trove:R2E-Gym__R2E-Gym-V1": tasktrove_r2egym_native.pipeline(),
        "Task Trove:SankalpKJ__nemotron-math-oracle-filtered-v2": tasktrove_math_oracle.pipeline(),
        "Task Trove:SWE-Gym__SWE-Gym": tasktrove_swegym_native.pipeline(),
        "Task Trove:XiaomiMiMo__MiMo-V2.6-RL-oss__code": tasktrove_mimo_code_native.pipeline(),
        "Task Trove:XiaomiMiMo__MiMo-V2.6-RL-oss__music": tasktrove_mimo_music_native.pipeline(),
    }
    metadata = atlas_metadata()
    declarations = {key: replace(definition, atlas=metadata.get(key)) for key, definition in declarations.items()}
    return declarations
