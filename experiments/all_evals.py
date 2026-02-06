# Copyright 2025 The Marin Authors
# Include fix v2: CohereLabs/include-base-44 (2026-01-28 02:55 UTC)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Comprehensive evaluation suite.

This module provides evaluation configurations organized into logprob
(multiple choice) and generative tasks.

Logprob tasks use Levanter's JAX-based inference (TPU-native).
Generative tasks use vLLM for text generation.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Literal

import fsspec
from fray.cluster import ResourceConfig

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, InputName, executor_main, this_output_path
from marin.utils import fsspec_exists, fsspec_glob, fsspec_mkdirs, fsspec_mtime

from experiments.evals.evals import (
    evaluate_levanter_lm_evaluation_harness,
    evaluate_lm_evaluation_harness,
    extract_model_name_and_path,
)
from experiments.models import qwen2_5_0_5b

logger = logging.getLogger(__name__)


# =============================================================================
# Edit Here (Grug Surface)
# =============================================================================

# Set the model to evaluate. Options:
# - an ExecutorStep
# - an InputName (e.g., InputName.hardcoded("gs://.../hf/step-12345"))
# - a string path or HF model id
MODEL_STEP: ExecutorStep | InputName | str | None = qwen2_5_0_5b

# General knobs.
RESOURCE_CONFIG = ResourceConfig.with_tpu("v5p-8")
DISCOVER_LATEST_CHECKPOINT = True
MAX_EVAL_INSTANCES: int | None = None

# vLLM / lm-eval args.
DEFAULT_ENGINE_KWARGS = {
    "max_model_len": 4096,
    "max_length": 4096,
    "max_gen_toks": 1024,
}
ENGINE_KWARGS = DEFAULT_ENGINE_KWARGS


# =============================================================================
# Task Configurations
# =============================================================================

# -----------------------------------------------------------------------------
# Core QA / Reading Comprehension Tasks (Logprob - Multiple Choice)
# -----------------------------------------------------------------------------
CORE_QA_MC_TASKS = (
    # ARC (AI2 Reasoning Challenge)
    EvalTaskConfig("arc_easy", 0, task_alias="arc_easy_0shot"),
    EvalTaskConfig("arc_challenge", 0, task_alias="arc_challenge_0shot"),
    # MMLU variants
    EvalTaskConfig("mmlu", 0, task_alias="mmlu_0shot"),
    EvalTaskConfig("mmlu", 5, task_alias="mmlu_5shot"),
    # CommonsenseQA
    EvalTaskConfig("commonsense_qa", 0, task_alias="csqa_0shot"),
    # HellaSwag
    EvalTaskConfig("hellaswag", 0, task_alias="hellaswag_0shot"),
    # WinoGrande
    EvalTaskConfig("winogrande", 0, task_alias="winogrande_0shot"),
    # Social IQa
    EvalTaskConfig("social_iqa", 0, task_alias="socialiqa_0shot"),
    # PIQA (Physical IQa)
    EvalTaskConfig("piqa", 0, task_alias="piqa_0shot"),
    # SciQ
    EvalTaskConfig("sciq", 0, task_alias="sciq_0shot"),
    # OpenBookQA
    EvalTaskConfig("openbookqa", 0, task_alias="openbookqa_0shot"),
    # BoolQ
    EvalTaskConfig("boolq", 0, task_alias="boolq_0shot"),
)

# Medical QA tasks
MEDICAL_MC_TASKS = (
    EvalTaskConfig("medmcqa", 0, task_alias="medmcqa_0shot"),
    EvalTaskConfig("medqa_4options", 0, task_alias="medqa_0shot"),
    EvalTaskConfig("pubmedqa", 0, task_alias="pubmedqa_0shot"),
)

# -----------------------------------------------------------------------------
# Language Understanding Tasks (Logprob)
# -----------------------------------------------------------------------------
LANGUAGE_UNDERSTANDING_TASKS = (
    # LAMBADA - predict word endings
    EvalTaskConfig("lambada_openai", 0, task_alias="lambada_0shot"),
    # WSC273 - Winograd Schema Challenge
    EvalTaskConfig("wsc273", 0, task_alias="wsc273_0shot"),
    # COPA - causal reasoning
    EvalTaskConfig("copa", 0, task_alias="copa_0shot"),
)

# -----------------------------------------------------------------------------
# Reasoning Tasks (Logprob - BBH zeroshot/fewshot)
# -----------------------------------------------------------------------------
REASONING_MC_TASKS = (
    EvalTaskConfig("bbh_zeroshot", 0, task_alias="bbh_zeroshot"),
    EvalTaskConfig("bbh_fewshot", 3, task_alias="bbh_3shot"),
    EvalTaskConfig("logiqa", 0, task_alias="logiqa_0shot"),
    EvalTaskConfig("logiqa2", 0, task_alias="logiqa2_0shot"),
)

# -----------------------------------------------------------------------------
# Leaderboard Tasks (Logprob - GPQA, MUSR, etc.)
# -----------------------------------------------------------------------------
LEADERBOARD_MC_TASKS = (
    EvalTaskConfig("leaderboard_gpqa", 0, task_alias="gpqa_0shot"),
    EvalTaskConfig("leaderboard_musr", 0, task_alias="musr_0shot"),
    EvalTaskConfig("leaderboard_bbh", 3, task_alias="lb_bbh_3shot"),
    EvalTaskConfig("leaderboard_mmlu_pro", 5, task_alias="mmlu_pro_5shot"),
)

# -----------------------------------------------------------------------------
# AGI Eval (Logprob)
# -----------------------------------------------------------------------------
AGI_EVAL_TASKS = (
    EvalTaskConfig("agieval_aqua_rat", 0, task_alias="agieval_aqua_rat_0shot"),
    EvalTaskConfig("agieval_lsat_ar", 0, task_alias="agieval_lsat_ar_0shot"),
    EvalTaskConfig("agieval_lsat_lr", 0, task_alias="agieval_lsat_lr_0shot"),
    EvalTaskConfig("agieval_lsat_rc", 0, task_alias="agieval_lsat_rc_0shot"),
    EvalTaskConfig("agieval_sat_en", 0, task_alias="agieval_sat_en_0shot"),
    EvalTaskConfig("agieval_sat_math", 0, task_alias="agieval_sat_math_0shot"),
)

# -----------------------------------------------------------------------------
# Truthfulness Tasks (Logprob)
# -----------------------------------------------------------------------------
TRUTHFULNESS_MC_TASKS = (
    EvalTaskConfig("truthfulqa_mc1", 0, task_alias="truthfulqa_mc1_0shot"),
    EvalTaskConfig("truthfulqa_mc2", 0, task_alias="truthfulqa_mc2_0shot"),
)

# -----------------------------------------------------------------------------
# Multilingual Tasks (Logprob) - Diverse subset
# Selected for linguistic diversity across language families and scripts:
#   - Romance: Spanish, French, Portuguese
#   - Germanic: German
#   - Slavic: Russian
#   - East Asian: Chinese, Japanese, Korean
#   - South Asian: Hindi, Bengali
#   - Semitic: Arabic, Hebrew
#   - Turkic: Turkish
#   - Southeast Asian: Vietnamese, Indonesian, Thai
#   - African: Swahili, Yoruba
# -----------------------------------------------------------------------------
BELEBELE_DIVERSE_TASKS = (
    # Romance
    EvalTaskConfig("belebele_spa_Latn", 0, task_alias="belebele_spanish"),
    EvalTaskConfig("belebele_fra_Latn", 0, task_alias="belebele_french"),
    EvalTaskConfig("belebele_por_Latn", 0, task_alias="belebele_portuguese"),
    # Germanic
    EvalTaskConfig("belebele_deu_Latn", 0, task_alias="belebele_german"),
    # Slavic
    EvalTaskConfig("belebele_rus_Cyrl", 0, task_alias="belebele_russian"),
    # East Asian
    EvalTaskConfig("belebele_zho_Hans", 0, task_alias="belebele_chinese_simplified"),
    EvalTaskConfig("belebele_jpn_Jpan", 0, task_alias="belebele_japanese"),
    EvalTaskConfig("belebele_kor_Hang", 0, task_alias="belebele_korean"),
    # South Asian
    EvalTaskConfig("belebele_hin_Deva", 0, task_alias="belebele_hindi"),
    EvalTaskConfig("belebele_ben_Beng", 0, task_alias="belebele_bengali"),
    # Semitic
    EvalTaskConfig("belebele_arb_Arab", 0, task_alias="belebele_arabic"),
    EvalTaskConfig("belebele_heb_Hebr", 0, task_alias="belebele_hebrew"),
    # Turkic
    EvalTaskConfig("belebele_tur_Latn", 0, task_alias="belebele_turkish"),
    # Southeast Asian
    EvalTaskConfig("belebele_vie_Latn", 0, task_alias="belebele_vietnamese"),
    EvalTaskConfig("belebele_ind_Latn", 0, task_alias="belebele_indonesian"),
    EvalTaskConfig("belebele_tha_Thai", 0, task_alias="belebele_thai"),
    # African
    EvalTaskConfig("belebele_swh_Latn", 0, task_alias="belebele_swahili"),
    EvalTaskConfig("belebele_yor_Latn", 0, task_alias="belebele_yoruba"),
)

INCLUDE_DIVERSE_TASKS = (
    # Romance (group names don't have _few_shot_og suffix)
    EvalTaskConfig("include_base_44_spanish", 5, task_alias="include_spanish"),
    EvalTaskConfig("include_base_44_french", 5, task_alias="include_french"),
    EvalTaskConfig("include_base_44_portuguese", 5, task_alias="include_portuguese"),
    # Germanic
    EvalTaskConfig("include_base_44_german", 5, task_alias="include_german"),
    # Slavic
    EvalTaskConfig("include_base_44_russian", 5, task_alias="include_russian"),
    # East Asian
    EvalTaskConfig("include_base_44_chinese", 5, task_alias="include_chinese"),
    EvalTaskConfig("include_base_44_japanese", 5, task_alias="include_japanese"),
    EvalTaskConfig("include_base_44_korean", 5, task_alias="include_korean"),
    # South Asian
    EvalTaskConfig("include_base_44_hindi", 5, task_alias="include_hindi"),
    EvalTaskConfig("include_base_44_bengali", 5, task_alias="include_bengali"),
    # Semitic
    EvalTaskConfig("include_base_44_arabic", 5, task_alias="include_arabic"),
    EvalTaskConfig("include_base_44_hebrew", 5, task_alias="include_hebrew"),
    # Turkic
    EvalTaskConfig("include_base_44_turkish", 5, task_alias="include_turkish"),
    # Southeast Asian
    EvalTaskConfig("include_base_44_vietnamese", 5, task_alias="include_vietnamese"),
    EvalTaskConfig("include_base_44_indonesian", 5, task_alias="include_indonesian"),
)

# Combined diverse multilingual tasks (logprob)
DIVERSE_MULTILINGUAL_TASKS = (
    *BELEBELE_DIVERSE_TASKS,
    *INCLUDE_DIVERSE_TASKS,
)

# MGSM - Multilingual Grade School Math (generative, requires CoT)
# Subset matching our diverse language selection
MGSM_DIVERSE_TASKS = (
    EvalTaskConfig("mgsm_direct_es", 0, task_alias="mgsm_spanish"),
    EvalTaskConfig("mgsm_direct_fr", 0, task_alias="mgsm_french"),
    EvalTaskConfig("mgsm_direct_de", 0, task_alias="mgsm_german"),
    EvalTaskConfig("mgsm_direct_ru", 0, task_alias="mgsm_russian"),
    EvalTaskConfig("mgsm_direct_zh", 0, task_alias="mgsm_chinese"),
    EvalTaskConfig("mgsm_direct_ja", 0, task_alias="mgsm_japanese"),
)

# =============================================================================
# Generative Tasks (require vLLM)
# =============================================================================

# -----------------------------------------------------------------------------
# Instruction Following (Generative)
# -----------------------------------------------------------------------------
# ifeval removed: requires langdetect which isn't in runtime env
INSTRUCTION_FOLLOWING_GEN_TASKS = ()

# -----------------------------------------------------------------------------
# Math Tasks (Generative - CoT)
# -----------------------------------------------------------------------------
MATH_GEN_TASKS = (
    EvalTaskConfig("gsm8k", 5, task_alias="gsm8k_5shot"),
    EvalTaskConfig("gsm8k_cot", 8, task_alias="gsm8k_cot_8shot"),
    EvalTaskConfig("minerva_math_algebra", 4, task_alias="minerva_math_algebra_4shot"),
    EvalTaskConfig("minerva_math_counting_and_prob", 4, task_alias="minerva_math_counting_4shot"),
    EvalTaskConfig("minerva_math_geometry", 4, task_alias="minerva_math_geometry_4shot"),
    EvalTaskConfig("minerva_math_intermediate_algebra", 4, task_alias="minerva_math_intermediate_4shot"),
    EvalTaskConfig("minerva_math_num_theory", 4, task_alias="minerva_math_num_theory_4shot"),
    EvalTaskConfig("minerva_math_prealgebra", 4, task_alias="minerva_math_prealgebra_4shot"),
    EvalTaskConfig("minerva_math_precalc", 4, task_alias="minerva_math_precalc_4shot"),
)

# AIME (competition math) - removed: requires max_gen_toks=32768 which exceeds 4096 context
AIME_GEN_TASKS = ()

# -----------------------------------------------------------------------------
# Reasoning Tasks (Generative - BBH requires generation, both CoT and non-CoT)
# -----------------------------------------------------------------------------
REASONING_GEN_TASKS = (
    EvalTaskConfig("bbh_cot_fewshot", 3, task_alias="bbh_cot_3shot"),
    EvalTaskConfig("bbh_cot_zeroshot", 0, task_alias="bbh_cot_0shot"),
    # Non-CoT BBH variants also use generate_until
    EvalTaskConfig("bbh_zeroshot", 0, task_alias="bbh_zeroshot"),
    EvalTaskConfig("bbh_fewshot", 3, task_alias="bbh_fewshot"),
    # logieval uses generate_until (see logiqa2/logieval.yaml)
    EvalTaskConfig("logieval", 1, task_alias="logiqa2_1shot"),
)

# -----------------------------------------------------------------------------
# Code Tasks (Generative)
# -----------------------------------------------------------------------------
CODE_GEN_TASKS = (
    EvalTaskConfig("code2text_go", 0, task_alias="code2text_go_0shot"),
    EvalTaskConfig("code2text_java", 0, task_alias="code2text_java_0shot"),
    EvalTaskConfig("code2text_javascript", 0, task_alias="code2text_javascript_0shot"),
    EvalTaskConfig("code2text_php", 0, task_alias="code2text_php_0shot"),
    EvalTaskConfig("code2text_python", 0, task_alias="code2text_python_0shot"),
    EvalTaskConfig("code2text_ruby", 0, task_alias="code2text_ruby_0shot"),
    EvalTaskConfig("humaneval", 0, task_alias="humaneval_0shot"),
    EvalTaskConfig("mbpp", 3, task_alias="mbpp_3shot"),
    # bigcodebench and ds1000 removed: require max_gen_toks=32768 which exceeds 4096 context
)

# MultiPL-E (multi-language code)
MULTIPL_E_GEN_TASKS = (
    EvalTaskConfig("multipl_e_humaneval_cpp", 0, task_alias="multipl_e_humaneval_cpp"),
    EvalTaskConfig("multipl_e_humaneval_java", 0, task_alias="multipl_e_humaneval_java"),
    EvalTaskConfig("multipl_e_humaneval_js", 0, task_alias="multipl_e_humaneval_js"),
    EvalTaskConfig("multipl_e_humaneval_rs", 0, task_alias="multipl_e_humaneval_rs"),
    EvalTaskConfig("multipl_e_mbpp_cpp", 0, task_alias="multipl_e_mbpp_cpp"),
    EvalTaskConfig("multipl_e_mbpp_java", 0, task_alias="multipl_e_mbpp_java"),
    EvalTaskConfig("multipl_e_mbpp_js", 0, task_alias="multipl_e_mbpp_js"),
    EvalTaskConfig("multipl_e_mbpp_rs", 0, task_alias="multipl_e_mbpp_rs"),
)

# -----------------------------------------------------------------------------
# Reading Comprehension / QA (Generative)
# -----------------------------------------------------------------------------
QA_GEN_TASKS = (
    EvalTaskConfig("drop", 0, task_alias="drop_0shot"),
    EvalTaskConfig("coqa", 0, task_alias="coqa_0shot"),
    # squadv2 uses generate_until + loglikelihood (for "unanswerable" detection)
    EvalTaskConfig("nq_open", 0, task_alias="naturalqs_0shot"),
    EvalTaskConfig("triviaqa", 0, task_alias="triviaqa_0shot"),
    EvalTaskConfig("squadv2", 0, task_alias="squadv2_0shot"),
    EvalTaskConfig("popqa", 0, task_alias="popqa_0shot"),
)

# -----------------------------------------------------------------------------
# Perplexity / Loglikelihood (run with vLLM, not Levanter)
# -----------------------------------------------------------------------------
PPL_GEN_TASKS = (EvalTaskConfig("wikitext", 0, task_alias="wikitext_0shot"),)

# Jeopardy
JEOPARDY_GEN_TASKS = (EvalTaskConfig("jeopardy", 0, task_alias="jeopardy_0shot"),)

# -----------------------------------------------------------------------------
# Leaderboard Generative Tasks
# -----------------------------------------------------------------------------
# leaderboard_ifeval removed: requires langdetect which isn't in runtime env
LEADERBOARD_GEN_TASKS = (EvalTaskConfig("leaderboard_math_hard", 4, task_alias="lb_math_4shot"),)

# -----------------------------------------------------------------------------
# Hendrycks MATH (Generative - all variants use generate_until)
# -----------------------------------------------------------------------------
HENDRYCKS_MATH_GEN_TASKS = (
    EvalTaskConfig("hendrycks_math_algebra", 0, task_alias="hendrycks_math_algebra_0shot"),
    EvalTaskConfig("hendrycks_math_counting_and_prob", 0, task_alias="hendrycks_math_counting_0shot"),
    EvalTaskConfig("hendrycks_math_geometry", 0, task_alias="hendrycks_math_geometry_0shot"),
    EvalTaskConfig("hendrycks_math_intermediate_algebra", 0, task_alias="hendrycks_math_intermediate_0shot"),
    EvalTaskConfig("hendrycks_math_num_theory", 0, task_alias="hendrycks_math_num_theory_0shot"),
    EvalTaskConfig("hendrycks_math_prealgebra", 0, task_alias="hendrycks_math_prealgebra_0shot"),
    EvalTaskConfig("hendrycks_math_precalc", 0, task_alias="hendrycks_math_precalc_0shot"),
)

# -----------------------------------------------------------------------------
# JSON Schema Bench (Generative - all variants use generate_until)
# -----------------------------------------------------------------------------
JSONSCHEMA_GEN_TASKS = (
    EvalTaskConfig("jsonschema_bench_easy", 2, task_alias="jsonschema_bench_easy_2shot"),
    EvalTaskConfig("jsonschema_bench_medium", 2, task_alias="jsonschema_bench_medium_2shot"),
    EvalTaskConfig("jsonschema_bench_hard", 2, task_alias="jsonschema_bench_hard_2shot"),
)

# -----------------------------------------------------------------------------
# TruthfulQA Generation variant
# -----------------------------------------------------------------------------
TRUTHFULQA_GEN_TASKS = (EvalTaskConfig("truthfulqa_gen", 0, task_alias="truthfulqa_gen_0shot"),)

# -----------------------------------------------------------------------------
# SQUAD Completion (Generative - uses generate_until, see task.py)
# -----------------------------------------------------------------------------
SQUAD_COMPLETION_GEN_TASKS = (EvalTaskConfig("squad_completion", 0, task_alias="squad_completion_0shot"),)


# -----------------------------------------------------------------------------
# Korean Medical MCQA (Generative - all variants use generate_until)
# See: lm_eval/tasks/kormedmcqa/_template_yaml
# -----------------------------------------------------------------------------
KORMEDMCQA_GEN_TASKS = (
    EvalTaskConfig("kormedmcqa_dentist", 0, task_alias="kormedmcqa_dentist_0shot"),
    EvalTaskConfig("kormedmcqa_doctor", 0, task_alias="kormedmcqa_doctor_0shot"),
    EvalTaskConfig("kormedmcqa_nurse", 0, task_alias="kormedmcqa_nurse_0shot"),
    EvalTaskConfig("kormedmcqa_pharm", 0, task_alias="kormedmcqa_pharm_0shot"),
)

# -----------------------------------------------------------------------------
# EQ-Bench (Generative - uses generate_until)
# See: lm_eval/tasks/eq_bench/default.yaml
# -----------------------------------------------------------------------------
EQ_BENCH_GEN_TASKS = (EvalTaskConfig("eq_bench", 0, task_alias="eq_bench_0shot"),)

# -----------------------------------------------------------------------------
# BBQ (Generative - uses generate_until when using bbq_generate variant)
# Note: "bbq" task name may resolve to generate variant
# See: lm_eval/tasks/bbq/bbq_generate.yaml
# -----------------------------------------------------------------------------
BBQ_GEN_TASKS = (EvalTaskConfig("bbq", 0, task_alias="bbq_0shot"),)

# -----------------------------------------------------------------------------
# Specialized Domain Tasks that use generate_until
# Note: These use custom classes or lack output_type (defaults to generate_until)
# See: fda/task.py, swde/task.py, fld/*.yaml, qasper/freeform.yaml
# -----------------------------------------------------------------------------
SPECIALIZED_GEN_TASKS = (
    EvalTaskConfig("fda", 0, task_alias="fda_0shot"),
    EvalTaskConfig("fld_default", 0, task_alias="fld_default_0shot"),
    EvalTaskConfig("fld_star", 0, task_alias="fld_star_0shot"),
    EvalTaskConfig("qasper_freeform", 0, task_alias="qasper_freeform_0shot"),
    EvalTaskConfig("swde", 0, task_alias="swde_0shot"),
)

# -----------------------------------------------------------------------------
# HLE (Humanity's Last Exam) - Loglikelihood variant
# See: lm_eval/tasks/hle/hle_loglikelihood.yaml
# -----------------------------------------------------------------------------
HLE_LOGPROB_TASKS = (EvalTaskConfig("hle_loglikelihood", 0, task_alias="hle_0shot"),)

# -----------------------------------------------------------------------------
# OpenAI SimpleQA Test Set - Logprob and Generative variants
# See: lm_eval/tasks/openai_simple_qa_test_set/
# -----------------------------------------------------------------------------
SIMPLEQA_LOGPROB_TASKS = (EvalTaskConfig("openai_simple_qa_test_set_logprob", 0, task_alias="simpleqa_logprob_0shot"),)

SIMPLEQA_GEN_TASKS = (EvalTaskConfig("openai_simple_qa_test_set_gen", 0, task_alias="simpleqa_gen_0shot"),)

# =============================================================================
# Marin Task Configurations (local, Grug-friendly)
# =============================================================================

# -----------------------------------------------------------------------------
# Core tasks (from DCLM paper subset)
# -----------------------------------------------------------------------------
CORE_TASKS = (
    EvalTaskConfig("agieval_lsat_ar", 3),
    EvalTaskConfig("arc_easy", 10),
    EvalTaskConfig("arc_challenge", 10),
    EvalTaskConfig("boolq", 10),
    EvalTaskConfig("commonsense_qa", 10),
    EvalTaskConfig("copa", 0),
    EvalTaskConfig("hellaswag", 0, task_alias="hellaswag_0shot"),
    EvalTaskConfig("hellaswag", 10, task_alias="hellaswag_10shot"),
    EvalTaskConfig("lambada_openai", 0),
    EvalTaskConfig("openbookqa", 0),
    EvalTaskConfig("piqa", 10),
    EvalTaskConfig("wsc273", 0),
    EvalTaskConfig("winogrande", 0),
)

MMLU_0_SHOT = EvalTaskConfig("mmlu", 0, task_alias="mmlu_0shot")
MMLU_5_SHOT = EvalTaskConfig("mmlu", 5, task_alias="mmlu_5shot")
MMLU_PRO_5_SHOT = EvalTaskConfig("leaderboard_mmlu_pro", 5, task_alias="mmlu_pro_5shot")

CORE_TASKS_PLUS_LEADERBOARD = (
    EvalTaskConfig("leaderboard_bbh", 3, task_alias="bbh_3shot"),
    EvalTaskConfig("leaderboard_gpqa", 0, task_alias="gpqa_0shot"),
    *CORE_TASKS,
)

# -----------------------------------------------------------------------------
# Marin base model suites
# -----------------------------------------------------------------------------
BASE_GENERATION_TASKS = (
    EvalTaskConfig(name="bbh_cot_fewshot", num_fewshot=3),
    EvalTaskConfig(name="gsm8k_cot", num_fewshot=8),
    EvalTaskConfig(name="nq_open", num_fewshot=0, task_alias="nq_open"),
    EvalTaskConfig(name="triviaqa", num_fewshot=0, task_alias="triviaqa"),
)

# Settings are chosen to compare to Olmo2
# ifeval removed: requires langdetect which isn't in runtime env
KEY_GENERATION_TASKS = (
    EvalTaskConfig(name="gsm8k_cot", num_fewshot=8),
    EvalTaskConfig(name="drop", num_fewshot=0),
    EvalTaskConfig(name="humaneval", num_fewshot=10),
    EvalTaskConfig(name="bbh_cot_fewshot", num_fewshot=3, task_alias="bbh"),
    EvalTaskConfig(name="minerva_math", num_fewshot=4, task_alias="math_4shot"),
)

KEY_MULTIPLE_CHOICE_TASKS = (
    EvalTaskConfig("mmlu", 0, task_alias="mmlu_0shot"),
    EvalTaskConfig("mmlu", 5, task_alias="mmlu_5shot"),
    EvalTaskConfig(name="truthfulqa_mc2", num_fewshot=6, task_alias="truthqa"),
)

# -----------------------------------------------------------------------------
# Additional task categories (logprob)
# -----------------------------------------------------------------------------
REASONING_TASKS = (
    EvalTaskConfig("anli_r1", 0, task_alias="anli_r1_0shot"),
    EvalTaskConfig("anli_r2", 0, task_alias="anli_r2_0shot"),
    EvalTaskConfig("anli_r3", 0, task_alias="anli_r3_0shot"),
    EvalTaskConfig("arc_easy", 25, task_alias="arc_easy_25shot"),
    EvalTaskConfig("arc_challenge", 25, task_alias="arc_challenge_25shot"),
    # NOTE: babi removed - it's a generative task (output_type: generate_until)
    EvalTaskConfig("commonsense_qa", 0, task_alias="commonsense_qa_0shot"),
    EvalTaskConfig("copal_id_standard", 0, task_alias="copal_id_standard_0shot"),
    EvalTaskConfig("copal_id_colloquial", 0, task_alias="copal_id_colloquial_0shot"),
    EvalTaskConfig("logiqa", 0, task_alias="logiqa_0shot"),
    EvalTaskConfig("logiqa2", 0, task_alias="logiqa2_0shot"),
    EvalTaskConfig("mastermind_24_easy", 0, task_alias="mastermind_24_easy_0shot"),
    EvalTaskConfig("mastermind_24_hard", 0, task_alias="mastermind_24_hard_0shot"),
    EvalTaskConfig("mastermind_35_easy", 0, task_alias="mastermind_35_easy_0shot"),
    EvalTaskConfig("mastermind_35_hard", 0, task_alias="mastermind_35_hard_0shot"),
    EvalTaskConfig("mastermind_46_easy", 0, task_alias="mastermind_46_easy_0shot"),
    EvalTaskConfig("mastermind_46_hard", 0, task_alias="mastermind_46_hard_0shot"),
    EvalTaskConfig("openbookqa", 0, task_alias="openbookqa_0shot"),
    EvalTaskConfig("piqa", 0, task_alias="piqa_0shot"),
    EvalTaskConfig("social_iqa", 0, task_alias="social_iqa_0shot"),
    EvalTaskConfig("winogrande", 5, task_alias="winogrande_5shot"),
    EvalTaskConfig("wsc273", 0, task_alias="wsc273_0shot"),
)

MATH_TASKS = (
    EvalTaskConfig("arithmetic_1dc", 0, task_alias="arithmetic_1dc_0shot"),
    EvalTaskConfig("arithmetic_2da", 0, task_alias="arithmetic_2da_0shot"),
    EvalTaskConfig("arithmetic_2dm", 0, task_alias="arithmetic_2dm_0shot"),
    EvalTaskConfig("arithmetic_2ds", 0, task_alias="arithmetic_2ds_0shot"),
    EvalTaskConfig("arithmetic_3da", 0, task_alias="arithmetic_3da_0shot"),
    EvalTaskConfig("arithmetic_3ds", 0, task_alias="arithmetic_3ds_0shot"),
    EvalTaskConfig("arithmetic_4da", 0, task_alias="arithmetic_4da_0shot"),
    EvalTaskConfig("arithmetic_4ds", 0, task_alias="arithmetic_4ds_0shot"),
    EvalTaskConfig("arithmetic_5da", 0, task_alias="arithmetic_5da_0shot"),
    EvalTaskConfig("arithmetic_5ds", 0, task_alias="arithmetic_5ds_0shot"),
    EvalTaskConfig("asdiv", 0, task_alias="asdiv_0shot"),
    EvalTaskConfig("mathqa", 0, task_alias="mathqa_0shot"),
)

LANGUAGE_TASKS = (
    EvalTaskConfig("cola", 0, task_alias="cola_0shot"),
    EvalTaskConfig("mnli", 0, task_alias="mnli_0shot"),
    EvalTaskConfig("mrpc", 0, task_alias="mrpc_0shot"),
    EvalTaskConfig("qnli", 0, task_alias="qnli_0shot"),
    EvalTaskConfig("qqp", 0, task_alias="qqp_0shot"),
    EvalTaskConfig("rte", 0, task_alias="rte_0shot"),
    EvalTaskConfig("sst2", 0, task_alias="sst2_0shot"),
    EvalTaskConfig("wnli", 0, task_alias="wnli_0shot"),
    EvalTaskConfig("lambada_openai", 0, task_alias="lambada_openai_0shot"),
    EvalTaskConfig("lambada_openai_cloze_yaml", 0, task_alias="lambada_openai_cloze_yaml_0shot"),
    # NOTE: lambada_standard removed - use lambada_openai (EleutherAI/lambada_openai) instead
    EvalTaskConfig("mutual", 0, task_alias="mutual_0shot"),
    EvalTaskConfig("mutual_plus", 0, task_alias="mutual_plus_0shot"),
    EvalTaskConfig("race", 0, task_alias="race_0shot"),
    EvalTaskConfig("swag", 0, task_alias="swag_0shot"),
)

CODE_TASKS = ()

MARIN_MEDICAL_TASKS = (
    EvalTaskConfig("careqa_en", 0, task_alias="careqa_en_0shot"),
    EvalTaskConfig("careqa_es", 0, task_alias="careqa_es_0shot"),
    EvalTaskConfig("med_concepts_qa", 0, task_alias="med_concepts_qa_0shot"),
    EvalTaskConfig("medmcqa", 0, task_alias="medmcqa_0shot"),
    EvalTaskConfig("medqa_4options", 0, task_alias="medqa_0shot"),
    EvalTaskConfig("pubmedqa", 0, task_alias="pubmedqa_0shot"),
    EvalTaskConfig("kormedmcqa_dentist", 0, task_alias="kormedmcqa_dentist_0shot"),
    EvalTaskConfig("kormedmcqa_doctor", 0, task_alias="kormedmcqa_doctor_0shot"),
    EvalTaskConfig("kormedmcqa_nurse", 0, task_alias="kormedmcqa_nurse_0shot"),
    EvalTaskConfig("kormedmcqa_pharm", 0, task_alias="kormedmcqa_pharm_0shot"),
)

KNOWLEDGE_TASKS = (
    EvalTaskConfig("cmmlu", 0, task_alias="cmmlu_0shot"),
    EvalTaskConfig("kmmlu", 0, task_alias="kmmlu_0shot"),
)

TRUTHFULNESS_TASKS = (
    EvalTaskConfig("truthfulqa_mc1", 0, task_alias="truthfulqa_0shot"),
    EvalTaskConfig("truthfulqa_mc2", 0, task_alias="truthfulqa_mc2_0shot"),
)

SPECIALIZED_TASKS = (
    EvalTaskConfig("haerae", 0, task_alias="haerae_0shot"),
    EvalTaskConfig("prost", 0, task_alias="prost_0shot"),
    EvalTaskConfig("qa4mre_2011", 0, task_alias="qa4mre_2011_0shot"),
    EvalTaskConfig("qa4mre_2012", 0, task_alias="qa4mre_2012_0shot"),
    EvalTaskConfig("qa4mre_2013", 0, task_alias="qa4mre_2013_0shot"),
    EvalTaskConfig("qasper_bool", 0, task_alias="qasper_bool_0shot"),
    EvalTaskConfig("sciq", 0, task_alias="sciq_0shot"),
    EvalTaskConfig("webqs", 0, task_alias="webqs_0shot"),
)

# -----------------------------------------------------------------------------
# Gen2MC tasks (low-noise logprob evaluation for QA)
# -----------------------------------------------------------------------------
GEN2MC_TASKS = (
    EvalTaskConfig("jeopardy_gen2mc", 0, task_alias="jeopardy_gen2mc"),
    EvalTaskConfig("naturalqs_gen2mc", 0, task_alias="naturalqs_gen2mc"),
    EvalTaskConfig("squad_gen2mc", 0, task_alias="squad_gen2mc"),
    EvalTaskConfig("coqa_gen2mc", 0, task_alias="coqa_gen2mc"),
    EvalTaskConfig("drop_gen2mc", 0, task_alias="drop_gen2mc"),
)

# -----------------------------------------------------------------------------
# Logprob variants of generative tasks (fast, deterministic)
# -----------------------------------------------------------------------------
LOGPROB_MATH_TASKS = (
    EvalTaskConfig("logprob_gsm8k", 5, task_alias="logprob_gsm8k_5shot"),
    EvalTaskConfig("logprob_gsm8k_cot", 8, task_alias="logprob_gsm8k_cot_8shot"),
    EvalTaskConfig("logprob_hendrycks_math_algebra", 0, task_alias="logprob_math_algebra"),
    EvalTaskConfig("logprob_hendrycks_math_counting_and_prob", 0, task_alias="logprob_math_counting"),
    EvalTaskConfig("logprob_hendrycks_math_geometry", 0, task_alias="logprob_math_geometry"),
    EvalTaskConfig("logprob_hendrycks_math_intermediate_algebra", 0, task_alias="logprob_math_intermediate"),
    EvalTaskConfig("logprob_hendrycks_math_num_theory", 0, task_alias="logprob_math_num_theory"),
    EvalTaskConfig("logprob_hendrycks_math_prealgebra", 0, task_alias="logprob_math_prealgebra"),
    EvalTaskConfig("logprob_hendrycks_math_precalc", 0, task_alias="logprob_math_precalc"),
)

LOGPROB_CODE_TASKS = (
    EvalTaskConfig("logprob_humaneval", 0, task_alias="logprob_humaneval"),
    EvalTaskConfig("logprob_mbpp", 0, task_alias="logprob_mbpp"),
)

LOGPROB_GENERATIVE_TASKS = (
    *LOGPROB_MATH_TASKS,
    *LOGPROB_CODE_TASKS,
)

LOGPROB_GENERATIVE_CORE = (
    EvalTaskConfig("logprob_gsm8k", 5, task_alias="logprob_gsm8k_5shot"),
    EvalTaskConfig("logprob_humaneval", 0, task_alias="logprob_humaneval"),
    EvalTaskConfig("logprob_hendrycks_math_algebra", 0, task_alias="logprob_math_algebra"),
)

# =============================================================================
# Task Suites (combined)
# =============================================================================


def _dedupe_tasks(tasks: tuple[EvalTaskConfig, ...]) -> tuple[EvalTaskConfig, ...]:
    """Deduplicate tasks by (name, num_fewshot) tuple, ignoring task_alias."""
    seen = set()
    result = []
    for task in tasks:
        key = (task.name, task.num_fewshot)
        if key not in seen:
            seen.add(key)
            result.append(task)
    return tuple(result)


def _remove_generative_from_logprob(
    logprob_tasks: tuple[EvalTaskConfig, ...],
    generative_tasks: tuple[EvalTaskConfig, ...],
) -> tuple[EvalTaskConfig, ...]:
    """Remove tasks from logprob suite if they also appear in generative suite.

    Tasks that require generation (like CoT tasks) should only run in generative mode.
    We match by (name, num_fewshot) to handle cases where task_alias differs.
    """
    generative_keys = {(t.name, t.num_fewshot) for t in generative_tasks}
    return tuple(t for t in logprob_tasks if (t.name, t.num_fewshot) not in generative_keys)


# -----------------------------------------------------------------------------
# Marin default base model tasks
# These are the standard tasks used for base model evaluation
# -----------------------------------------------------------------------------
MARIN_BASE_LOGPROB_TASKS = _dedupe_tasks(
    (
        *CORE_TASKS,
        *CORE_TASKS_PLUS_LEADERBOARD,
        MMLU_0_SHOT,
        MMLU_5_SHOT,
        MMLU_PRO_5_SHOT,
        *KEY_MULTIPLE_CHOICE_TASKS,
    )
)

MARIN_BASE_GENERATIVE_TASKS = _dedupe_tasks(
    (
        *BASE_GENERATION_TASKS,
        *KEY_GENERATION_TASKS,
    )
)

# -----------------------------------------------------------------------------
# Combined comprehensive suites (deduplicated)
# -----------------------------------------------------------------------------
# All generative tasks (define first for cross-suite dedup)
_ALL_GENERATIVE_TASKS_RAW = _dedupe_tasks(
    (
        *BASE_GENERATION_TASKS,
        *KEY_GENERATION_TASKS,
        *CODE_TASKS,
        # Additional generative suites
        *INSTRUCTION_FOLLOWING_GEN_TASKS,
        *MATH_GEN_TASKS,  # gsm8k, minerva_math require generation
        *AIME_GEN_TASKS,
        *REASONING_GEN_TASKS,  # BBH CoT tasks require generation
        *CODE_GEN_TASKS,
        *QA_GEN_TASKS,
        *PPL_GEN_TASKS,
        *JEOPARDY_GEN_TASKS,
        # Multilingual math (diverse subset)
        *MGSM_DIVERSE_TASKS,
        # Additional generative tasks
        *LEADERBOARD_GEN_TASKS,  # leaderboard_math_hard (ifeval removed: langdetect)
        *HENDRYCKS_MATH_GEN_TASKS,  # all hendrycks_math variants use generate_until
        *JSONSCHEMA_GEN_TASKS,  # all jsonschema_bench variants use generate_until
        *TRUTHFULQA_GEN_TASKS,  # truthfulqa_gen uses generate_until
        *SQUAD_COMPLETION_GEN_TASKS,  # squad_completion uses generate_until (custom class)
        *KORMEDMCQA_GEN_TASKS,  # all kormedmcqa variants use generate_until
        *EQ_BENCH_GEN_TASKS,  # eq_bench uses generate_until
        *BBQ_GEN_TASKS,  # bbq uses generate_until
        *SPECIALIZED_GEN_TASKS,  # fda, fld, qasper_freeform, swde use generate_until
        *SIMPLEQA_GEN_TASKS,  # openai_simple_qa_test_set_gen uses generate_until
    )
)

# All logprob tasks: Marin defaults + extended categories
# Remove any tasks that are also in generative suite (those require generation)
_ALL_LOGPROB_TASKS_RAW = _dedupe_tasks(
    (
        *CORE_TASKS,
        *CORE_TASKS_PLUS_LEADERBOARD,
        MMLU_0_SHOT,
        MMLU_5_SHOT,
        MMLU_PRO_5_SHOT,
        *KEY_MULTIPLE_CHOICE_TASKS,
        *REASONING_TASKS,
        *MATH_TASKS,
        *LANGUAGE_TASKS,
        *MARIN_MEDICAL_TASKS,
        *KNOWLEDGE_TASKS,
        *TRUTHFULNESS_TASKS,
        *SPECIALIZED_TASKS,
        # Additional logprob suites
        *AGI_EVAL_TASKS,
        *MEDICAL_MC_TASKS,
        # Multilingual (diverse subset)
        *DIVERSE_MULTILINGUAL_TASKS,
        # HLE and SimpleQA
        *HLE_LOGPROB_TASKS,
        *SIMPLEQA_LOGPROB_TASKS,
        # Logprob variants of generative tasks (for low-noise pretraining eval)
        # These score log-prob of full reasoning traces for math/code/QA tasks
        *LOGPROB_GENERATIVE_TASKS,
        # Gen2MC tasks - MCQ versions of generative QA benchmarks (OLMES-style)
        *GEN2MC_TASKS,
    )
)

# Final deduplicated suites - tasks only appear in one suite
ALL_GENERATIVE_TASKS = _ALL_GENERATIVE_TASKS_RAW
ALL_LOGPROB_TASKS = _remove_generative_from_logprob(_ALL_LOGPROB_TASKS_RAW, _ALL_GENERATIVE_TASKS_RAW)

# -----------------------------------------------------------------------------
# Core subset for quick evaluation (smaller, faster)
# -----------------------------------------------------------------------------
CORE_LOGPROB_TASKS = CORE_TASKS  # Use Marin's CORE_TASKS (already deduplicated)

CORE_GENERATIVE_TASKS = (
    EvalTaskConfig("gsm8k_cot", 8, task_alias="gsm8k_cot_8shot"),
    EvalTaskConfig("humaneval", 0, task_alias="humaneval_0shot"),
    EvalTaskConfig("drop", 0, task_alias="drop_0shot"),
)


# =============================================================================
# Evaluation Functions
# =============================================================================


def batched_eval(
    step: ExecutorStep | InputName | str,
    resource_config: ResourceConfig = ResourceConfig.with_tpu("v5p-8"),
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = None,
    run_generative: bool = True,
    discover_latest_checkpoint: bool = True,
    batch_size: int = 5,
) -> list[ExecutorStep]:
    """
    Run evaluation suite with tasks batched into separate executor steps.

    This isolates failures - if one batch fails (e.g., due to a dataset 404),
    other batches can still succeed. Useful for debugging broken tasks.

    Args:
        step: Model step to evaluate
        resource_config: TPU/GPU resources for evaluation
        max_eval_instances: Limit number of eval instances (for debugging)
        engine_kwargs: vLLM engine kwargs for generative tasks
        run_generative: Whether to run generative tasks (requires vLLM)
        discover_latest_checkpoint: Auto-discover latest HF checkpoint
        batch_size: Number of tasks per executor step

    Returns:
        List of evaluation ExecutorSteps (one per batch)
    """
    if engine_kwargs is None:
        engine_kwargs = DEFAULT_ENGINE_KWARGS

    name, model_path = extract_model_name_and_path(step)
    eval_jobs = []

    # Batch logprob tasks
    logprob_tasks = list(ALL_LOGPROB_TASKS)
    for i in range(0, len(logprob_tasks), batch_size):
        batch = tuple(logprob_tasks[i : i + batch_size])
        batch_names = [t.task_alias or t.name for t in batch]
        batch_id = f"vbatch{i // batch_size + 1}"

        logprob_eval = evaluate_levanter_lm_evaluation_harness(
            model_name=f"{name}_logprob_{batch_id}",
            model_path=model_path,
            evals=batch,
            resource_config=resource_config,
            max_eval_instances=max_eval_instances,
            discover_latest_checkpoint=discover_latest_checkpoint,
        )
        eval_jobs.append(logprob_eval)

    # Batch generative tasks
    if run_generative:
        gen_tasks = list(ALL_GENERATIVE_TASKS)
        for i in range(0, len(gen_tasks), batch_size):
            batch = tuple(gen_tasks[i : i + batch_size])
            batch_id = f"batch{i // batch_size + 1}"

            gen_eval = evaluate_lm_evaluation_harness(
                model_name=f"{name}_gen_{batch_id}",
                model_path=model_path,
                evals=batch,
                max_eval_instances=max_eval_instances,
                engine_kwargs=engine_kwargs,
                resource_config=resource_config,
                discover_latest_checkpoint=discover_latest_checkpoint,
            )
            eval_jobs.append(gen_eval)

    return eval_jobs


def logprob_generative_eval(
    step: ExecutorStep | InputName | str,
    resource_config: ResourceConfig = ResourceConfig.with_tpu("v5p-8"),
    max_eval_instances: int | None = None,
    discover_latest_checkpoint: bool = True,
    core_only: bool = False,
) -> list[ExecutorStep]:
    """
    Run logprob evaluation on generative benchmark reference answers.

    This evaluates log-probability of full reasoning traces (chain-of-thought solutions)
    for math, code, and QA tasks. Provides a low-noise, fast signal that correlates
    with generative benchmark performance - ideal for pretraining comparisons.

    Tasks include:
    - Math: GSM8K (full CoT solutions), Hendrycks Math (all 7 categories)
    - Code: HumanEval, MBPP (canonical solutions)
    - QA: NQ Open, TriviaQA, DROP

    Metrics:
    - bpb (bits-per-byte): Length-normalized, lower is better
    - logprob: Raw log-probability, higher is better

    Args:
        step: Model step to evaluate
        resource_config: TPU/GPU resources for evaluation
        max_eval_instances: Limit number of eval instances (for debugging)
        discover_latest_checkpoint: Auto-discover latest HF checkpoint
        core_only: If True, only run core subset (GSM8K, HumanEval, Math Algebra)

    Returns:
        List of evaluation ExecutorSteps

    NOTE: Requires custom task definitions in:
        submodules/lm-evaluation-harness/lm_eval/tasks/logprob_generative/
    """
    name, model_path = extract_model_name_and_path(step)

    tasks = LOGPROB_GENERATIVE_CORE if core_only else LOGPROB_GENERATIVE_TASKS

    eval_step = evaluate_levanter_lm_evaluation_harness(
        model_name=f"{name}_logprob_gen_bpb",
        model_path=model_path,
        evals=tasks,
        resource_config=resource_config,
        max_eval_instances=max_eval_instances,
        discover_latest_checkpoint=discover_latest_checkpoint,
    )

    return [eval_step]


def _require_model_step() -> ExecutorStep | InputName | str:
    if MODEL_STEP is None:
        raise RuntimeError("Set MODEL_STEP near the top of experiments/all_evals.py before running.")
    return MODEL_STEP


def build_steps(
    step: ExecutorStep | InputName | str,
    *,
    resource_config: ResourceConfig = RESOURCE_CONFIG,
    max_eval_instances: int | None = MAX_EVAL_INSTANCES,
    engine_kwargs: dict | None = ENGINE_KWARGS,
    discover_latest_checkpoint: bool = DISCOVER_LATEST_CHECKPOINT,
) -> list[ExecutorStep]:
    return batched_eval(
        step=step,
        resource_config=resource_config,
        max_eval_instances=max_eval_instances,
        engine_kwargs=engine_kwargs,
        run_generative=True,
        discover_latest_checkpoint=discover_latest_checkpoint,
    )


# =============================================================================
# Aggregation (all_evals)
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class AllEvalsAggregateConfig:
    """Configuration for aggregating all_evals outputs."""

    eval_runs: list[str]
    """Eval output roots (Executor resolves InputName to str at runtime)."""

    output_path: str


def _discover_result_files(eval_root: str) -> list[str]:
    root_result = os.path.join(eval_root, "results.json")
    candidates: list[str] = []
    if fsspec_exists(root_result):
        candidates.append(root_result)

    candidates.extend(fsspec_glob(os.path.join(eval_root, "**", "results_*.json")))
    if not candidates:
        candidates.extend(fsspec_glob(os.path.join(eval_root, "**", "results.json")))

    if not candidates:
        return []

    unique = sorted(set(candidates))
    unique.sort(key=lambda path: fsspec_mtime(path) if fsspec_exists(path) else 0)
    return unique


def _load_json(path: str) -> dict:
    with fsspec.open(path, "r") as f:
        return json.load(f)


def run_all_evals_aggregate(config: AllEvalsAggregateConfig) -> str:
    """Aggregate evaluation results into a single JSON output."""
    result_files: list[str] = []
    for root in config.eval_runs:
        files = _discover_result_files(root)
        if not files:
            logger.warning("No result files found under %s", root)
            continue
        result_files.extend(files)

    if not result_files:
        raise RuntimeError("No evaluation results found; cannot aggregate")

    aggregated: dict[str, object] = {"sources": result_files, "results": {}}
    results_out: dict[str, object] = aggregated["results"]  # type: ignore[assignment]

    for result_path in result_files:
        data = _load_json(result_path)
        results = data.get("results")
        if not isinstance(results, dict):
            logger.warning("Skipping %s: missing results", result_path)
            continue

        for task_name, metrics in results.items():
            if not isinstance(metrics, dict):
                continue
            entry = {"source": result_path, "metrics": metrics}
            existing = results_out.get(task_name)
            if existing is None:
                results_out[task_name] = entry
                continue
            if isinstance(existing, list):
                existing.append(entry)
            else:
                results_out[task_name] = [existing, entry]

    output_path = config.output_path
    fsspec_mkdirs(output_path, exist_ok=True)

    all_results_path = os.path.join(output_path, "all_results.json")
    with fsspec.open(all_results_path, "w") as f:
        json.dump(aggregated, f, indent=2, sort_keys=True)

    logger.info("Wrote aggregated results to %s", output_path)
    return output_path


def build_all_evals_aggregate_step(
    eval_steps: list[ExecutorStep | InputName],
    *,
    output_path: str | None = None,
) -> ExecutorStep:
    """Create an ExecutorStep that aggregates eval outputs."""
    inputs: list[InputName] = []
    for step in eval_steps:
        if isinstance(step, ExecutorStep):
            inputs.append(step.as_input_name())
        elif isinstance(step, InputName):
            inputs.append(step)
        else:
            raise TypeError(f"Unexpected eval step type: {type(step)}")

    return ExecutorStep(
        name="evaluation/all_evals/aggregate",
        fn=run_all_evals_aggregate,
        config=AllEvalsAggregateConfig(
            eval_runs=inputs,
            output_path=output_path or this_output_path(),
        ),
        description="Aggregate all_evals outputs into a consolidated JSON",
    )


def main() -> None:
    step = _require_model_step()
    steps = build_steps(step)

    # Print task counts for verification
    print(f"Logprob tasks: {len(ALL_LOGPROB_TASKS)}")
    print(f"Generative tasks: {len(ALL_GENERATIVE_TASKS)}")
    print(f"Logprob generative tasks: {len(LOGPROB_GENERATIVE_TASKS)}")
    print(f"Gen2MC tasks: {len(GEN2MC_TASKS)}")
    print(f"Total unique tasks: {len(ALL_LOGPROB_TASKS) + len(ALL_GENERATIVE_TASKS)}")
    print(f"Executor steps: {len(steps)}")

    # Steps are run with max parallelism by default (max_concurrent=None)
    # Both eval steps will start as soon as the model is ready
    executor_main(steps=[build_all_evals_aggregate_step(steps)])


if __name__ == "__main__":
    main()
