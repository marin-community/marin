# Copyright 2025 The Marin Authors
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

from collections.abc import Sequence

from levanter.eval_harness import TaskConfig

from marin.evaluation.evaluation_config import EvalTaskConfig

# tasks to run (corresponding to lm_eval_harness tasks)
# subset from from page 43 of the DCLM paper: https://arxiv.org/pdf/2406.11794
# TODO: add more once supported in lm-eval-harness and/or tested on our end
CORE_TASKS = (
    EvalTaskConfig("agieval_lsat_ar", 3),  # 3-shot tests in legal domain
    EvalTaskConfig("arc_easy", 10),  # 10-shot, four-way MCQ questions involving grade 3-9 basic science
    EvalTaskConfig("arc_challenge", 10),  # a (harder) version of arc_easy
    EvalTaskConfig("boolq", 10),  # answer yes/no questions based on a passage
    EvalTaskConfig("commonsense_qa", 10),  # 5-way multiple-choice questions based on common-sense, everyday scenarios
    EvalTaskConfig("copa", 0),  # use causal reasoning to predict the correct outcome of a given scenario
    EvalTaskConfig("hellaswag", 0, task_alias="hellaswag_0shot"),  # 4-way multiple choice commonsense reasoning dataset
    EvalTaskConfig("hellaswag", 10, task_alias="hellaswag_10shot"),  # 4-way MCQ commonsense reasoning dataset
    EvalTaskConfig("lambada_openai", 0),  # predict the endings of text passages
    EvalTaskConfig("openbookqa", 0),  # 4-way multiple choice question answering task that requires multi-step reasoning
    EvalTaskConfig("piqa", 10),  # answer questions based on a passage
    # (requires generation which is not supported in Levanter at the moment)
    # EvalTaskConfig("squadv2", 10),  # reading comprehension benchmark
    EvalTaskConfig("wsc273", 0),  # Winograd Schema Challenge
    EvalTaskConfig("winogrande", 0),  # Winograd challenge, extended to more domains
)

MMLU_0_SHOT = EvalTaskConfig("mmlu", 0, task_alias="mmlu_0shot")
MMLU_5_SHOT = EvalTaskConfig("mmlu", 5, task_alias="mmlu_5shot")
MMLU_PRO_5_SHOT = EvalTaskConfig("leaderboard_mmlu_pro", 5, task_alias="mmlu_5shot")

OPEN_LM_LEADERBOARD_MCQ = (
    EvalTaskConfig("leaderboard_bbh", 3, task_alias="lb_bbh_3shot"),
    EvalTaskConfig("leaderboard_mmlu_pro", 5, task_alias="lb_mmlu_pro_5shot"),
    EvalTaskConfig("leaderboard_gpqa", 0, task_alias="lb_gpqa_0shot"),
    EvalTaskConfig("leaderboard_musr", 0, task_alias="lb_musr_0shot"),
)


OPEN_LM_LEADERBOARD_GEN = (
    EvalTaskConfig("leaderboard_ifeval", 0, task_alias="lb_ifeval_0shot"),
    EvalTaskConfig("leaderboard_math_hard", 4, task_alias="lb_math_4shot"),
)

MMLU_TASKS = (
    MMLU_0_SHOT,
    MMLU_5_SHOT,
)

PUBMED_QA = EvalTaskConfig("pubmedqa", 0, task_alias="pubmedqa_0shot")
MEDMCQA = EvalTaskConfig("medmcqa", 0, task_alias="medmcqa_0shot")
HEADQA = EvalTaskConfig("headqa_en", 0, task_alias="headqa_en_0shot")

MEDICAL_TASKS = (
    PUBMED_QA,
    MEDMCQA,
    HEADQA,
)

CORE_TASKS_PLUS_LEADERBOARD = (
    EvalTaskConfig(
        "leaderboard_bbh",
        3,
        task_alias="bbh_3shot",
    ),
    EvalTaskConfig(
        "leaderboard_gpqa",
        0,
        task_alias="gpqa_0shot",
    ),
    *CORE_TASKS,
)

CORE_TASKS_PLUS_MMLU = CORE_TASKS + MMLU_TASKS

BASE_GENERATION_TASKS = (
    EvalTaskConfig(name="bbh_cot_fewshot", num_fewshot=3),
    EvalTaskConfig(name="gsm8k_cot", num_fewshot=8),
    EvalTaskConfig(name="nq_open", num_fewshot=0, task_alias="nq_open"),
    EvalTaskConfig(name="triviaqa", num_fewshot=0, task_alias="triviaqa"),
)

# Settings are chosen to compare to Olmo2
KEY_GENERATION_TASKS = (
    EvalTaskConfig(name="ifeval", num_fewshot=0),
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

# LM-Eval-Harness Tasks
# Reasoning and Logic Tasks
REASONING_TASKS = (
    EvalTaskConfig("anli_r1", 0, task_alias="anli_r1_0shot"),
    EvalTaskConfig("anli_r2", 0, task_alias="anli_r2_0shot"),
    EvalTaskConfig("anli_r3", 0, task_alias="anli_r3_0shot"),
    EvalTaskConfig("arc_easy", 25, task_alias="arc_easy_25shot"),
    EvalTaskConfig("arc_challenge", 25, task_alias="arc_challenge_25shot"),
    EvalTaskConfig("babi", 0, task_alias="babi_0shot"),
    EvalTaskConfig("bbh_zeroshot", 0, task_alias="bbh_zeroshot_0shot"),
    EvalTaskConfig("bbh_fewshot", 3, task_alias="bbh_fewshot_3shot"),
    EvalTaskConfig("bbh_cot_zeroshot", 0, task_alias="bbh_cot_zeroshot_0shot"),
    EvalTaskConfig("bbh_cot_fewshot", 3, task_alias="bbh_cot_fewshot_3shot"),
    EvalTaskConfig("commonsense_qa", 0, task_alias="commonsense_qa_0shot"),
    EvalTaskConfig("copal_id_standard", 0, task_alias="copal_id_standard_0shot"),
    EvalTaskConfig("copal_id_colloquial", 0, task_alias="copal_id_colloquial_0shot"),
    EvalTaskConfig("logiqa", 0, task_alias="logiqa_0shot"),
    EvalTaskConfig("logiqa2", 0, task_alias="logiqa2_0shot"),
    EvalTaskConfig("logieval", 1, task_alias="logiqa2_1shot"),
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

# Mathematical and Arithmetic Tasks
MATH_TASKS = (
    EvalTaskConfig("gsm8k", 5, task_alias="gsm8k_5shot"),  # included in core tasks
    EvalTaskConfig(name="gsm8k_cot", num_fewshot=8, task_alias="gsm8k_cot_8shot"),
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
    EvalTaskConfig("hendrycks_math_algebra", 0, task_alias="hendrycks_math_algebra_0shot"),
    EvalTaskConfig("hendrycks_math_counting_and_prob", 0, task_alias="hendrycks_math_counting_and_prob_0shot"),
    EvalTaskConfig("hendrycks_math_geometry", 0, task_alias="hendrycks_math_geometry_0shot"),
    EvalTaskConfig("hendrycks_math_intermediate_algebra", 0, task_alias="hendrycks_math_intermediate_algebra_0shot"),
    EvalTaskConfig("hendrycks_math_num_theory", 0, task_alias="hendrycks_math_num_theory_0shot"),
    EvalTaskConfig("hendrycks_math_prealgebra", 0, task_alias="hendrycks_math_prealgebra_0shot"),
    EvalTaskConfig("hendrycks_math_precalc", 0, task_alias="hendrycks_math_precalc_0shot"),
    EvalTaskConfig("mathqa", 0, task_alias="mathqa_0shot"),
)

# Language Understanding and Generation Tasks
LANGUAGE_TASKS = (
    EvalTaskConfig("coqa", 0, task_alias="coqa_0shot"),
    EvalTaskConfig("drop", 0, task_alias="drop_0shot"),
    # glue
    EvalTaskConfig("cola", 0, task_alias="cola_0shot"),
    EvalTaskConfig("mnli", 0, task_alias="mnli_0shot"),
    EvalTaskConfig("mrpc", 0, task_alias="mrpc_0shot"),
    EvalTaskConfig("qnli", 0, task_alias="qnli_0shot"),
    EvalTaskConfig("qqp", 0, task_alias="qqp_0shot"),
    EvalTaskConfig("rte", 0, task_alias="rte_0shot"),
    EvalTaskConfig("sst2", 0, task_alias="sst2_0shot"),
    EvalTaskConfig("wnli", 0, task_alias="wnli_0shot"),
    EvalTaskConfig("lambada_openai", 0, task_alias="lambada_openai_0shot"),
    EvalTaskConfig("lambada_standard", 0, task_alias="lambada_standard_0shot"),
    EvalTaskConfig("lambada_openai_cloze_yaml", 0, task_alias="lambada_openai_cloze_yaml_0shot"),
    EvalTaskConfig("lambada_standard_cloze_yaml", 0, task_alias="lambada_standard_cloze_yaml_0shot"),
    EvalTaskConfig("mutual", 0, task_alias="mutual_0shot"),
    EvalTaskConfig("mutual_plus", 0, task_alias="mutual_plus_0shot"),
    EvalTaskConfig("nq_open", 0, task_alias="nq_open_0shot"),
    EvalTaskConfig("race", 0, task_alias="race_0shot"),
    EvalTaskConfig("squad_completion", 0, task_alias="squad_completion_0shot"),
    EvalTaskConfig("squadv2", 0, task_alias="squadv2_0shot"),
    EvalTaskConfig("swag", 0, task_alias="swag_0shot"),
    EvalTaskConfig("triviaqa", 0, task_alias="triviaqa_0shot"),
    EvalTaskConfig("wikitext", 0, task_alias="wikitext_0shot"),
)

# Code Generation and Programming Tasks
CODE_TASKS = (
    EvalTaskConfig("code2text_go", 0, task_alias="code2text_go_0shot"),
    EvalTaskConfig("code2text_java", 0, task_alias="code2text_java_0shot"),
    EvalTaskConfig("code2text_javascript", 0, task_alias="code2text_javascript_0shot"),
    EvalTaskConfig("code2text_php", 0, task_alias="code2text_php_0shot"),
    EvalTaskConfig("code2text_python", 0, task_alias="code2text_python_0shot"),
    EvalTaskConfig("code2text_ruby", 0, task_alias="code2text_ruby_0shot"),
    EvalTaskConfig("jsonschema_bench_easy", 2, task_alias="jsonschema_bench_easy_2shot"),
    EvalTaskConfig("jsonschema_bench_medium", 2, task_alias="jsonschema_bench_medium_2shot"),
    EvalTaskConfig("jsonschema_bench_hard", 2, task_alias="jsonschema_bench_hard_2shot"),
    EvalTaskConfig("humaneval", 0, task_alias="humaneval_0shot"),
)

# Medical and Healthcare Tasks
MEDICAL_TASKS = (
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

# Comprehensive Knowledge Assessment Tasks
KNOWLEDGE_TASKS = (
    EvalTaskConfig("agieval", 0, task_alias="agieval_0shot"),
    EvalTaskConfig("cmmlu", 0, task_alias="cmmlu_0shot"),
    EvalTaskConfig("kmmlu", 0, task_alias="kmmlu_0shot"),
    EvalTaskConfig("mmlu_pro", 5, task_alias="mmlu_pro_5shot"),
)

# Emotional Intelligence and Ethics Tasks
EMOTIONAL_ETHICS_TASKS = (
    EvalTaskConfig("eq_bench", 0, task_alias="eq_bench_0shot"),
    EvalTaskConfig("ethics_cm", 0, task_alias="ethics_cm_0shot"),
    EvalTaskConfig("ethics_deontology", 0, task_alias="ethics_deontology_0shot"),
    EvalTaskConfig("ethics_justice", 0, task_alias="ethics_justice_0shot"),
    EvalTaskConfig("ethics_utilitarianism", 0, task_alias="ethics_utilitarianism_0shot"),
    EvalTaskConfig("ethics_virtue", 0, task_alias="ethics_virtue_0shot"),
    EvalTaskConfig("moral_stories", 0, task_alias="moral_stories_0shot"),
    EvalTaskConfig("histoires_morales", 0, task_alias="histoires_morales_0shot"),
)

# Bias and Safety Evaluation Tasks
BIAS_SAFETY_TASKS = (
    EvalTaskConfig("bbq", 0, task_alias="bbq_0shot"),
    EvalTaskConfig("cabbq", 0, task_alias="cabbq_0shot"),
    EvalTaskConfig("crows_pairs_english_age", 0, task_alias="crows_pairs_english_age_0shot"),
    EvalTaskConfig("crows_pairs_english_autre", 0, task_alias="crows_pairs_english_autre_0shot"),
    EvalTaskConfig("crows_pairs_english_disability", 0, task_alias="crows_pairs_english_disability_0shot"),
    EvalTaskConfig("crows_pairs_english_gender", 0, task_alias="crows_pairs_english_gender_0shot"),
    EvalTaskConfig("crows_pairs_english_nationality", 0, task_alias="crows_pairs_english_nationality_0shot"),
    EvalTaskConfig(
        "crows_pairs_english_physical_appearance", 0, task_alias="crows_pairs_english_physical_appearance_0shot"
    ),
    EvalTaskConfig("crows_pairs_english_race_color", 0, task_alias="crows_pairs_english_race_color_0shot"),
    EvalTaskConfig("crows_pairs_english_religion", 0, task_alias="crows_pairs_english_religion_0shot"),
    EvalTaskConfig(
        "crows_pairs_english_sexual_orientation", 0, task_alias="crows_pairs_english_sexual_orientation_0shot"
    ),
    EvalTaskConfig("crows_pairs_english_socioeconomic", 0, task_alias="crows_pairs_english_socioeconomic_0shot"),
    EvalTaskConfig("realtoxicityprompts", 0, task_alias="realtoxicityprompts_0shot"),
    EvalTaskConfig("simple_cooccurrence_bias", 0, task_alias="simple_cooccurrence_bias_0shot"),
    EvalTaskConfig("toxigen", 0, task_alias="toxigen_0shot"),
    EvalTaskConfig("winogender_all", 0, task_alias="winogender_0shot"),
)

# Action, Planning and Temporal Reasoning Tasks
ACTION_TASKS = (
    EvalTaskConfig("acp_areach_bool", 0, task_alias="acp_areach_bool_0shot"),
    EvalTaskConfig("acp_app_bool", 0, task_alias="acp_app_bool_0shot"),
    EvalTaskConfig("acp_just_bool", 0, task_alias="acp_just_bool_0shot"),
    EvalTaskConfig("acp_land_bool", 0, task_alias="acp_land_bool_0shot"),
    EvalTaskConfig("acp_prog_bool", 0, task_alias="acp_prog_bool_0shot"),
    EvalTaskConfig("acp_reach_bool", 0, task_alias="acp_reach_bool_0shot"),
    EvalTaskConfig("acp_val_bool", 0, task_alias="acp_val_bool_0shot"),
    EvalTaskConfig("acp_areach_mcq", 0, task_alias="acp_areach_mcq_0shot"),
    EvalTaskConfig("acp_app_mcq", 0, task_alias="acp_app_mcq_0shot"),
    EvalTaskConfig("acp_just_mcq", 0, task_alias="acp_just_mcq_0shot"),
    EvalTaskConfig("acp_land_mcq", 0, task_alias="acp_land_mcq_0shot"),
    EvalTaskConfig("acp_prog_mcq", 0, task_alias="acp_prog_mcq_0shot"),
    EvalTaskConfig("acp_reach_mcq", 0, task_alias="acp_reach_mcq_0shot"),
    EvalTaskConfig("acp_val_mcq", 0, task_alias="acp_val_mcq_0shot"),
    EvalTaskConfig("mc_taco", 0, task_alias="mc_taco_0shot"),
    EvalTaskConfig("groundcocoa", 0, task_alias="groundcocoa_0shot"),
)

# Truthfulness and Factuality Tasks
TRUTHFULNESS_TASKS = (
    EvalTaskConfig("truthfulqa_mc1", 0, task_alias="truthfulqa_0shot"),
    EvalTaskConfig("truthfulqa_mc2", 0, task_alias="truthfulqa_mc2_0shot"),
    EvalTaskConfig("truthfulqa_gen", 0, task_alias="truthfulqa_gen_0shot"),
)

# Specialized Domain Tasks
SPECIALIZED_TASKS = (
    EvalTaskConfig("fda", 0, task_alias="fda_0shot"),
    EvalTaskConfig("fld_default", 0, task_alias="fld_default_0shot"),
    EvalTaskConfig("fld_star", 0, task_alias="fld_star_0shot"),
    EvalTaskConfig("haerae", 0, task_alias="haerae_0shot"),
    EvalTaskConfig("prost", 0, task_alias="prost_0shot"),
    EvalTaskConfig("qa4mre_2011", 0, task_alias="qa4mre_2011_0shot"),
    EvalTaskConfig("qa4mre_2012", 0, task_alias="qa4mre_2012_0shot"),
    EvalTaskConfig("qa4mre_2013", 0, task_alias="qa4mre_2013_0shot"),
    EvalTaskConfig("qasper_bool", 0, task_alias="qasper_bool_0shot"),
    EvalTaskConfig("qasper_freeform", 0, task_alias="qasper_freeform_0shot"),
    EvalTaskConfig("sciq", 0, task_alias="sciq_0shot"),
    EvalTaskConfig("swde", 0, task_alias="swde_0shot"),
    EvalTaskConfig("webqs", 0, task_alias="webqs_0shot"),
)

BELEBELE_TASKS = tuple(
    EvalTaskConfig(task_name, 0, task_alias=alias)
    for task_name, alias in (
        ("belebele_acm_Arab", "belebele_acm_Arab"),
        ("belebele_afr_Latn", "belebele_afr_Latn"),
        ("belebele_als_Latn", "belebele_als_Latn"),
        ("belebele_amh_Ethi", "belebele_amh_Ethi"),
        ("belebele_apc_Arab", "belebele_apc_Arab"),
        ("belebele_arb_Arab", "belebele_arb_Arab"),
        ("belebele_arb_Latn", "belebele_arb_Latn"),
        ("belebele_ars_Arab", "belebele_ars_Arab"),
        ("belebele_ary_Arab", "belebele_ary_Arab"),
        ("belebele_arz_Arab", "belebele_arz_Arab"),
        ("belebele_asm_Beng", "belebele_asm_Beng"),
        ("belebele_azj_Latn", "belebele_azj_Latn"),
        ("belebele_bam_Latn", "belebele_bam_Latn"),
        ("belebele_ben_Beng", "belebele_ben_Beng"),
        ("belebele_ben_Latn", "belebele_ben_Latn"),
        ("belebele_bod_Tibt", "belebele_bod_Tibt"),
        ("belebele_bul_Cyrl", "belebele_bul_Cyrl"),
        ("belebele_cat_Latn", "belebele_cat_Latn"),
        ("belebele_ceb_Latn", "belebele_ceb_Latn"),
        ("belebele_ces_Latn", "belebele_ces_Latn"),
        ("belebele_ckb_Arab", "belebele_ckb_Arab"),
        ("belebele_dan_Latn", "belebele_dan_Latn"),
        ("belebele_deu_Latn", "belebele_deu_Latn"),
        ("belebele_ell_Grek", "belebele_ell_Grek"),
        ("belebele_eng_Latn", "belebele_eng_Latn"),
        ("belebele_est_Latn", "belebele_est_Latn"),
        ("belebele_eus_Latn", "belebele_eus_Latn"),
        ("belebele_fin_Latn", "belebele_fin_Latn"),
        ("belebele_fra_Latn", "belebele_fra_Latn"),
        ("belebele_fuv_Latn", "belebele_fuv_Latn"),
        ("belebele_gaz_Latn", "belebele_gaz_Latn"),
        ("belebele_grn_Latn", "belebele_grn_Latn"),
        ("belebele_guj_Gujr", "belebele_guj_Gujr"),
        ("belebele_hat_Latn", "belebele_hat_Latn"),
        ("belebele_hau_Latn", "belebele_hau_Latn"),
        ("belebele_heb_Hebr", "belebele_heb_Hebr"),
        ("belebele_hin_Deva", "belebele_hin_Deva"),
        ("belebele_hin_Latn", "belebele_hin_Latn"),
        ("belebele_hrv_Latn", "belebele_hrv_Latn"),
        ("belebele_hun_Latn", "belebele_hun_Latn"),
        ("belebele_hye_Armn", "belebele_hye_Armn"),
        ("belebele_ibo_Latn", "belebele_ibo_Latn"),
        ("belebele_ilo_Latn", "belebele_ilo_Latn"),
        ("belebele_ind_Latn", "belebele_ind_Latn"),
        ("belebele_isl_Latn", "belebele_isl_Latn"),
        ("belebele_ita_Latn", "belebele_ita_Latn"),
        ("belebele_jav_Latn", "belebele_jav_Latn"),
        ("belebele_jpn_Jpan", "belebele_jpn_Jpan"),
        ("belebele_kac_Latn", "belebele_kac_Latn"),
        ("belebele_kan_Knda", "belebele_kan_Knda"),
        ("belebele_kat_Geor", "belebele_kat_Geor"),
        ("belebele_kaz_Cyrl", "belebele_kaz_Cyrl"),
        ("belebele_kea_Latn", "belebele_kea_Latn"),
        ("belebele_khk_Cyrl", "belebele_khk_Cyrl"),
        ("belebele_khm_Khmr", "belebele_khm_Khmr"),
        ("belebele_kin_Latn", "belebele_kin_Latn"),
        ("belebele_kir_Cyrl", "belebele_kir_Cyrl"),
        ("belebele_kor_Hang", "belebele_kor_Hang"),
        ("belebele_lao_Laoo", "belebele_lao_Laoo"),
        ("belebele_lin_Latn", "belebele_lin_Latn"),
        ("belebele_lit_Latn", "belebele_lit_Latn"),
        ("belebele_lug_Latn", "belebele_lug_Latn"),
        ("belebele_luo_Latn", "belebele_luo_Latn"),
        ("belebele_lvs_Latn", "belebele_lvs_Latn"),
        ("belebele_mal_Mlym", "belebele_mal_Mlym"),
        ("belebele_mar_Deva", "belebele_mar_Deva"),
        ("belebele_mkd_Cyrl", "belebele_mkd_Cyrl"),
        ("belebele_mlt_Latn", "belebele_mlt_Latn"),
        ("belebele_mri_Latn", "belebele_mri_Latn"),
        ("belebele_mya_Mymr", "belebele_mya_Mymr"),
        ("belebele_nld_Latn", "belebele_nld_Latn"),
        ("belebele_nob_Latn", "belebele_nob_Latn"),
        ("belebele_npi_Deva", "belebele_npi_Deva"),
        ("belebele_npi_Latn", "belebele_npi_Latn"),
        ("belebele_nso_Latn", "belebele_nso_Latn"),
        ("belebele_nya_Latn", "belebele_nya_Latn"),
        ("belebele_ory_Orya", "belebele_ory_Orya"),
        ("belebele_pan_Guru", "belebele_pan_Guru"),
        ("belebele_pbt_Arab", "belebele_pbt_Arab"),
        ("belebele_pes_Arab", "belebele_pes_Arab"),
        ("belebele_plt_Latn", "belebele_plt_Latn"),
        ("belebele_pol_Latn", "belebele_pol_Latn"),
        ("belebele_por_Latn", "belebele_por_Latn"),
        ("belebele_ron_Latn", "belebele_ron_Latn"),
        ("belebele_rus_Cyrl", "belebele_rus_Cyrl"),
        ("belebele_shn_Mymr", "belebele_shn_Mymr"),
        ("belebele_sin_Latn", "belebele_sin_Latn"),
        ("belebele_sin_Sinh", "belebele_sin_Sinh"),
        ("belebele_slk_Latn", "belebele_slk_Latn"),
        ("belebele_slv_Latn", "belebele_slv_Latn"),
        ("belebele_sna_Latn", "belebele_sna_Latn"),
        ("belebele_snd_Arab", "belebele_snd_Arab"),
        ("belebele_som_Latn", "belebele_som_Latn"),
        ("belebele_sot_Latn", "belebele_sot_Latn"),
        ("belebele_spa_Latn", "belebele_spa_Latn"),
        ("belebele_srp_Cyrl", "belebele_srp_Cyrl"),
        ("belebele_ssw_Latn", "belebele_ssw_Latn"),
        ("belebele_sun_Latn", "belebele_sun_Latn"),
        ("belebele_swe_Latn", "belebele_swe_Latn"),
        ("belebele_swh_Latn", "belebele_swh_Latn"),
        ("belebele_tam_Taml", "belebele_tam_Taml"),
        ("belebele_tel_Telu", "belebele_tel_Telu"),
        ("belebele_tgk_Cyrl", "belebele_tgk_Cyrl"),
        ("belebele_tgl_Latn", "belebele_tgl_Latn"),
        ("belebele_tha_Thai", "belebele_tha_Thai"),
        ("belebele_tir_Ethi", "belebele_tir_Ethi"),
        ("belebele_tsn_Latn", "belebele_tsn_Latn"),
        ("belebele_tso_Latn", "belebele_tso_Latn"),
        ("belebele_tur_Latn", "belebele_tur_Latn"),
        ("belebele_ukr_Cyrl", "belebele_ukr_Cyrl"),
        ("belebele_urd_Arab", "belebele_urd_Arab"),
        ("belebele_urd_Latn", "belebele_urd_Latn"),
        ("belebele_uzn_Latn", "belebele_uzn_Latn"),
        ("belebele_vie_Latn", "belebele_vie_Latn"),
        ("belebele_war_Latn", "belebele_war_Latn"),
        ("belebele_wol_Latn", "belebele_wol_Latn"),
        ("belebele_xho_Latn", "belebele_xho_Latn"),
        ("belebele_yor_Latn", "belebele_yor_Latn"),
        ("belebele_zho_Hans", "belebele_zho_Hans"),
        ("belebele_zho_Hant", "belebele_zho_Hant"),
        ("belebele_zsm_Latn", "belebele_zsm_Latn"),
        ("belebele_zul_Latn", "belebele_zul_Latn"),
    )
)

INCLUDE_44_TASKS = tuple(
    EvalTaskConfig(task_name, num_fewshot=5, task_alias=alias)
    for task_name, alias in (
        ("include_base_44_albanian_few_shot_og", "include_base_44_albanian_few_shot_og_5_shot"),
        ("include_base_44_arabic_few_shot_og", "include_base_44_arabic_few_shot_og_5_shot"),
        ("include_base_44_armenian_few_shot_og", "include_base_44_armenian_few_shot_og_5_shot"),
        ("include_base_44_azerbaijani_few_shot_og", "include_base_44_azerbaijani_few_shot_og_5_shot"),
        ("include_base_44_basque_few_shot_og", "include_base_44_basque_few_shot_og_5_shot"),
        ("include_base_44_belarusian_few_shot_og", "include_base_44_belarusian_few_shot_og_5_shot"),
        ("include_base_44_bengali_few_shot_og", "include_base_44_bengali_few_shot_og_5_shot"),
        ("include_base_44_bulgarian_few_shot_og", "include_base_44_bulgarian_few_shot_og_5_shot"),
        ("include_base_44_chinese_few_shot_og", "include_base_44_chinese_few_shot_og_5_shot"),
        ("include_base_44_croatian_few_shot_og", "include_base_44_croatian_few_shot_og_5_shot"),
        ("include_base_44_dutch_few_shot_og", "include_base_44_dutch_few_shot_og_5_shot"),
        # ("include_base_44_estonian_few_shot_og", "include_base_44_estonian_few_shot_og_5_shot"),
        # Disabled: Estonian test domains top out at 2 docs, so 5-shot sampling fails.
        ("include_base_44_finnish_few_shot_og", "include_base_44_finnish_few_shot_og_5_shot"),
        ("include_base_44_french_few_shot_og", "include_base_44_french_few_shot_og_5_shot"),
        ("include_base_44_georgian_few_shot_og", "include_base_44_georgian_few_shot_og_5_shot"),
        ("include_base_44_german_few_shot_og", "include_base_44_german_few_shot_og_5_shot"),
        ("include_base_44_greek_few_shot_og", "include_base_44_greek_few_shot_og_5_shot"),
        ("include_base_44_hebrew_few_shot_og", "include_base_44_hebrew_few_shot_og_5_shot"),
        ("include_base_44_hindi_few_shot_og", "include_base_44_hindi_few_shot_og_5_shot"),
        ("include_base_44_hungarian_few_shot_og", "include_base_44_hungarian_few_shot_og_5_shot"),
        ("include_base_44_indonesian_few_shot_og", "include_base_44_indonesian_few_shot_og_5_shot"),
        ("include_base_44_italian_few_shot_og", "include_base_44_italian_few_shot_og_5_shot"),
        ("include_base_44_japanese_few_shot_og", "include_base_44_japanese_few_shot_og_5_shot"),
        ("include_base_44_kazakh_few_shot_og", "include_base_44_kazakh_few_shot_og_5_shot"),
        ("include_base_44_korean_few_shot_og", "include_base_44_korean_few_shot_og_5_shot"),
        ("include_base_44_lithuanian_few_shot_og", "include_base_44_lithuanian_few_shot_og_5_shot"),
        ("include_base_44_malay_few_shot_og", "include_base_44_malay_few_shot_og_5_shot"),
        # ("include_base_44_malayalam_few_shot_og", "include_base_44_malayalam_few_shot_og_5_shot"),
        # Disabled: Malayalam domains have <=4 test docs, so 5-shot sampling fails.
        ("include_base_44_nepali_few_shot_og", "include_base_44_nepali_few_shot_og_5_shot"),
        ("include_base_44_north macedonian_few_shot_og", "include_base_44_north macedonian_few_shot_og_5_shot"),
        ("include_base_44_persian_few_shot_og", "include_base_44_persian_few_shot_og_5_shot"),
        # ("include_base_44_polish_few_shot_og", "include_base_44_polish_few_shot_og_5_shot"),
        # Disabled: Polish domains have <=4 test docs, so 5-shot sampling fails.
        ("include_base_44_portuguese_few_shot_og", "include_base_44_portuguese_few_shot_og_5_shot"),
        ("include_base_44_russian_few_shot_og", "include_base_44_russian_few_shot_og_5_shot"),
        ("include_base_44_serbian_few_shot_og", "include_base_44_serbian_few_shot_og_5_shot"),
        ("include_base_44_spanish_few_shot_og", "include_base_44_spanish_few_shot_og_5_shot"),
        ("include_base_44_tagalog_few_shot_og", "include_base_44_tagalog_few_shot_og_5_shot"),
        ("include_base_44_tamil_few_shot_og", "include_base_44_tamil_few_shot_og_5_shot"),
        ("include_base_44_telugu_few_shot_og", "include_base_44_telugu_few_shot_og_5_shot"),
        ("include_base_44_turkish_few_shot_og", "include_base_44_turkish_few_shot_og_5_shot"),
        ("include_base_44_ukrainian_few_shot_og", "include_base_44_ukrainian_few_shot_og_5_shot"),
        # ("include_base_44_urdu_few_shot_og", "include_base_44_urdu_few_shot_og_5_shot"),
        # Disabled: Urdu domains have <=3 test docs, so 5-shot sampling fails.
        ("include_base_44_uzbek_few_shot_og", "include_base_44_uzbek_few_shot_og_5_shot"),
        ("include_base_44_vietnamese_few_shot_og", "include_base_44_vietnamese_few_shot_og_5_shot"),
    )
)

MGSM_MULTILINGUAL_TASKS = (
    EvalTaskConfig("mgsm_direct_de", 0, task_alias="mgsm_direct_de_0shot"),
    EvalTaskConfig("mgsm_direct_es", 0, task_alias="mgsm_direct_es_0shot"),
    EvalTaskConfig("mgsm_direct_fr", 0, task_alias="mgsm_direct_fr_0shot"),
    EvalTaskConfig("mgsm_direct_ja", 0, task_alias="mgsm_direct_ja_0shot"),
    EvalTaskConfig("mgsm_direct_ru", 0, task_alias="mgsm_direct_ru_0shot"),
    EvalTaskConfig("mgsm_direct_zh", 0, task_alias="mgsm_direct_zh_0shot"),
)

XSTORYCLOZE_MULTILINGUAL_TASKS = (
    EvalTaskConfig("xstorycloze_ar", 0, task_alias="xstorycloze_ar_0shot"),
    EvalTaskConfig("xstorycloze_es", 0, task_alias="xstorycloze_es_0shot"),
    EvalTaskConfig("xstorycloze_id", 0, task_alias="xstorycloze_id_0shot"),
    EvalTaskConfig("xstorycloze_ru", 0, task_alias="xstorycloze_ru_0shot"),
    EvalTaskConfig("xstorycloze_zh", 0, task_alias="xstorycloze_zh_0shot"),
)

MMMLU_MULTILINGUAL_TASKS = tuple(
    EvalTaskConfig(f"mmmlu_{slug}", 5, task_alias=f"mmmlu_{slug}_5shot")
    for slug in (
        "ar_xy",
        "bn_bd",
        "de_de",
        "es_la",
        "fr_fr",
        "hi_in",
        "id_id",
        "it_it",
        "ja_jp",
        "ko_kr",
        "pt_br",
        "sw_ke",
        "yo_ng",
        "zh_cn",
    )
)

MULTILINGUAL_LM_EVAL_LOGPROB_TASKS = (
    INCLUDE_44_TASKS
    + BELEBELE_TASKS
    + XSTORYCLOZE_MULTILINGUAL_TASKS
    + (
        EvalTaskConfig("cmmlu", 0, task_alias="cmmlu_0shot"),
        EvalTaskConfig("kmmlu", 0, task_alias="kmmlu_0shot"),
        EvalTaskConfig("lm_syneval", 0, task_alias="lm_syneval_0shot"),
        EvalTaskConfig("zhoblimp", 0, task_alias="zhoblimp_0shot"),
        EvalTaskConfig("turblimp_core", 0, task_alias="turblimp_core_0shot"),
        EvalTaskConfig("blimp_nl", 0, task_alias="blimp_nl_0shot"),
    )
    + MMMLU_MULTILINGUAL_TASKS
)

MULTILINGUAL_LM_EVAL_GENERATIVE_TASKS = MGSM_MULTILINGUAL_TASKS


def convert_to_levanter_task_config(tasks: Sequence[EvalTaskConfig]) -> list[TaskConfig]:
    """
    Convert a list of EvalTaskConfig to a list of TaskConfig that Levanter's eval_harness expects.
    """
    return [
        TaskConfig(
            task=task.name,
            num_fewshot=task.num_fewshot,
            task_alias=task.task_alias,
        )
        for task in tasks
    ]


def convert_to_task_metrics(tasks: Sequence[EvalTaskConfig], metric: str) -> list[str]:
    """
    Convert a list of EvalTaskConfig to a list of strings corresponding to
    task metrics that Levanter outputs. These can be used, for instance, as input to the scaling laws analysis.
    """
    if not tasks:
        raise ValueError("Tasks sequence cannot be empty")

    task_metrics = []
    for task in tasks:
        if not isinstance(task, EvalTaskConfig):
            raise TypeError(f"Expected task to be EvalTaskConfig, got {type(task)}")

        if task.task_alias:
            task_metrics.append(f"lm_eval/{task.task_alias}/{metric}")
        else:
            task_metrics.append(f"lm_eval/{task.name}/{metric}")

    return task_metrics
