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
# TODO: add bigbench, lingoly tasks
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
# TODO: add minerva_math
MATH_TASKS = (
    EvalTaskConfig("gsm8k", 5, task_alias="gsm8k_5shot"), # included in core tasks
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
    EvalTaskConfig("mathqa", 0, task_alias="mathqa_0shot"), # skip to save time
    EvalTaskConfig("aime", 0, task_alias="aime_0shot"),
    EvalTaskConfig("aime24", 0, task_alias="aime24_0shot"),
    EvalTaskConfig("aime25", 0, task_alias="aime25_0shot"),
)

# Language Understanding and Generation Tasks
LANGUAGE_TASKS = (
    # EvalTaskConfig("blimp", 0, task_alias="blimp_0shot"),
    # EvalTaskConfig("blimp_nl", 0, task_alias="blimp_nl_0shot"),
    EvalTaskConfig("c4", 0, task_alias="c4_0shot"), # loglikelihood_rolling not implemented
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
    # EvalTaskConfig("lm_syneval", 0, task_alias="lm_syneval_0shot"),
    EvalTaskConfig("mutual", 0, task_alias="mutual_0shot"),
    EvalTaskConfig("mutual_plus", 0, task_alias="mutual_plus_0shot"),
    EvalTaskConfig("nq_open", 0, task_alias="nq_open_0shot"),
    EvalTaskConfig("race", 0, task_alias="race_0shot"),
    # EvalTaskConfig("scrolls", 0, task_alias="scrolls_0shot"),
    EvalTaskConfig("squad_completion", 0, task_alias="squad_completion_0shot"),
    EvalTaskConfig("squadv2", 0, task_alias="squadv2_0shot"),
    # EvalTaskConfig("storycloze_2016", 0, task_alias="storycloze_2016_0shot"), # have to download manually
    # EvalTaskConfig("storycloze_2018", 0, task_alias="storycloze_2018_0shot"),
    # TODO: super_glue
    EvalTaskConfig("swag", 0, task_alias="swag_0shot"),
    EvalTaskConfig("triviaqa", 0, task_alias="triviaqa_0shot"),
    EvalTaskConfig("wikitext", 0, task_alias="wikitext_0shot"), # loglikelihood_rolling not implemented
    # EvalTaskConfig("xnli", 0, task_alias="xnli_0shot"),
    # EvalTaskConfig("xquad", 0, task_alias="xquad_0shot"),
    # EvalTaskConfig("xstorycloze", 0, task_alias="xstorycloze_0shot"),
)

# Multilingual and Cross-lingual Tasks
# TODO: varify these tasks
MULTILINGUAL_TASKS = (
    EvalTaskConfig("belebele", 0, task_alias="belebele_0shot"),
    EvalTaskConfig("multiblimp", 0, task_alias="multiblimp_0shot"),
    EvalTaskConfig("mlqa", 0, task_alias="mlqa_0shot"),
    EvalTaskConfig("paws-x", 0, task_alias="paws_x_0shot"),
    EvalTaskConfig("translation", 0, task_alias="translation_0shot"),
    EvalTaskConfig("wmt2016", 0, task_alias="wmt2016_0shot"),
    EvalTaskConfig("xcopa", 0, task_alias="xcopa_0shot"),
    EvalTaskConfig("xwinograd", 0, task_alias="xwinograd_0shot"),
    EvalTaskConfig("mgsm", 0, task_alias="mgsm_0shot"), # multilingual gsm8k
    EvalTaskConfig("mmlu_prox", 0, task_alias="mmlu_prox_0shot"),
    EvalTaskConfig("truthfulqa_multi", 0, task_alias="truthfulqa_multi_0shot"),
    EvalTaskConfig("global_mmlu", 0, task_alias="global_mmlu_0shot"),
)

# Code Generation and Programming Tasks
# TODO: add humaneval
CODE_TASKS = (
    EvalTaskConfig("code2text_go", 0, task_alias="code2text_go_0shot"),
    EvalTaskConfig("code2text_java", 0, task_alias="code2text_java_0shot"),
    EvalTaskConfig("code2text_javascript", 0, task_alias="code2text_javascript_0shot"),
    EvalTaskConfig("code2text_php", 0, task_alias="code2text_php_0shot"),
    EvalTaskConfig("code2text_python", 0, task_alias="code2text_python_0shot"),
    EvalTaskConfig("code2text_ruby", 0, task_alias="code2text_ruby_0shot"),
    # EvalTaskConfig("mbpp", 0, task_alias="mbpp_0shot"),
    # EvalTaskConfig("mbpp_instruct", 3, task_alias="mbpp_instruct_3shot"),
    # EvalTaskConfig("mbpp_plus", 0, task_alias="mbpp_plus_0shot"),
    # EvalTaskConfig("mbpp_plus_instruct", 0, task_alias="mbpp_plus_instruct_0shot"),
    EvalTaskConfig("jsonschema_bench_easy", 2, task_alias="jsonschema_bench_easy_2shot"),
    EvalTaskConfig("jsonschema_bench_medium", 2, task_alias="jsonschema_bench_medium_2shot"),
    EvalTaskConfig("jsonschema_bench_hard", 2, task_alias="jsonschema_bench_hard_2shot"),
)

# Medical and Healthcare Tasks
# TODO: add mediqa, meqsum, mimic_repsum, mts_dialog, olaph
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
# TODO: add ceval, gpqa
KNOWLEDGE_TASKS = (
    EvalTaskConfig("agieval", 0, task_alias="agieval_0shot"),
    EvalTaskConfig("cmmlu", 0, task_alias="cmmlu_0shot"),
    EvalTaskConfig("kmmlu", 0, task_alias="kmmlu_0shot"),
    EvalTaskConfig("mmlu_pro", 5, task_alias="mmlu_pro_5shot"),
    EvalTaskConfig("tmmluplus", 0, task_alias="tmmluplus_0shot"),
)

# Emotional Intelligence and Ethics Tasks
EMOTIONAL_ETHICS_TASKS = (
    EvalTaskConfig("eq_bench", 0, task_alias="eq_bench_0shot"), # generation task
    EvalTaskConfig("ethics_cm", 0, task_alias="ethics_cm_0shot"),
    EvalTaskConfig("ethics_deontology", 0, task_alias="ethics_deontology_0shot"),
    EvalTaskConfig("ethics_justice", 0, task_alias="ethics_justice_0shot"),
    EvalTaskConfig("ethics_utilitarianism", 0, task_alias="ethics_utilitarianism_0shot"),
    EvalTaskConfig("ethics_virtue", 0, task_alias="ethics_virtue_0shot"),
    EvalTaskConfig("moral_stories", 0, task_alias="moral_stories_0shot"),
    EvalTaskConfig("histoires_morales", 0, task_alias="histoires_morales_0shot"),
)

# Cultural and Regional Specific Tasks
# TODO: verify these tasks
# TODO: add arabculture
CULTURAL_TASKS = (
    EvalTaskConfig("aclue", 0, task_alias="aclue_0shot"),
    EvalTaskConfig("arabic_leaderboard_complete", 0, task_alias="arabic_leaderboard_complete_0shot"),
    EvalTaskConfig("arabic_leaderboard_light", 0, task_alias="arabic_leaderboard_light_0shot"),
    EvalTaskConfig("AraDICE", 0, task_alias="AraDICE_0shot"),
    EvalTaskConfig("bertaqa", 0, task_alias="bertaqa_0shot"),
)

# Bias and Safety Evaluation Tasks
BIAS_SAFETY_TASKS = (
    EvalTaskConfig("bbq", 0, task_alias="bbq_0shot"),
    EvalTaskConfig("cabbq", 0, task_alias="cabbq_0shot"),
    EvalTaskConfig("crows_pairs_english_age", 0, task_alias="crows_pairs_english_age_0shot"), # TODO: add French
    EvalTaskConfig("crows_pairs_english_autre", 0, task_alias="crows_pairs_english_autre_0shot"),
    EvalTaskConfig("crows_pairs_english_disability", 0, task_alias="crows_pairs_english_disability_0shot"),
    EvalTaskConfig("crows_pairs_english_gender", 0, task_alias="crows_pairs_english_gender_0shot"),
    EvalTaskConfig("crows_pairs_english_nationality", 0, task_alias="crows_pairs_english_nationality_0shot"),
    EvalTaskConfig("crows_pairs_english_physical_appearance", 0, task_alias="crows_pairs_english_physical_appearance_0shot"),
    EvalTaskConfig("crows_pairs_english_race_color", 0, task_alias="crows_pairs_english_race_color_0shot"),
    EvalTaskConfig("crows_pairs_english_religion", 0, task_alias="crows_pairs_english_religion_0shot"),
    EvalTaskConfig("crows_pairs_english_sexual_orientation", 0, task_alias="crows_pairs_english_sexual_orientation_0shot"),
    EvalTaskConfig("crows_pairs_english_socioeconomic", 0, task_alias="crows_pairs_english_socioeconomic_0shot"),
    # EvalTaskConfig("discrim_eval_explicit", 0, task_alias="discrim_eval_explicit_0shot"),
    # EvalTaskConfig("discrim_eval_implicit", 0, task_alias="discrim_eval_implicit_0shot"),
    
    # EvalTaskConfig("esbbq_age", 0, task_alias="esbbq_age_0shot"),
    # EvalTaskConfig("esbbq_disability_status", 0, task_alias="esbbq_disability_status_0shot"),
    # EvalTaskConfig("esbbq_gender", 0, task_alias="esbbq_gender_0shot"),
    # EvalTaskConfig("esbbq_lgbtqia", 0, task_alias="esbbq_lgbtqia_0shot"),
    # EvalTaskConfig("esbbq_nationality", 0, task_alias="esbbq_nationality_0shot"),
    # EvalTaskConfig("esbbq_physical_appearance", 0, task_alias="esbbq_physical_appearance_0shot"),
    # EvalTaskConfig("esbbq_race_ethnicity", 0, task_alias="esbbq_race_ethnicity_0shot"),
    # EvalTaskConfig("esbbq_religion", 0, task_alias="esbbq_religion_0shot"),
    # EvalTaskConfig("esbbq_ses", 0, task_alias="esbbq_ses_0shot"),
    # EvalTaskConfig("esbbq_spanish_region", 0, task_alias="esbbq_spanish_region_0shot"),

    EvalTaskConfig("realtoxicityprompts", 0, task_alias="realtoxicityprompts_0shot"),
    EvalTaskConfig("simple_cooccurrence_bias", 0, task_alias="simple_cooccurrence_bias_0shot"),
    EvalTaskConfig("toxigen", 0, task_alias="toxigen_0shot"),
    EvalTaskConfig("winogender_all", 0, task_alias="winogender_0shot"),
)

# Long Context Understanding Tasks
LONG_CONTEXT_TASKS = (
    EvalTaskConfig("babilong", 0, task_alias="babilong_0shot"),

    # TODO: verify longbench
    # EvalTaskConfig("longbench_2wikimqa", 0, task_alias="longbench_2wikimqa_0shot"),
    # EvalTaskConfig("longbench_2wikimqa_e", 0, task_alias="longbench_2wikimqa_e_0shot"),
    # EvalTaskConfig("longbench_dureader", 0, task_alias="longbench_dureader_0shot"),
    # EvalTaskConfig("longbench_gov_report", 0, task_alias="longbench_gov_report_0shot"),
    # EvalTaskConfig("longbench_gov_report_e", 0, task_alias="longbench_gov_report_e_0shot"),
    # EvalTaskConfig("longbench_hotpotqa", 0, task_alias="longbench_hotpotqa_0shot"),
    # EvalTaskConfig("longbench_hotpotqa_e", 0, task_alias="longbench_hotpotqa_e_0shot"),
    # EvalTaskConfig("longbench_lcc", 0, task_alias="longbench_lcc_0shot"),
    # EvalTaskConfig("longbench_lcc_e", 0, task_alias="longbench_lcc_e_0shot"),
    # EvalTaskConfig("longbench_lsht", 0, task_alias="longbench_lsht_0shot"),
    # EvalTaskConfig("longbench_multi_news", 0, task_alias="longbench_multi_news_0shot"),
    # EvalTaskConfig("longbench_multi_news_e", 0, task_alias="longbench_multi_news_e_0shot"),
    # EvalTaskConfig("longbench_multifieldqa_en", 0, task_alias="longbench_multifieldqa_en_0shot"),
    # EvalTaskConfig("longbench_multifieldqa_en_e", 0, task_alias="longbench_multifieldqa_en_e_0shot"),
    # EvalTaskConfig("longbench_multifieldqa_zh", 0, task_alias="longbench_multifieldqa_zh_0shot"),
    # EvalTaskConfig("longbench_musique", 0, task_alias="longbench_musique_0shot"),
    # EvalTaskConfig("longbench_narrativeqa", 0, task_alias="longbench_narrativeqa_0shot"),
    # EvalTaskConfig("longbench_passage_count", 0, task_alias="longbench_passage_count_0shot"),
    # EvalTaskConfig("longbench_passage_count_e", 0, task_alias="longbench_passage_count_e_0shot"),
    # EvalTaskConfig("longbench_passage_retrieval_en", 0, task_alias="longbench_passage_retrieval_en_0shot"),
    # EvalTaskConfig("longbench_passage_retrieval_en_e", 0, task_alias="longbench_passage_retrieval_en_e_0shot"),
    # EvalTaskConfig("longbench_passage_retrieval_zh", 0, task_alias="longbench_passage_retrieval_zh_0shot"),
    # EvalTaskConfig("longbench_qasper", 0, task_alias="longbench_qasper_0shot"),
    # EvalTaskConfig("longbench_qasper_e", 0, task_alias="longbench_qasper_e_0shot"),
    # EvalTaskConfig("longbench_qmsum", 0, task_alias="longbench_qmsum_0shot"),
    # EvalTaskConfig("longbench_repobench-p", 0, task_alias="longbench_repobench-p_0shot"),
    # EvalTaskConfig("longbench_repobench-p_e", 0, task_alias="longbench_repobench-p_e_0shot"),
    # EvalTaskConfig("longbench_samsum", 0, task_alias="longbench_samsum_0shot"),
    # EvalTaskConfig("longbench_samsum_e", 0, task_alias="longbench_samsum_e_0shot"),
    # EvalTaskConfig("longbench_trec", 0, task_alias="longbench_trec_0shot"),
    # EvalTaskConfig("longbench_trec_e", 0, task_alias="longbench_trec_e_0shot"),
    # EvalTaskConfig("longbench_triviaqa", 0, task_alias="longbench_triviaqa_0shot"),
    # EvalTaskConfig("longbench_triviaqa_e", 0, task_alias="longbench_triviaqa_e_0shot"),
    # EvalTaskConfig("longbench_vcsum", 0, task_alias="longbench_vcsum_0shot"),

    # ruler
    # EvalTaskConfig("niah_single_1", 0, task_alias="niah_single_1_0shot"),
    # EvalTaskConfig("niah_single_2", 0, task_alias="niah_single_2_0shot"),
    # EvalTaskConfig("niah_single_3", 0, task_alias="niah_single_3_0shot"),
    # EvalTaskConfig("niah_multikey_1", 0, task_alias="niah_multikey_1_0shot"),
    # EvalTaskConfig("niah_multikey_2", 0, task_alias="niah_multikey_2_0shot"),
    # EvalTaskConfig("niah_multikey_3", 0, task_alias="niah_multikey_3_0shot"),
    # EvalTaskConfig("niah_multiquery", 0, task_alias="niah_multiquery_0shot"),
    # EvalTaskConfig("niah_multivalue", 0, task_alias="niah_multivalue_0shot"),
    # EvalTaskConfig("ruler_vt", 0, task_alias="ruler_vt_0shot"),
    # EvalTaskConfig("ruler_cwe", 0, task_alias="ruler_cwe_0shot"),
    # EvalTaskConfig("ruler_fwe", 0, task_alias="ruler_fwe_0shot"),
    # EvalTaskConfig("ruler_qa_hotpot", 0, task_alias="ruler_qa_hotpot_0shot"),
    # EvalTaskConfig("ruler_qa_squad", 0, task_alias="ruler_qa_squad_0shot"),
)

# Language Specific Benchmarks
# TODO: varify these tasks
LANGUAGE_SPECIFIC_TASKS = (
    # Spanish
    EvalTaskConfig("spanish_bench", 0, task_alias="spanish_bench_0shot"),
    EvalTaskConfig("headqa", 0, task_alias="headqa_0shot"),
    
    # Catalan
    EvalTaskConfig("catalan_bench", 0, task_alias="catalan_bench_0shot"),
    
    # Basque
    EvalTaskConfig("basque_bench", 0, task_alias="basque_bench_0shot"),
    EvalTaskConfig("basqueglue", 0, task_alias="basqueglue_0shot"),
    EvalTaskConfig("eus_exams", 0, task_alias="eus_exams_0shot"),
    EvalTaskConfig("eus_proficiency", 0, task_alias="eus_proficiency_0shot"),
    EvalTaskConfig("eus_reading", 0, task_alias="eus_reading_0shot"),
    EvalTaskConfig("eus_trivia", 0, task_alias="eus_trivia_0shot"),
    EvalTaskConfig("xnli_eu", 0, task_alias="xnli_eu_0shot"),
    
    # French
    EvalTaskConfig("french_bench", 0, task_alias="french_bench_0shot"),
    
    # Galician
    EvalTaskConfig("galician_bench", 0, task_alias="galician_bench_0shot"),
    
    # Italian
    EvalTaskConfig("evalita_LLM", 0, task_alias="evalita_LLM_0shot"),
    
    # Portuguese
    EvalTaskConfig("portuguese_bench", 0, task_alias="portuguese_bench_0shot"),
    
    # Norwegian
    EvalTaskConfig("noreval", 0, task_alias="noreval_0shot"),
    
    # Icelandic
    EvalTaskConfig("icelandic_winogrande", 0, task_alias="icelandic_winogrande_0shot"),
    
    # Polish
    EvalTaskConfig("polemo2", 0, task_alias="polemo2_0shot"),
    
    # Turkish
    EvalTaskConfig("turkishmmlu", 0, task_alias="turkishmmlu_0shot"),
    EvalTaskConfig("turblimp_core", 0, task_alias="turblimp_core_0shot"),
    
    # Arabic Dialects
    EvalTaskConfig("darija_bench", 0, task_alias="darija_bench_0shot"),
    EvalTaskConfig("darijahellaswag", 0, task_alias="darijahellaswag_0shot"),
    EvalTaskConfig("darijammlu", 0, task_alias="darijammlu_0shot"),
    EvalTaskConfig("egyhellaswag", 0, task_alias="egyhellaswag_0shot"),
    EvalTaskConfig("egymmlu", 0, task_alias="egymmlu_0shot"),
    
    # Korean
    EvalTaskConfig("kobest", 0, task_alias="kobest_0shot"),
    EvalTaskConfig("csatqa", 0, task_alias="csatqa_0shot"),
    EvalTaskConfig("hrm8k", 0, task_alias="hrm8k_0shot"), # Korean math benchmark
    EvalTaskConfig("kbl", 0, task_alias="kbl_0shot"),
    EvalTaskConfig("click", 0, task_alias="click_0shot"),
    
    # Japanese
    EvalTaskConfig("japanese_leaderboard", 0, task_alias="japanese_leaderboard_0shot"),
    
    # Chinese
    EvalTaskConfig("zhoblimp", 0, task_alias="zhoblimp_0shot"),

    # Russian
    EvalTaskConfig("ru_quality", 0, task_alias="ru_quality_0shot"),
    EvalTaskConfig("ru_sci_passage_count", 0, task_alias="ru_sci_passage_count_0shot"),
    EvalTaskConfig("ru_qasper", 0, task_alias="ru_qasper_0shot"),
    EvalTaskConfig("ru_gsm100", 0, task_alias="ru_gsm100_0shot"),
)

# Action, Planning and Temporal Reasoning Tasks
ACTION_TASKS = (
    EvalTaskConfig("acp_areach_bool", 0, task_alias="acp_areach_bool_0shot"), # TODO: add hard and gen
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

# Advanced and Specialized Evaluation Tasks
# TODO: verify these tasks
ADVANCED_TASKS = (
    EvalTaskConfig("inverse_scaling", 0, task_alias="inverse_scaling_0shot"), # TODO: verify this task
    EvalTaskConfig("okapi/arc_multilingual", 0, task_alias="okapi_arc_multilingual_0shot"),
    EvalTaskConfig("okapi/hellaswag_multilingual", 0, task_alias="okapi_hellaswag_multilingual_0shot"),
    EvalTaskConfig("okapi/mmlu_multilingual", 0, task_alias="okapi_mmlu_multilingual_0shot"),
    EvalTaskConfig("okapi/truthfulqa_multilingual", 0, task_alias="okapi_truthfulqa_multilingual_0shot"),
    EvalTaskConfig("paloma", 0, task_alias="paloma_0shot"),
    EvalTaskConfig("pile", 0, task_alias="pile_0shot"),
    EvalTaskConfig("pile_10k", 0, task_alias="pile_10k_0shot"),
)


LM_EVAL_HARNESS_TASKS = REASONING_TASKS + MATH_TASKS + LANGUAGE_TASKS + MULTILINGUAL_TASKS + CODE_TASKS + MEDICAL_TASKS + KNOWLEDGE_TASKS + EMOTIONAL_ETHICS_TASKS + CULTURAL_TASKS + BIAS_SAFETY_TASKS + LONG_CONTEXT_TASKS + LANGUAGE_SPECIFIC_TASKS + ACTION_TASKS + TRUTHFULNESS_TASKS + SPECIALIZED_TASKS

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
