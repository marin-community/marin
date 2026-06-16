# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch all eval jobs for finished Grug-MoE swarm candidates."""

import re
from concurrent.futures import ThreadPoolExecutor

import fsspec
from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, InputName, executor_main
from marin.execution.remote import remote
from marin.execution.types import this_output_path

from experiments.evals.perplexity_gap_registry import registered_perplexity_gap_bundles
from experiments.exp1337_eval_suite import LOGPROB_TASKS
from experiments.grug.moe.eval_logprob import GrugLogprobEvalConfig, run_grug_logprob_eval
from experiments.grug.moe.eval_perplexity import GrugPerplexityEvalConfig, run_grug_perplexity_eval
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch_swarm import _BUDGET as _EVAL_BUDGET
from experiments.grug.moe.launch_swarm import _HIDDEN_DIM, _TARGET_STEPS

_OUTPUT_PREFIX = "gs://marin-us-central2/grug/"
_CANDIDATE_RE = re.compile(r"swarm_fisher_dsp_d512_(\d{6})-[a-f0-9]+")

# Logprob task surface — we keep growing this from the all_evals.py PR
# (marin-community/marin#2663) one eval at a time.
_EXTRA_LOGPROB_TASKS: tuple[EvalTaskConfig, ...] = (
    EvalTaskConfig("arc_easy", 0, task_alias="arc_easy_0shot"),
    EvalTaskConfig("arc_challenge", 0, task_alias="arc_challenge_0shot"),
    EvalTaskConfig("hellaswag", 0, task_alias="hellaswag_0shot"),
    EvalTaskConfig("winogrande", 0, task_alias="winogrande_0shot"),
    EvalTaskConfig("piqa", 0, task_alias="piqa_0shot"),
    # social_iqa skipped: HF script-loader deprecated in datasets>=4.0;
    # bye-fork doesn't host a parquet copy.
    EvalTaskConfig("sciq", 0, task_alias="sciq_0shot"),
    EvalTaskConfig("openbookqa", 0, task_alias="openbookqa_0shot"),
    EvalTaskConfig("boolq", 0, task_alias="boolq_0shot"),
    EvalTaskConfig("commonsense_qa", 0, task_alias="csqa_0shot"),
    EvalTaskConfig("lambada_openai", 0, task_alias="lambada_0shot"),
    # wsc273 skipped: same HF script-loader issue (winograd_wsc.py).
    EvalTaskConfig("copa", 0, task_alias="copa_0shot"),
    EvalTaskConfig("truthfulqa_mc1", 0, task_alias="truthfulqa_mc1_0shot"),
    EvalTaskConfig("truthfulqa_mc2", 0, task_alias="truthfulqa_mc2_0shot"),
    # logiqa skipped: same HF script-loader issue (logiqa.py).
    # logiqa2 keeps: uses parquet-hosted datasets.
    # Further tasks added one-at-a-time by the orchestrator
    # (`experiments/grug/moe/_mega_evals_orchestrator.py`).
    EvalTaskConfig("medmcqa", 0, task_alias="medmcqa_0shot"),
    EvalTaskConfig("medqa_4options", 0, task_alias="medqa_0shot"),
    # pubmedqa skipped: HF script-loader deprecated in datasets>=4.0
    # (pubmed_qa.py); no parquet-hosted copy, fails on every candidate.
    # bbh_zeroshot / bbh_fewshot skipped: generate_until tasks, unsupported by
    # the loglikelihood-only harness ("generation not supported"). The
    # multiple-choice leaderboard_bbh (lb_bbh_3shot) below works.
    EvalTaskConfig("leaderboard_gpqa", 0, task_alias="gpqa_0shot"),
    EvalTaskConfig("leaderboard_musr", 0, task_alias="musr_0shot"),
    EvalTaskConfig("leaderboard_bbh", 3, task_alias="lb_bbh_3shot"),
    EvalTaskConfig("leaderboard_mmlu_pro", 5, task_alias="mmlu_pro_5shot"),
    EvalTaskConfig("agieval_aqua_rat", 0, task_alias="agieval_aqua_rat_0shot"),
    EvalTaskConfig("belebele_spa_Latn", 0, task_alias="belebele_spanish"),
    EvalTaskConfig("belebele_fra_Latn", 0, task_alias="belebele_french"),
    EvalTaskConfig("belebele_por_Latn", 0, task_alias="belebele_portuguese"),
    EvalTaskConfig("belebele_deu_Latn", 0, task_alias="belebele_german"),
    EvalTaskConfig("belebele_rus_Cyrl", 0, task_alias="belebele_russian"),
    EvalTaskConfig("belebele_zho_Hans", 0, task_alias="belebele_chinese_simplified"),
    EvalTaskConfig("belebele_jpn_Jpan", 0, task_alias="belebele_japanese"),
    EvalTaskConfig("belebele_kor_Hang", 0, task_alias="belebele_korean"),
    EvalTaskConfig("belebele_hin_Deva", 0, task_alias="belebele_hindi"),
    EvalTaskConfig("belebele_ben_Beng", 0, task_alias="belebele_bengali"),
    EvalTaskConfig("belebele_arb_Arab", 0, task_alias="belebele_arabic"),
    EvalTaskConfig("belebele_heb_Hebr", 0, task_alias="belebele_hebrew"),
    EvalTaskConfig("belebele_tur_Latn", 0, task_alias="belebele_turkish"),
    EvalTaskConfig("belebele_vie_Latn", 0, task_alias="belebele_vietnamese"),
    EvalTaskConfig("belebele_ind_Latn", 0, task_alias="belebele_indonesian"),
    EvalTaskConfig("belebele_tha_Thai", 0, task_alias="belebele_thai"),
    EvalTaskConfig("belebele_swh_Latn", 0, task_alias="belebele_swahili"),
    EvalTaskConfig("belebele_yor_Latn", 0, task_alias="belebele_yoruba"),
    # Mega-evals batch (non-generative; generative/broken skipped).
    EvalTaskConfig("agieval_lsat_ar", 0, task_alias="agieval_lsat_ar_0shot"),
    EvalTaskConfig("agieval_lsat_lr", 0, task_alias="agieval_lsat_lr_0shot"),
    EvalTaskConfig("agieval_lsat_rc", 0, task_alias="agieval_lsat_rc_0shot"),
    EvalTaskConfig("agieval_sat_en", 0, task_alias="agieval_sat_en_0shot"),
    EvalTaskConfig("agieval_sat_math", 0, task_alias="agieval_sat_math_0shot"),
    EvalTaskConfig("include_base_44_spanish", 5, task_alias="include_spanish"),
    EvalTaskConfig("include_base_44_french", 5, task_alias="include_french"),
    EvalTaskConfig("include_base_44_portuguese", 5, task_alias="include_portuguese"),
    EvalTaskConfig("include_base_44_german", 5, task_alias="include_german"),
    EvalTaskConfig("include_base_44_russian", 5, task_alias="include_russian"),
    EvalTaskConfig("include_base_44_chinese", 5, task_alias="include_chinese"),
    EvalTaskConfig("include_base_44_japanese", 5, task_alias="include_japanese"),
    EvalTaskConfig("include_base_44_korean", 5, task_alias="include_korean"),
    EvalTaskConfig("include_base_44_hindi", 5, task_alias="include_hindi"),
    EvalTaskConfig("include_base_44_bengali", 5, task_alias="include_bengali"),
    EvalTaskConfig("include_base_44_arabic", 5, task_alias="include_arabic"),
    EvalTaskConfig("include_base_44_hebrew", 5, task_alias="include_hebrew"),
    EvalTaskConfig("include_base_44_turkish", 5, task_alias="include_turkish"),
    EvalTaskConfig("include_base_44_vietnamese", 5, task_alias="include_vietnamese"),
    EvalTaskConfig("include_base_44_indonesian", 5, task_alias="include_indonesian"),
    # hle skipped per request (do not eval Humanity's Last Exam).
    EvalTaskConfig("openai_simple_qa_test_set_logprob", 0, task_alias="simpleqa_logprob_0shot"),
    EvalTaskConfig("agieval_lsat_ar", 3, task_alias="agieval_lsat_ar_3shot"),
    EvalTaskConfig("arc_easy", 10, task_alias="arc_easy_10shot"),
    EvalTaskConfig("arc_challenge", 10, task_alias="arc_challenge_10shot"),
    EvalTaskConfig("boolq", 10, task_alias="boolq_10shot"),
    EvalTaskConfig("commonsense_qa", 10, task_alias="commonsense_qa_10shot"),
    EvalTaskConfig("hellaswag", 10, task_alias="hellaswag_10shot"),
    EvalTaskConfig("piqa", 10, task_alias="piqa_10shot"),
    EvalTaskConfig("truthfulqa_mc2", 6, task_alias="truthqa"),
    EvalTaskConfig("anli_r1", 0, task_alias="anli_r1_0shot"),
    EvalTaskConfig("anli_r2", 0, task_alias="anli_r2_0shot"),
    EvalTaskConfig("anli_r3", 0, task_alias="anli_r3_0shot"),
    EvalTaskConfig("arc_easy", 25, task_alias="arc_easy_25shot"),
    EvalTaskConfig("arc_challenge", 25, task_alias="arc_challenge_25shot"),
    EvalTaskConfig("copal_id_standard", 0, task_alias="copal_id_standard_0shot"),
    EvalTaskConfig("copal_id_colloquial", 0, task_alias="copal_id_colloquial_0shot"),
    EvalTaskConfig("mastermind_24_easy", 0, task_alias="mastermind_24_easy_0shot"),
    EvalTaskConfig("mastermind_24_hard", 0, task_alias="mastermind_24_hard_0shot"),
    EvalTaskConfig("mastermind_35_easy", 0, task_alias="mastermind_35_easy_0shot"),
    EvalTaskConfig("mastermind_35_hard", 0, task_alias="mastermind_35_hard_0shot"),
    EvalTaskConfig("mastermind_46_easy", 0, task_alias="mastermind_46_easy_0shot"),
    EvalTaskConfig("mastermind_46_hard", 0, task_alias="mastermind_46_hard_0shot"),
    EvalTaskConfig("winogrande", 5, task_alias="winogrande_5shot"),
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
    EvalTaskConfig("cola", 0, task_alias="cola_0shot"),
    EvalTaskConfig("mnli", 0, task_alias="mnli_0shot"),
    EvalTaskConfig("mrpc", 0, task_alias="mrpc_0shot"),
    EvalTaskConfig("qnli", 0, task_alias="qnli_0shot"),
    EvalTaskConfig("qqp", 0, task_alias="qqp_0shot"),
    EvalTaskConfig("rte", 0, task_alias="rte_0shot"),
    EvalTaskConfig("sst2", 0, task_alias="sst2_0shot"),
    EvalTaskConfig("wnli", 0, task_alias="wnli_0shot"),
    EvalTaskConfig("lambada_openai_cloze_yaml", 0, task_alias="lambada_openai_cloze_yaml_0shot"),
    EvalTaskConfig("mutual", 0, task_alias="mutual_0shot"),
    EvalTaskConfig("mutual_plus", 0, task_alias="mutual_plus_0shot"),
    EvalTaskConfig("race", 0, task_alias="race_0shot"),
    EvalTaskConfig("swag", 0, task_alias="swag_0shot"),
    EvalTaskConfig("careqa_en", 0, task_alias="careqa_en_0shot"),
    EvalTaskConfig("careqa_es", 0, task_alias="careqa_es_0shot"),
    EvalTaskConfig("med_concepts_qa", 0, task_alias="med_concepts_qa_0shot"),
    EvalTaskConfig("kormedmcqa_dentist", 0, task_alias="kormedmcqa_dentist_0shot"),
    EvalTaskConfig("kormedmcqa_doctor", 0, task_alias="kormedmcqa_doctor_0shot"),
    EvalTaskConfig("kormedmcqa_nurse", 0, task_alias="kormedmcqa_nurse_0shot"),
    EvalTaskConfig("kormedmcqa_pharm", 0, task_alias="kormedmcqa_pharm_0shot"),
    EvalTaskConfig("cmmlu", 0, task_alias="cmmlu_0shot"),
    EvalTaskConfig("kmmlu", 0, task_alias="kmmlu_0shot"),
    EvalTaskConfig("haerae", 0, task_alias="haerae_0shot"),
    EvalTaskConfig("prost", 0, task_alias="prost_0shot"),
    EvalTaskConfig("qa4mre_2011", 0, task_alias="qa4mre_2011_0shot"),
    EvalTaskConfig("qa4mre_2012", 0, task_alias="qa4mre_2012_0shot"),
    EvalTaskConfig("qa4mre_2013", 0, task_alias="qa4mre_2013_0shot"),
    EvalTaskConfig("qasper_bool", 0, task_alias="qasper_bool_0shot"),
    EvalTaskConfig("logprob_gsm8k_cot", 8, task_alias="logprob_gsm8k_cot_8shot"),
    EvalTaskConfig("logprob_hendrycks_math_algebra", 0, task_alias="logprob_math_algebra"),
    EvalTaskConfig("logprob_hendrycks_math_counting_and_prob", 0, task_alias="logprob_math_counting"),
    EvalTaskConfig("logprob_hendrycks_math_geometry", 0, task_alias="logprob_math_geometry"),
    EvalTaskConfig("logprob_hendrycks_math_intermediate_algebra", 0, task_alias="logprob_math_intermediate"),
    EvalTaskConfig("logprob_hendrycks_math_num_theory", 0, task_alias="logprob_math_num_theory"),
    EvalTaskConfig("logprob_hendrycks_math_prealgebra", 0, task_alias="logprob_math_prealgebra"),
    EvalTaskConfig("logprob_hendrycks_math_precalc", 0, task_alias="logprob_math_precalc"),
    EvalTaskConfig("logprob_humaneval", 0, task_alias="logprob_humaneval"),
    EvalTaskConfig("logprob_mbpp", 0, task_alias="logprob_mbpp"),
)

_LOGPROB_TASKS: tuple[EvalTaskConfig, ...] = LOGPROB_TASKS + _EXTRA_LOGPROB_TASKS

_PPL_BUNDLES = tuple(b for b in registered_perplexity_gap_bundles() if b.key != "base_raw")
_PPL_DATASETS_SKIP = ("bio_chem/refseq/refseq_viral_gff",)


def _find_finished_checkpoints() -> dict[int, str]:
    """Returns {idx: checkpoint_subpath} for every candidate whose
    .executor_status reads SUCCESS, picking the highest-step attempt per idx.
    The path is the GCS-prefix-relative checkpoints/ dir used by the
    Levanter checkpointer."""
    fs = fsspec.filesystem("gs")
    cand_dirs = [d for d in fs.ls(_OUTPUT_PREFIX, detail=False) if "swarm_fisher_dsp_d512_" in d]

    def probe(d):
        try:
            with fs.open(f"gs://{d}/.executor_status", "rt") as f:
                return d, f.read().strip() == "SUCCESS"
        except FileNotFoundError:
            return d, False

    with ThreadPoolExecutor(max_workers=64) as ex:
        results = list(ex.map(probe, cand_dirs))

    by_idx: dict[int, str] = {}
    for d, ok in results:
        if not ok:
            continue
        m = _CANDIDATE_RE.search(d)
        if not m:
            continue
        idx = int(m.group(1))
        existing = by_idx.get(idx)
        if existing is None or d > existing:
            by_idx[idx] = d
    return {idx: f"{d.removeprefix('marin-us-central2/')}/checkpoints" for idx, d in by_idx.items()}


_MODEL, _, _, _ = build_from_heuristic(
    budget=_EVAL_BUDGET,
    hidden_dim=_HIDDEN_DIM,
    target_steps=_TARGET_STEPS,
)


def _build_logprob_step(idx: int, ckpt_subpath: str, task: EvalTaskConfig) -> ExecutorStep:
    task_key = task.task_alias or f"{task.name}_{task.num_fewshot}shot"
    slug = f"swarm_fisher_dsp_d{_HIDDEN_DIM}_{idx:06d}"
    return ExecutorStep(
        name=f"evaluation/grug_logprob/{slug}/{task_key}",
        fn=remote(
            run_grug_logprob_eval,
            resources=ResourceConfig.with_tpu("v4-8", zone="us-central2-b", preemptible=True),
            pip_dependency_groups=["eval", "tpu"],
        ),
        config=GrugLogprobEvalConfig(
            grug_model_config=_MODEL,
            checkpoint_path=InputName.hardcoded(ckpt_subpath),
            output_path=this_output_path(),
            wandb_run_name=f"{slug}_{task_key}",
            task=task,
            wandb_tags=("grug", "logprob_eval", "swarm_fisher_dsp", slug, task_key),
        ),
    )


def _build_ppl_step(idx: int, ckpt_subpath: str, bundle) -> ExecutorStep:
    slug = f"swarm_fisher_dsp_d{_HIDDEN_DIM}_{idx:06d}"
    datasets = {k: v for k, v in bundle.datasets().items() if k not in _PPL_DATASETS_SKIP}
    return ExecutorStep(
        name=f"evaluation/grug_ppl/{slug}/{bundle.key}",
        fn=remote(
            run_grug_perplexity_eval,
            resources=ResourceConfig.with_tpu("v4-8", zone="us-central2-b", preemptible=True),
            pip_dependency_groups=["eval", "tpu"],
        ),
        config=GrugPerplexityEvalConfig(
            grug_model_config=_MODEL,
            checkpoint_path=InputName.hardcoded(ckpt_subpath),
            output_path=this_output_path(),
            bundle_key=bundle.key,
            datasets=datasets,
            max_eval_length=bundle.max_eval_length,
            max_docs_per_dataset=bundle.max_docs_per_dataset,
            max_doc_bytes=bundle.max_doc_bytes,
        ),
    )


def _build_steps() -> list[ExecutorStep]:
    finished = _find_finished_checkpoints()
    print(
        "launch_swarm_evals: "
        f"{len(finished)} finished candidates, "
        f"{len(_LOGPROB_TASKS)} logprob tasks, "
        f"{len(_PPL_BUNDLES)} ppl bundles"
    )
    steps = [_build_logprob_step(idx, ckpt, task) for idx, ckpt in sorted(finished.items()) for task in _LOGPROB_TASKS]
    steps.extend(_build_ppl_step(idx, ckpt, bundle) for idx, ckpt in sorted(finished.items()) for bundle in _PPL_BUNDLES)
    return steps


swarm_eval_steps: list[ExecutorStep] = _build_steps()


if __name__ == "__main__":
    executor_main(
        steps=swarm_eval_steps,
        description=(
            "Logprob and PPL eval suites for finished Grug-MoE swarm candidates. " "v4-8 preemptible, us-central2-b."
        ),
        max_concurrent=240,
    )
