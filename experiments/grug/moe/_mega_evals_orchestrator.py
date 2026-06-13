#!/usr/bin/env python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Orchestrate adding mega-eval logprob tasks one-at-a-time to swarm_mmlu_eval.py.

For each task in ``_TASKS_TO_ADD``:
  1. Patch ``swarm_mmlu_eval.py`` so the task is appended to
     ``_EXTRA_LOGPROB_TASKS``.
  2. Submit the iris job.
  3. Poll until parent reaches a terminal state.
  4. Verify at least one candidate's ``results.json`` for the new task contains
     a real metric (any populated dict under ``results[<task_alias>]``).
  5. If parent succeeded OR counts climbed, move on. Otherwise, stop the loop
     so a human can investigate.

Designed to run as a single long-lived background bash invocation. Logs every
state change to stdout.
"""

import logging
import re
import subprocess
import time
from pathlib import Path

import fsspec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mega_evals")

_SCRIPT_PATH = Path("experiments/grug/moe/swarm_mmlu_eval.py")
_EXTRA_BLOCK_START = "_EXTRA_LOGPROB_TASKS: tuple[EvalTaskConfig, ...] = ("
_EXTRA_BLOCK_END = ")\n"

# Task names + alias derived from PR mega-evals. Order = priority.
# arc_challenge is also the first task — already patched into the file by the
# previous (broken) orchestrator run, so the patcher's idempotency check
# (``if new_line in text:``) silently skips re-patching it.
_TASKS_TO_ADD: list[tuple[str, int, str]] = [
    ('medmcqa', 0, 'medmcqa_0shot'),
    ('medqa_4options', 0, 'medqa_0shot'),
    ('pubmedqa', 0, 'pubmedqa_0shot'),
    ('bbh_zeroshot', 0, 'bbh_zeroshot'),
    ('bbh_fewshot', 3, 'bbh_3shot'),
    ('leaderboard_gpqa', 0, 'gpqa_0shot'),
    ('leaderboard_musr', 0, 'musr_0shot'),
    ('leaderboard_bbh', 3, 'lb_bbh_3shot'),
    ('leaderboard_mmlu_pro', 5, 'mmlu_pro_5shot'),
    ('agieval_aqua_rat', 0, 'agieval_aqua_rat_0shot'),
    ('agieval_lsat_ar', 0, 'agieval_lsat_ar_0shot'),
    ('agieval_lsat_lr', 0, 'agieval_lsat_lr_0shot'),
    ('agieval_lsat_rc', 0, 'agieval_lsat_rc_0shot'),
    ('agieval_sat_en', 0, 'agieval_sat_en_0shot'),
    ('agieval_sat_math', 0, 'agieval_sat_math_0shot'),
    ('belebele_spa_Latn', 0, 'belebele_spanish'),
    ('belebele_fra_Latn', 0, 'belebele_french'),
    ('belebele_por_Latn', 0, 'belebele_portuguese'),
    ('belebele_deu_Latn', 0, 'belebele_german'),
    ('belebele_rus_Cyrl', 0, 'belebele_russian'),
    ('belebele_zho_Hans', 0, 'belebele_chinese_simplified'),
    ('belebele_jpn_Jpan', 0, 'belebele_japanese'),
    ('belebele_kor_Hang', 0, 'belebele_korean'),
    ('belebele_hin_Deva', 0, 'belebele_hindi'),
    ('belebele_ben_Beng', 0, 'belebele_bengali'),
    ('belebele_arb_Arab', 0, 'belebele_arabic'),
    ('belebele_heb_Hebr', 0, 'belebele_hebrew'),
    ('belebele_tur_Latn', 0, 'belebele_turkish'),
    ('belebele_vie_Latn', 0, 'belebele_vietnamese'),
    ('belebele_ind_Latn', 0, 'belebele_indonesian'),
    ('belebele_tha_Thai', 0, 'belebele_thai'),
    ('belebele_swh_Latn', 0, 'belebele_swahili'),
    ('belebele_yor_Latn', 0, 'belebele_yoruba'),
    ('include_base_44_spanish', 5, 'include_spanish'),
    ('include_base_44_french', 5, 'include_french'),
    ('include_base_44_portuguese', 5, 'include_portuguese'),
    ('include_base_44_german', 5, 'include_german'),
    ('include_base_44_russian', 5, 'include_russian'),
    ('include_base_44_chinese', 5, 'include_chinese'),
    ('include_base_44_japanese', 5, 'include_japanese'),
    ('include_base_44_korean', 5, 'include_korean'),
    ('include_base_44_hindi', 5, 'include_hindi'),
    ('include_base_44_bengali', 5, 'include_bengali'),
    ('include_base_44_arabic', 5, 'include_arabic'),
    ('include_base_44_hebrew', 5, 'include_hebrew'),
    ('include_base_44_turkish', 5, 'include_turkish'),
    ('include_base_44_vietnamese', 5, 'include_vietnamese'),
    ('include_base_44_indonesian', 5, 'include_indonesian'),
    ('mgsm_direct_es', 0, 'mgsm_spanish'),
    ('mgsm_direct_fr', 0, 'mgsm_french'),
    ('mgsm_direct_de', 0, 'mgsm_german'),
    ('mgsm_direct_ru', 0, 'mgsm_russian'),
    ('mgsm_direct_zh', 0, 'mgsm_chinese'),
    ('mgsm_direct_ja', 0, 'mgsm_japanese'),
    ('hle_loglikelihood', 0, 'hle_0shot'),
    ('openai_simple_qa_test_set_logprob', 0, 'simpleqa_logprob_0shot'),
    ('openai_simple_qa_test_set_gen', 0, 'simpleqa_gen_0shot'),
    ('agieval_lsat_ar', 3, 'agieval_lsat_ar_3shot'),
    ('arc_easy', 10, 'arc_easy_10shot'),
    ('arc_challenge', 10, 'arc_challenge_10shot'),
    ('boolq', 10, 'boolq_10shot'),
    ('commonsense_qa', 10, 'commonsense_qa_10shot'),
    ('hellaswag', 10, 'hellaswag_10shot'),
    ('piqa', 10, 'piqa_10shot'),
    ('truthfulqa_mc2', 6, 'truthqa'),
    ('anli_r1', 0, 'anli_r1_0shot'),
    ('anli_r2', 0, 'anli_r2_0shot'),
    ('anli_r3', 0, 'anli_r3_0shot'),
    ('arc_easy', 25, 'arc_easy_25shot'),
    ('arc_challenge', 25, 'arc_challenge_25shot'),
    ('copal_id_standard', 0, 'copal_id_standard_0shot'),
    ('copal_id_colloquial', 0, 'copal_id_colloquial_0shot'),
    ('mastermind_24_easy', 0, 'mastermind_24_easy_0shot'),
    ('mastermind_24_hard', 0, 'mastermind_24_hard_0shot'),
    ('mastermind_35_easy', 0, 'mastermind_35_easy_0shot'),
    ('mastermind_35_hard', 0, 'mastermind_35_hard_0shot'),
    ('mastermind_46_easy', 0, 'mastermind_46_easy_0shot'),
    ('mastermind_46_hard', 0, 'mastermind_46_hard_0shot'),
    ('winogrande', 5, 'winogrande_5shot'),
    ('arithmetic_1dc', 0, 'arithmetic_1dc_0shot'),
    ('arithmetic_2da', 0, 'arithmetic_2da_0shot'),
    ('arithmetic_2dm', 0, 'arithmetic_2dm_0shot'),
    ('arithmetic_2ds', 0, 'arithmetic_2ds_0shot'),
    ('arithmetic_3da', 0, 'arithmetic_3da_0shot'),
    ('arithmetic_3ds', 0, 'arithmetic_3ds_0shot'),
    ('arithmetic_4da', 0, 'arithmetic_4da_0shot'),
    ('arithmetic_4ds', 0, 'arithmetic_4ds_0shot'),
    ('arithmetic_5da', 0, 'arithmetic_5da_0shot'),
    ('arithmetic_5ds', 0, 'arithmetic_5ds_0shot'),
    ('asdiv', 0, 'asdiv_0shot'),
    ('mathqa', 0, 'mathqa_0shot'),
    ('cola', 0, 'cola_0shot'),
    ('mnli', 0, 'mnli_0shot'),
    ('mrpc', 0, 'mrpc_0shot'),
    ('qnli', 0, 'qnli_0shot'),
    ('qqp', 0, 'qqp_0shot'),
    ('rte', 0, 'rte_0shot'),
    ('sst2', 0, 'sst2_0shot'),
    ('wnli', 0, 'wnli_0shot'),
    ('lambada_openai_cloze_yaml', 0, 'lambada_openai_cloze_yaml_0shot'),
    ('mutual', 0, 'mutual_0shot'),
    ('mutual_plus', 0, 'mutual_plus_0shot'),
    ('race', 0, 'race_0shot'),
    ('swag', 0, 'swag_0shot'),
    ('careqa_en', 0, 'careqa_en_0shot'),
    ('careqa_es', 0, 'careqa_es_0shot'),
    ('med_concepts_qa', 0, 'med_concepts_qa_0shot'),
    ('kormedmcqa_dentist', 0, 'kormedmcqa_dentist_0shot'),
    ('kormedmcqa_doctor', 0, 'kormedmcqa_doctor_0shot'),
    ('kormedmcqa_nurse', 0, 'kormedmcqa_nurse_0shot'),
    ('kormedmcqa_pharm', 0, 'kormedmcqa_pharm_0shot'),
    ('cmmlu', 0, 'cmmlu_0shot'),
    ('kmmlu', 0, 'kmmlu_0shot'),
    ('haerae', 0, 'haerae_0shot'),
    ('prost', 0, 'prost_0shot'),
    ('qa4mre_2011', 0, 'qa4mre_2011_0shot'),
    ('qa4mre_2012', 0, 'qa4mre_2012_0shot'),
    ('qa4mre_2013', 0, 'qa4mre_2013_0shot'),
    ('qasper_bool', 0, 'qasper_bool_0shot'),
    ('webqs', 0, 'webqs_0shot'),
    ('logprob_gsm8k_cot', 8, 'logprob_gsm8k_cot_8shot'),
    ('logprob_hendrycks_math_algebra', 0, 'logprob_math_algebra'),
    ('logprob_hendrycks_math_counting_and_prob', 0, 'logprob_math_counting'),
    ('logprob_hendrycks_math_geometry', 0, 'logprob_math_geometry'),
    ('logprob_hendrycks_math_intermediate_algebra', 0, 'logprob_math_intermediate'),
    ('logprob_hendrycks_math_num_theory', 0, 'logprob_math_num_theory'),
    ('logprob_hendrycks_math_prealgebra', 0, 'logprob_math_prealgebra'),
    ('logprob_hendrycks_math_precalc', 0, 'logprob_math_precalc'),
    ('logprob_humaneval', 0, 'logprob_humaneval'),
    ('logprob_mbpp', 0, 'logprob_mbpp'),
]

_PPL_PREFIX = "gs://marin-us-central2/evaluation/grug_logprob/"
_WANDB_KEY = "7c86993d6d6a1af7a92c1c22a44eb7aaccc50504"
_POLL_SECONDS = 300


def _add_task_to_script(task_name: str, num_fewshot: int, task_alias: str) -> None:
    """Append a new ``EvalTaskConfig`` line to ``_EXTRA_LOGPROB_TASKS``.

    Locate the tuple by its assignment, then walk a paren-depth counter from
    the opening ``(`` until depth returns to 0 — that's the tuple's closing
    paren, not the inner ``EvalTaskConfig(...)`` paren that a naive
    ``str.index(')')`` would find first.
    """
    text = _SCRIPT_PATH.read_text()
    new_line = f'    EvalTaskConfig("{task_name}", {num_fewshot}, task_alias="{task_alias}"),\n'
    if new_line in text:
        logger.info("task %s already in script, skipping patch", task_alias)
        return
    start = text.index(_EXTRA_BLOCK_START)
    paren_open = text.index("(", start)
    depth = 0
    paren_close = None
    for i in range(paren_open, len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                paren_close = i
                break
    if paren_close is None:
        raise RuntimeError(f"could not find matching ')' for _EXTRA_LOGPROB_TASKS tuple at offset {paren_open}")
    patched = text[:paren_close] + new_line + text[paren_close:]
    _SCRIPT_PATH.write_text(patched)
    logger.info("patched script: appended %s", task_alias)


def _submit_iris_job() -> str:
    out = subprocess.run(
        [
            ".venv/bin/iris", "--cluster=marin", "job", "run", "--no-wait",
            "--cpu=1", "--memory=8G", "--enable-extra-resources", "--extra=cpu",
            "--region", "us-central2", "--priority", "production",
            "-e", "WANDB_API_KEY", _WANDB_KEY,
            "--", "python", "experiments/grug/moe/swarm_mmlu_eval.py",
        ],
        capture_output=True, text=True, check=True,
    )
    m = re.search(r"Job submitted: (/held/iris-run-\S+)", out.stdout + out.stderr)
    if not m:
        raise RuntimeError(f"could not parse submitted job id; stdout={out.stdout!r} stderr={out.stderr!r}")
    return m.group(1)


def _poll_until_terminal(job_id: str) -> str:
    while True:
        result = subprocess.run(
            [".venv/bin/iris", "--cluster=marin", "job", "summary", job_id],
            capture_output=True, text=True,
        )
        text = result.stdout + result.stderr
        m = re.search(r"^State:\s+(\S+)", text, re.MULTILINE)
        state = m.group(1) if m else "unknown"
        logger.info("job %s state=%s", job_id, state)
        if state in ("succeeded", "failed", "killed", "cancelled"):
            return state
        time.sleep(_POLL_SECONDS)


def _count_real_results(task_alias: str) -> int:
    """Number of result.json files for this task that have a non-empty,
    error-free entry under ``results[<task_alias>]`` (or any task whose key
    contains the alias prefix — group tasks like mmlu_sl_verb publish under
    multiple sub-keys)."""
    fs = fsspec.filesystem("gs")
    paths = fs.glob(f"{_PPL_PREFIX}swarm_fisher_dsp_d512_*/{task_alias}*/results.json")
    n_real = 0
    for p in paths:
        try:
            import json
            with fs.open(f"gs://{p}", "rt") as f:
                blob = json.load(f)
        except Exception:  # pragma: no cover
            continue
        results = blob.get("results") or {}
        for k, v in results.items():
            if isinstance(v, dict) and any(
                isinstance(val, (int, float)) for key, val in v.items() if key not in ("alias",) and "stderr" not in key
            ):
                n_real += 1
                break
    return n_real


_INITIAL_JOB_TO_WAIT_ON = "/held/iris-run-swarm_mmlu_eval-20260612-162712"


def main() -> None:
    if _INITIAL_JOB_TO_WAIT_ON:
        logger.info("waiting on initial job %s before starting loop", _INITIAL_JOB_TO_WAIT_ON)
        state = _poll_until_terminal(_INITIAL_JOB_TO_WAIT_ON)
        logger.info("initial job %s reached state=%s", _INITIAL_JOB_TO_WAIT_ON, state)
        before = _count_real_results("arc_easy_0shot")
        logger.info("arc_easy_0shot real_results: %d", before)
    for task_name, num_fewshot, task_alias in _TASKS_TO_ADD:
        before = _count_real_results(task_alias)
        logger.info("=== adding %s (current real results: %d) ===", task_alias, before)
        _add_task_to_script(task_name, num_fewshot, task_alias)
        job_id = _submit_iris_job()
        logger.info("submitted %s", job_id)
        state = _poll_until_terminal(job_id)
        after = _count_real_results(task_alias)
        logger.info("after %s: state=%s real_results=%d (delta=+%d)", task_alias, state, after, after - before)
        if state != "succeeded" and after <= before:
            logger.warning("SKIPPING: %s neither succeeded nor produced new results — moving on", task_alias)
    logger.info("all tasks added")


if __name__ == "__main__":
    main()
