# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen Olmix log-linear baseline for the two-phase many-domain sweep."""

from __future__ import annotations

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.nextgen.import_sources import NamedWandbRunImportSource

OLMIX_LOGLINEAR_RUN_ID = 2
OLMIX_LOGLINEAR_RUN_NAME = "baseline_olmix_loglinear"
OLMIX_LOGLINEAR_SOURCE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_olmix_bpb"
OLMIX_LOGLINEAR_OBJECTIVE_METRIC = "lm_eval/mmlu_5shot/bpb"
OLMIX_LOGLINEAR_KL_LAMBDA = 0.05
OLMIX_LOGLINEAR_HUBER_DELTA = 0.02
OLMIX_LOGLINEAR_PREDICTED_BPB = 2.106809784018966
OLMIX_LOGLINEAR_REGULARIZED_OBJECTIVE = 2.113417080125139

PHASE_0_OLMIX_LOGLINEAR_WEIGHTS = {
    "dolma3_arxiv": 0.004181597070343052,
    "dolma3_cc/art_and_design_high": 0.013529081058519913,
    "dolma3_cc/art_and_design_low": 0.005727997560529426,
    "dolma3_cc/crime_and_law_high": 0.01958919445933106,
    "dolma3_cc/crime_and_law_low": 0.010201311865408383,
    "dolma3_cc/education_and_jobs_high": 0.0268565126359373,
    "dolma3_cc/education_and_jobs_low": 0.018448327627272373,
    "dolma3_cc/electronics_and_hardware_high": 0.015952673358409383,
    "dolma3_cc/electronics_and_hardware_low": 0.008449587608349,
    "dolma3_cc/entertainment_high": 0.06187217030140463,
    "dolma3_cc/entertainment_low": 0.019855782604524842,
    "dolma3_cc/finance_and_business_high": 0.08702631209445205,
    "dolma3_cc/finance_and_business_low": 0.033958024947687564,
    "dolma3_cc/food_and_dining_high": 0.02177623652436231,
    "dolma3_cc/food_and_dining_low": 0.011137018326723615,
    "dolma3_cc/games_high": 0.03918841647182723,
    "dolma3_cc/games_low": 0.012129059399440346,
    "dolma3_cc/health_high": 0.05634993608121278,
    "dolma3_cc/health_low": 0.026131315275197953,
    "dolma3_cc/history_and_geography_high": 0.019298698816523713,
    "dolma3_cc/history_and_geography_low": 0.004074688987317308,
    "dolma3_cc/industrial_high": 0.010208420395537704,
    "dolma3_cc/industrial_low": 0.006319724605806395,
    "dolma3_cc/literature_high": 0.03592577391412614,
    "dolma3_cc/literature_low": 0.008245001526946063,
    "dolma3_cc/science_math_and_technology_high": 0.03147191295155292,
    "dolma3_cc/science_math_and_technology_low": 0.018221466753903712,
    "dolma3_finemath_3plus": 0.004649992125541896,
    "dolma3_stack_edu": 0.020663775499114297,
    "dolma3_wikipedia": 0.000525925419678892,
    "dolmino_common_crawl_hq": 0.18912294667021817,
    "dolmino_olmocr_pdfs_hq": 0.0221487089399883,
    "dolmino_stack_edu_fim": 0.026695628870636458,
    "dolmino_stem_heavy_crawl": 0.000620773204421459,
    "dolmino_synth_code": 0.002468977831462413,
    "dolmino_synth_instruction": 0.003090195170052788,
    "dolmino_synth_math": 0.0035561556301803494,
    "dolmino_synth_qa": 0.09504420009618818,
    "dolmino_synth_thinking": 0.005286477319869687,
}

PHASE_1_OLMIX_LOGLINEAR_WEIGHTS = {
    "dolma3_arxiv": 0.006445978411083285,
    "dolma3_cc/art_and_design_high": 0.006038726254745851,
    "dolma3_cc/art_and_design_low": 0.004660244468631693,
    "dolma3_cc/crime_and_law_high": 0.016072079685609017,
    "dolma3_cc/crime_and_law_low": 0.011461010800718117,
    "dolma3_cc/education_and_jobs_high": 0.00695053374174437,
    "dolma3_cc/education_and_jobs_low": 0.010823123119864646,
    "dolma3_cc/electronics_and_hardware_high": 0.007972057918857035,
    "dolma3_cc/electronics_and_hardware_low": 0.0044209354697020125,
    "dolma3_cc/entertainment_high": 0.03272262549415415,
    "dolma3_cc/entertainment_low": 0.005817304657478063,
    "dolma3_cc/finance_and_business_high": 0.02058207852562888,
    "dolma3_cc/finance_and_business_low": 0.018484581336424482,
    "dolma3_cc/food_and_dining_high": 0.012339725028076343,
    "dolma3_cc/food_and_dining_low": 0.0031005620936607457,
    "dolma3_cc/games_high": 0.05144379109273224,
    "dolma3_cc/games_low": 0.009107369102170061,
    "dolma3_cc/health_high": 0.028387838313563993,
    "dolma3_cc/health_low": 0.009805617771249574,
    "dolma3_cc/history_and_geography_high": 0.005805521827135372,
    "dolma3_cc/history_and_geography_low": 0.005660454691059422,
    "dolma3_cc/industrial_high": 0.006334936236989001,
    "dolma3_cc/industrial_low": 0.0014513868627416767,
    "dolma3_cc/literature_high": 0.014709710718524307,
    "dolma3_cc/literature_low": 0.005704163640583256,
    "dolma3_cc/science_math_and_technology_high": 0.03125853393752279,
    "dolma3_cc/science_math_and_technology_low": 0.006246907513062085,
    "dolma3_finemath_3plus": 0.005047887229639024,
    "dolma3_stack_edu": 0.009370964937395766,
    "dolma3_wikipedia": 0.0003452626649451857,
    "dolmino_common_crawl_hq": 0.1300418456614081,
    "dolmino_olmocr_pdfs_hq": 0.02216943457810969,
    "dolmino_stack_edu_fim": 0.011220785708951502,
    "dolmino_stem_heavy_crawl": 0.00033357506078538255,
    "dolmino_synth_code": 0.000995501302884064,
    "dolmino_synth_instruction": 0.0031780026544459064,
    "dolmino_synth_math": 0.0033240279120405015,
    "dolmino_synth_qa": 0.4690148702069648,
    "dolmino_synth_thinking": 0.0011500433687174874,
}

OLMIX_LOGLINEAR_PHASE_WEIGHTS = {
    "phase_0": PHASE_0_OLMIX_LOGLINEAR_WEIGHTS,
    "phase_1": PHASE_1_OLMIX_LOGLINEAR_WEIGHTS,
}


def create_olmix_loglinear_weight_config(run_id: int = OLMIX_LOGLINEAR_RUN_ID) -> WeightConfig:
    """Return the frozen two-phase many-domain Olmix log-linear baseline."""
    phase_items = OLMIX_LOGLINEAR_PHASE_WEIGHTS.items()
    copied_phase_weights = {phase_name: dict(phase_weights) for phase_name, phase_weights in phase_items}
    return WeightConfig(run_id=run_id, phase_weights=copied_phase_weights)


def create_olmix_loglinear_import_source(
    *,
    local_run_id: int = 240,
    source_experiment: str = OLMIX_LOGLINEAR_SOURCE_EXPERIMENT,
) -> NamedWandbRunImportSource:
    """Return a named-run import source for the standalone Olmix baseline."""
    return NamedWandbRunImportSource(
        source_experiment=source_experiment,
        local_run_id=local_run_id,
        run_name=OLMIX_LOGLINEAR_RUN_NAME,
        phase_weights=OLMIX_LOGLINEAR_PHASE_WEIGHTS,
    )
