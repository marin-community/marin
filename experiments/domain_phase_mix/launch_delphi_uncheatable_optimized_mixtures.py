# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train selected Uncheatable-optimized mixtures on the Delphi scaling ladder.

This launcher is for issues #6602 and #6608.  It validates and submits three
two-phase Dolma3/Dolmino top-level mixtures:

- OLMix delta=0.01, KL=0.05, aggregate cap=4.
- DSP effective-exposure, KL=0.1.
- Canonical DSP, KL=0.1.

Unlike ``launch_delphi_baseline_mixtures.py``, this script intentionally accepts
phase-asymmetric mixtures and uses the historical 80/20 Dolma3/Dolmino
two-phase schedule.  The simulated-epoch target budget is fixed across Delphi
scales; each rung only changes the realized training token budget.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

import fsspec
import jmp
from fray.cluster import ResourceConfig
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import DatasetComponent
from levanter.main import train_lm
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.execution.types import ExecutorStep, this_output_path
from marin.processing.tokenize import step_to_lm_mixture_component
from marin.rl.placement import marin_prefix_for_region
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

from experiments.defaults import default_validation_sets
from experiments.domain_phase_mix.config import PhaseSchedule, WeightConfig
from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import (
    TOP_LEVEL_DOMAIN_TOKEN_COUNTS,
    TOP_LEVEL_TOTAL_AVAILABLE_TOKENS,
)
from experiments.domain_phase_mix.experiment import MixtureExperiment
from experiments.domain_phase_mix.launch_delphi_baseline_mixtures import (
    DEFAULT_TPU_REGION,
    DEFAULT_TPU_ZONE,
    LABEL,
    SEQ_LEN_DELPHI,
    SIMULATED_EPOCH_TARGET_BUDGET,
    TARGET_BUDGETS,
    _add_validation_components,
    _candidate_for_budget,
    _read_scaling_fits,
    _slug,
    _tensor_parallel_size,
)
from experiments.domain_phase_mix.qsplit240_replay import SKIP_EVAL_HARNESS_ENV_VAR
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    DEFAULT_RUNTIME_CACHE_REGION,
    DOMAIN_NAMES,
    PHASE_BOUNDARIES,
    PHASE_NAMES,
    build_top_level_domains,
)
from experiments.llama import llama3_tokenizer
from experiments.scaling_law_sweeps.completed_adamh import completed_adamh_heuristic

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUT_DIR = SCRIPT_DIR / "exploratory" / "two_phase_many" / "reference_outputs"
LOCAL_ARTIFACT_DIR = REFERENCE_OUTPUT_DIR / "delphi_uncheatable_optimized_mixtures_20260625"

EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_uncheatable_optimized_mixtures_20260625"
DEFAULT_ANALYSIS_OUTPUT_PATH = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/" "delphi_baseline_mixtures_issue6607_20260623/analysis-af9355"
)
DEFAULT_MAX_CONCURRENT = 4
RUN_ID_BASE = 662_000
PHASE_SCHEDULE = PhaseSchedule.from_boundaries(boundaries=PHASE_BOUNDARIES, names=list(PHASE_NAMES))
PHASE_FRACTIONS = {phase.name: phase.end_fraction - phase.start_fraction for phase in PHASE_SCHEDULE.phases}


class DelphiValidationMixture(StrEnum):
    """Selected mixtures for Uncheatable scaling validation."""

    OLMIX_D001_KL005_CAP4 = "olmix_d001_kl005_cap4"
    DSP_EFFECTIVE_EXPOSURE_KL01 = "dsp_effexp_kl01"
    DSP_CANONICAL_KL01 = "dsp_canon_kl01"


@dataclass(frozen=True)
class MixtureSource:
    """Source metadata for one selected mixture."""

    key: DelphiValidationMixture
    display_name: str
    source_csv: str
    github_issue: int
    target_metric: str
    method: str
    expected_max_simulated_epoch: float | None = None


MIXTURE_SOURCES: dict[DelphiValidationMixture, MixtureSource] = {
    DelphiValidationMixture.OLMIX_D001_KL005_CAP4: MixtureSource(
        key=DelphiValidationMixture.OLMIX_D001_KL005_CAP4,
        display_name="OLMix delta=0.01 KL=0.05 cap=4",
        source_csv=str(
            REFERENCE_OUTPUT_DIR
            / "olmix_huber_delta_sweep_300m_20260625"
            / "delta_0p01"
            / "uncheatable_eval_bpb_rep_cap4"
            / "proposed_mixture_weights.csv"
        ),
        github_issue=6608,
        target_metric="eval/uncheatable_eval/bpb",
        method="olmix_delta0p01_kl0p05_cap4",
        expected_max_simulated_epoch=4.0,
    ),
    DelphiValidationMixture.DSP_EFFECTIVE_EXPOSURE_KL01: MixtureSource(
        key=DelphiValidationMixture.DSP_EFFECTIVE_EXPOSURE_KL01,
        display_name="DSP effective-exposure KL=0.1",
        source_csv=str(
            REFERENCE_OUTPUT_DIR
            / "dsp_effective_exposure_l2_kl_sweep_deletion_augmented_300m_20260625"
            / "dsp_effective_exposure_l2_0.01_kl_only_0.1"
            / "proposed_mixture_weights.csv"
        ),
        github_issue=6602,
        target_metric="eval/uncheatable_eval/bpb",
        method="dsp_effective_exposure_l2_0p01_kl0p1",
    ),
    DelphiValidationMixture.DSP_CANONICAL_KL01: MixtureSource(
        key=DelphiValidationMixture.DSP_CANONICAL_KL01,
        display_name="DSP canonical KL=0.1",
        source_csv=str(
            REFERENCE_OUTPUT_DIR
            / "dsp_l2_kl_sweep_deletion_augmented_300m_20260625"
            / "dsp_l2_0.0001_kl_only_0.1"
            / "proposed_mixture_weights.csv"
        ),
        github_issue=6602,
        target_metric="eval/uncheatable_eval/bpb",
        method="dsp_canonical_l2_1e-4_kl0p1",
    ),
}


_EMBEDDED_MIXTURE_WEIGHT_CSVS: dict[DelphiValidationMixture, str] = {
    DelphiValidationMixture.OLMIX_D001_KL005_CAP4: (
        """domain,phase_0_weight,phase_1_weight,simulated_epochs
dolma3_arxiv,0.007877294593986481,0.03635606452360494,3.0403477019912675
dolma3_cc/art_and_design_high,0.019545498593021696,8.909429720890187e-05,0.8672650871365877
dolma3_cc/art_and_design_low,0.010559552793511558,1.4178515221360545e-05,1.1534396594560405
dolma3_cc/crime_and_law_high,0.0350226967748449,2.3302335359916474e-05,1.1534070479003278
dolma3_cc/crime_and_law_low,0.004444682047110862,0.00018463120527987726,0.3291871115125159
dolma3_cc/education_and_jobs_high,0.02769255808963606,5.397760991762897e-05,0.55797264751664
dolma3_cc/education_and_jobs_low,0.013209173864450583,7.917510118851106e-08,0.543094141834381
dolma3_cc/electronics_and_hardware_high,0.005247668930145996,4.396399977411061e-07,0.20030218024588944
dolma3_cc/electronics_and_hardware_low,0.010727686200166602,6.59459897955764e-06,0.8712681324978486
dolma3_cc/entertainment_high,0.07726800801307862,0.00028258350863736084,0.9242637707306016
dolma3_cc/entertainment_low,0.026996886657516674,2.367135322374188e-06,0.886595644868703
dolma3_cc/finance_and_business_high,0.07760905940328383,0.00018272102282068198,0.7223722035529303
dolma3_cc/finance_and_business_low,0.04349776302900368,2.6405194886169533e-06,0.8492025306231142
dolma3_cc/food_and_dining_high,0.03299884658156384,1.0566800465957603e-05,0.977481643559478
dolma3_cc/food_and_dining_low,0.007385770274324416,5.605031620699921e-07,0.4444490351412025
dolma3_cc/games_high,0.05614035870985765,2.7289288307112236e-05,1.257456981766967
dolma3_cc/games_low,0.009016072631240187,4.82044607840731e-07,0.5385800080589155
dolma3_cc/health_high,0.0481271848004615,1.3943237532411648e-05,0.5592788323776305
dolma3_cc/health_low,0.024205053021422954,7.316178873678304e-07,0.6599353173096475
dolma3_cc/history_and_geography_high,0.03309734432407341,0.00010039062286777329,1.2404876770179627
dolma3_cc/history_and_geography_low,0.006717721213536731,1.3899621783751634e-06,0.9142665882938764
dolma3_cc/industrial_high,0.014182816573017766,8.130294143831049e-06,0.8788443496954568
dolma3_cc/industrial_low,0.005080310894108908,2.0871428104730416e-06,0.5945394055875848
dolma3_cc/literature_high,0.015992248004829252,0.003890390398153202,0.33229225875028834
dolma3_cc/literature_low,0.01573398897331479,1.2379914229727515e-05,1.1560714601742572
dolma3_cc/science_math_and_technology_high,0.04563389575405995,0.00016872404620079144,0.9509964956876682
dolma3_cc/science_math_and_technology_low,0.036095959019950066,0.005142212704291994,1.704982311651281
dolma3_finemath_3plus,0.007230096389844306,1.4656763005626623e-05,1.076526296404441
dolma3_stack_edu,3.3034799816058055e-06,0.4239145709133125,4.000000010484327
dolma3_wikipedia,8.902715629914813e-05,1.1324309380465921e-09,0.12277867212100839
dolmino_common_crawl_hq,0.13784731244802997,0.00018506099690930963,0.529575772386458
dolmino_olmocr_pdfs_hq,0.04567876841609875,0.020512910853750344,1.2486055698761394
dolmino_stack_edu_fim,4.6230379254813127e-07,0.42333152777432126,4.000000013287558
dolmino_stem_heavy_crawl,0.00028873226847738267,1.4690930209578784e-09,0.2802260657207796
dolmino_synth_code,8.063119726239299e-05,0.0593146659964898,4.000000079432466
dolmino_synth_instruction,0.001855576000032917,2.0187990837630566e-05,0.5236260931662129
dolmino_synth_math,0.005530836836822305,0.0001429933298053315,1.2908608741494498
dolmino_synth_qa,0.08486734804427445,0.025985264500008917,0.8770001537214391
dolmino_synth_thinking,0.006421805693565501,2.0561625634680298e-07,0.8144671989820418
"""
    ),
    DelphiValidationMixture.DSP_EFFECTIVE_EXPOSURE_KL01: (
        """domain,phase_0_weight,phase_1_weight,simulated_epochs
dolma3_arxiv,0.0051155766373394505,0.01854371428944535,1.7474610088202231
dolma3_cc/art_and_design_high,0.013744238117693347,0.00025625142231538924,0.6119990177926778
dolma3_cc/art_and_design_low,0.00454817730617484,4.991236437238256e-05,0.49800173168972456
dolma3_cc/crime_and_law_high,0.022760137836410187,0.006323880119517936,0.8014955663615003
dolma3_cc/crime_and_law_low,0.01087821519164199,0.0067580402079638275,0.9212381894847661
dolma3_cc/education_and_jobs_high,0.03276850691384163,0.0016118800445175207,0.6680409201971444
dolma3_cc/education_and_jobs_low,0.014117712875959748,0.00019995379739571283,0.5825030499301324
dolma3_cc/electronics_and_hardware_high,0.020189632879773614,0.008137137830140672,0.8482634714354707
dolma3_cc/electronics_and_hardware_low,0.009987453748804097,0.0077078511292916685,0.9675018615867099
dolma3_cc/entertainment_high,0.06410116857390159,0.0235099969500146,0.8363055746207865
dolma3_cc/entertainment_low,0.021140174496557335,0.0024478805880340723,0.7143391835440304
dolma3_cc/finance_and_business_high,0.07497932883154632,0.008372862660610355,0.7169564156161192
dolma3_cc/finance_and_business_low,0.02164244042255889,3.7932770469705914e-05,0.42270195082035433
dolma3_cc/food_and_dining_high,0.02412643618074609,0.0036172256110543442,0.7413935756838379
dolma3_cc/food_and_dining_low,0.007588355223559377,4.149816510193049e-05,0.4572554944921937
dolma3_cc/games_high,0.028771343508736377,0.0013595628829366452,0.6519672770193224
dolma3_cc/games_low,0.011070316183926452,0.0007680319492347343,0.6727520747516524
dolma3_cc/health_high,0.04909793035645304,0.00043765841949835045,0.5717897981821503
dolma3_cc/health_low,0.026868032358364803,0.005635900866222897,0.7709487328962129
dolma3_cc/history_and_geography_high,0.021402040121816582,0.014236627225983303,0.9348361818275015
dolma3_cc/history_and_geography_low,0.005774037395029583,0.0027737987589755405,0.8801647634592228
dolma3_cc/industrial_high,0.013168842771057962,0.010993801738491061,0.9861806108566931
dolma3_cc/industrial_low,0.005126705545786675,0.00014245129173535816,0.6040745460066631
dolma3_cc/literature_high,0.04415072820495569,0.07522076646892732,1.2331243848179203
dolma3_cc/literature_low,0.01092155538625683,0.007224637671886842,0.9349983696728944
dolma3_cc/science_math_and_technology_high,0.041088297898680966,0.061000887642721775,1.1729933633720497
dolma3_cc/science_math_and_technology_low,0.01892633585969825,0.0317663475868531,1.2254542515636415
dolma3_finemath_3plus,0.005577482100055393,0.005452500762526859,1.0328993029550533
dolma3_stack_edu,0.025719911147963983,0.18843849413284333,2.748752498993329
dolma3_wikipedia,0.0006723503175560467,0.006323235595895515,3.107355423516416
dolmino_common_crawl_hq,0.1819513856350587,0.020889615818311297,0.7188348133954006
dolmino_olmocr_pdfs_hq,0.03947889697723864,0.2883745619268828,2.741947021652079
dolmino_stack_edu_fim,0.024221203355460918,0.08989333414011078,1.7648327142137648
dolmino_stem_heavy_crawl,0.00085811346240548,0.0017320372360831956,1.253084029308579
dolmino_synth_code,0.0037594709533817546,0.03947676543992924,3.656419735265803
dolmino_synth_instruction,0.003214634853798436,0.009694668245809166,1.5867599226477924
dolmino_synth_math,0.0039365087266317825,0.012379813853897277,1.630557900985739
dolmino_synth_qa,0.08130647543191534,0.03775423887508535,0.8710619314606202
dolmino_synth_thinking,0.005249846211261922,0.0004142435189125362,0.6789585025208178
"""
    ),
    DelphiValidationMixture.DSP_CANONICAL_KL01: (
        """domain,phase_0_weight,phase_1_weight,simulated_epochs
dolma3_arxiv,0.01201784024613365,0.014864854312305095,2.8195283679009804
dolma3_cc/art_and_design_high,0.03805705444901422,0.0004260914111440201,1.6914515311742888
dolma3_cc/art_and_design_low,0.002244637421547694,0.00012816217473236017,0.24860233243317767
dolma3_cc/crime_and_law_high,0.03387406944216076,0.00024217597476987662,1.117387193895774
dolma3_cc/crime_and_law_low,0.014848829196421486,0.015138242910614694,1.3658629596724516
dolma3_cc/education_and_jobs_high,0.00797480523038342,0.0010880972934410068,0.1660830362257602
dolma3_cc/education_and_jobs_low,0.00435527299378773,0.0001277902057498704,0.1803799564679302
dolma3_cc/electronics_and_hardware_high,0.007653558286399227,0.00013397519207277253,0.2934066657675398
dolma3_cc/electronics_and_hardware_low,0.021099445985763323,0.000321684805312015,1.7198963019410052
dolma3_cc/entertainment_high,0.05441391150355928,0.0012331296643950775,0.6539775228906396
dolma3_cc/entertainment_low,0.028235377326223145,0.0001572954003765224,0.9285395662912843
dolma3_cc/finance_and_business_high,0.06507237295082377,0.0023311505719335194,0.6107478209448866
dolma3_cc/finance_and_business_low,0.027005563667412944,0.00022172524661847055,0.5283011141224426
dolma3_cc/food_and_dining_high,0.02846915714900511,0.00044558263747524867,0.8465365408527974
dolma3_cc/food_and_dining_low,0.003925019383151708,0.00010882830328640503,0.2378262314586005
dolma3_cc/games_high,0.04285754910208404,0.0001838793224200761,0.9608555306382474
dolma3_cc/games_low,0.011952685064681202,0.002547706698356032,0.7520372606166003
dolma3_cc/health_high,0.05568377545211494,0.0008189167394229707,0.6494249346757164
dolma3_cc/health_low,0.03701380145143279,0.00010684299804476066,1.0098782861476365
dolma3_cc/history_and_geography_high,0.033185886301789154,0.0009565367474517036,1.2518197192569516
dolma3_cc/history_and_geography_low,0.002553570463915716,0.00010025378173587318,0.3509280614126178
dolma3_cc/industrial_high,0.02059080037354194,0.00022238609936002228,1.2791795716855532
dolma3_cc/industrial_low,0.0027558896007895753,0.000991184471151667,0.35147969520642125
dolma3_cc/literature_high,0.03232828779164631,0.005719783930591238,0.6612261622582951
dolma3_cc/literature_low,0.017667692868805245,0.0007281763332820511,1.3112703424013588
dolma3_cc/science_math_and_technology_high,0.019585117269499967,0.021150540461245017,0.5178619100200778
dolma3_cc/science_math_and_technology_low,0.004388158624537967,0.05409370184853902,0.8169528798431147
dolma3_finemath_3plus,0.007301562970828354,0.00023281146319178815,1.095278364584023
dolma3_stack_edu,4.009233804672937e-05,0.35681141891346563,3.3682320618991963
dolma3_wikipedia,0.0014565784189044444,7.391355717284336e-06,2.0113308943421986
dolmino_common_crawl_hq,0.20819611827251772,0.0015308309645392247,0.8010401552721311
dolmino_olmocr_pdfs_hq,0.055952616135912556,0.17414192199152193,2.444966265602824
dolmino_stack_edu_fim,6.353430208795893e-05,0.15954036014498924,1.5098691156414754
dolmino_stem_heavy_crawl,0.0029367994259667215,1.7487925431158873e-05,2.8545193373872895
dolmino_synth_code,3.743494675221391e-05,0.08376343255426603,5.6282445837793835
dolmino_synth_instruction,0.001167903871822112,0.025952096786692632,2.154570651299081
dolmino_synth_math,0.004192670318781768,0.0016058663867584472,1.0653554044950464
dolmino_synth_qa,0.07558406213063826,0.07173514165857404,0.8976781679963518
dolmino_synth_thinking,0.013260497271114668,7.254431902505372e-05,1.68409404743456
"""
    ),
}


@dataclass(frozen=True)
class MixtureDiagnostics:
    """Static diagnostics for one selected mixture."""

    mixture: str
    source_csv: str
    phase0_sum: float
    phase1_sum: float
    aggregate_sum: float
    max_simulated_epoch: float
    q95_simulated_epoch: float
    max_weight: float
    min_weight: float
    mean_phase_tv_to_proportional: float


@dataclass(frozen=True)
class DelphiOptimizedRunSpec:
    """One Delphi scaling validation training run."""

    run_order: int
    run_id: int
    run_name: str
    mixture: str
    mixture_display_name: str
    source_csv: str
    github_issue: int
    target_metric: str
    method: str
    target_flops: float
    tpu_type: str
    tpu_region: str
    tpu_zone: str
    batch_size: int
    train_tokens: int
    train_steps: int
    realized_train_tokens: int
    expected_checkpoint_step: int
    model_hidden_dim: int
    model_layers: int
    non_embedding_params: int
    total_trainable_params: int
    tensor_parallel_size: int
    data_seed: int
    trainer_seed: int
    phase_boundary: float
    phase_0_fraction: float
    phase_1_fraction: float
    simulated_epoch_target_budget: int
    available_top_level_tokens: int
    max_simulated_epoch: float
    q95_simulated_epoch: float
    mean_phase_tv_to_proportional: float
    phase_weights: dict[str, dict[str, float]]


@dataclass(frozen=True)
class DelphiOptimizedTrainingConfig:
    """Config resolved by the executor before one Delphi optimized-mixture train."""

    analysis_output_path: str
    target_flops: float
    tpu_type: str
    tpu_region: str
    tpu_zone: str
    batch_size: int
    mixture: DelphiValidationMixture
    label: str
    output_path: str
    run_id: int
    run_name: str
    data_seed: int
    trainer_seed: int = 0
    validation_configs: dict[str, DatasetComponent] | None = None


@dataclass(frozen=True)
class SaveDelphiOptimizedManifestConfig:
    """Config for persisting a resolved launcher manifest."""

    output_path: str
    analysis_output_path: str
    mixtures: tuple[DelphiValidationMixture, ...]
    target_budgets_json: str
    tpu_region: str
    tpu_zone: str
    run_order_offset: int = 0


@dataclass(frozen=True)
class LaunchArtifacts:
    """Resolved optimized-mixture launcher graph."""

    manifest_step: ExecutorStep
    training_steps: list[ExecutorStep]

    @property
    def steps(self) -> list[ExecutorStep]:
        return [self.manifest_step, *self.training_steps]


def _proportional_weights() -> dict[str, float]:
    total_tokens = float(TOP_LEVEL_TOTAL_AVAILABLE_TOKENS)
    return {domain_name: TOP_LEVEL_DOMAIN_TOKEN_COUNTS[domain_name] / total_tokens for domain_name in DOMAIN_NAMES}


def _phase_lengths() -> dict[str, float]:
    if set(PHASE_FRACTIONS) != set(PHASE_NAMES):
        raise ValueError(f"Phase names mismatch: {PHASE_FRACTIONS.keys()} vs {PHASE_NAMES}")
    if abs(sum(PHASE_FRACTIONS.values()) - 1.0) > 1e-12:
        raise ValueError(f"Phase fractions sum to {sum(PHASE_FRACTIONS.values())}, expected 1")
    if abs(PHASE_FRACTIONS["phase_0"] - 0.8) > 1e-12 or abs(PHASE_FRACTIONS["phase_1"] - 0.2) > 1e-12:
        raise ValueError(f"This launcher expects historical 80/20 phase fractions, got {PHASE_FRACTIONS}")
    return dict(PHASE_FRACTIONS)


def _q95(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute q95 of empty list")
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return ordered[index]


def _parse_weight_rows(csv_text: str, *, source_label: str) -> list[dict[str, str]]:
    required_columns = {"domain", "phase_0_weight", "phase_1_weight"}
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    missing = required_columns.difference(reader.fieldnames or [])
    if missing:
        raise ValueError(f"{source_label} missing required columns: {sorted(missing)}")
    rows = list(reader)
    if not rows:
        raise ValueError(f"{source_label} has no rows")
    return rows


def _validate_embedded_matches_local(
    embedded_rows: list[dict[str, str]],
    local_rows: list[dict[str, str]],
    *,
    source: MixtureSource,
) -> None:
    embedded_by_domain = {row["domain"]: row for row in embedded_rows}
    local_by_domain = {row["domain"]: row for row in local_rows}
    if set(embedded_by_domain) != set(local_by_domain):
        missing = sorted(set(embedded_by_domain).difference(local_by_domain))
        extra = sorted(set(local_by_domain).difference(embedded_by_domain))
        raise ValueError(f"{source.key.value} embedded/local domain mismatch: missing={missing}, extra={extra}")
    for domain, embedded in embedded_by_domain.items():
        local = local_by_domain[domain]
        for column in ["phase_0_weight", "phase_1_weight", "simulated_epochs"]:
            if column not in embedded or column not in local:
                continue
            embedded_value = float(embedded[column])
            local_value = float(local[column])
            if abs(embedded_value - local_value) > max(1e-12, 1e-12 * abs(local_value)):
                raise ValueError(
                    f"{source.key.value}/{domain}/{column} embedded={embedded_value} "
                    f"does not match local CSV {local_value}"
                )


def _read_phase_weight_rows(source: MixtureSource) -> list[dict[str, str]]:
    embedded_rows = _parse_weight_rows(
        _EMBEDDED_MIXTURE_WEIGHT_CSVS[source.key],
        source_label=f"embedded:{source.key.value}",
    )
    path = Path(source.source_csv)
    if path.exists():
        local_rows = _parse_weight_rows(path.read_text(), source_label=str(path))
        _validate_embedded_matches_local(embedded_rows, local_rows, source=source)
    else:
        logger.info("Using embedded weights for %s; local provenance CSV is absent at %s", source.key.value, path)
    return embedded_rows


def _read_phase_weights(source: MixtureSource) -> tuple[dict[str, dict[str, float]], MixtureDiagnostics]:
    rows = _read_phase_weight_rows(source)

    domains = [row["domain"] for row in rows]
    if sorted(domains) != sorted(DOMAIN_NAMES):
        missing = sorted(set(DOMAIN_NAMES).difference(domains))
        extra = sorted(set(domains).difference(DOMAIN_NAMES))
        raise ValueError(f"{source.key.value} domain mismatch: missing={missing}, extra={extra}")
    if len(domains) != len(set(domains)):
        raise ValueError(f"{source.key.value} has duplicate domains")

    phase_weights = {"phase_0": {}, "phase_1": {}}
    phase_lengths = _phase_lengths()
    aggregate_weights: dict[str, float] = {}
    simulated_epochs: list[float] = []
    for row in rows:
        domain = row["domain"]
        phase0 = float(row["phase_0_weight"])
        phase1 = float(row["phase_1_weight"])
        if phase0 < 0 or phase1 < 0:
            raise ValueError(f"{source.key.value}/{domain} has negative phase weights")
        phase_weights["phase_0"][domain] = phase0
        phase_weights["phase_1"][domain] = phase1
        aggregate = phase_lengths["phase_0"] * phase0 + phase_lengths["phase_1"] * phase1
        aggregate_weights[domain] = aggregate
        simulated_epoch = SIMULATED_EPOCH_TARGET_BUDGET * aggregate / TOP_LEVEL_DOMAIN_TOKEN_COUNTS[domain]
        simulated_epochs.append(simulated_epoch)
        if row.get("simulated_epochs"):
            recorded = float(row["simulated_epochs"])
            if abs(recorded - simulated_epoch) > max(1e-6, 1e-6 * abs(simulated_epoch)):
                raise ValueError(
                    f"{source.key.value}/{domain} recorded simulated_epochs={recorded} "
                    f"but recomputed {simulated_epoch}"
                )

    phase0_sum = sum(phase_weights["phase_0"].values())
    phase1_sum = sum(phase_weights["phase_1"].values())
    aggregate_sum = sum(aggregate_weights.values())
    for phase_name, total in [("phase_0", phase0_sum), ("phase_1", phase1_sum)]:
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"{source.key.value}/{phase_name} sums to {total}, expected 1")
    if abs(aggregate_sum - 1.0) > 1e-9:
        raise ValueError(f"{source.key.value} aggregate weights sum to {aggregate_sum}, expected 1")
    max_simulated_epoch = max(simulated_epochs)
    if source.expected_max_simulated_epoch is not None:
        if max_simulated_epoch > source.expected_max_simulated_epoch + 2e-6:
            raise ValueError(
                f"{source.key.value} max simulated epoch {max_simulated_epoch} exceeds expected "
                f"{source.expected_max_simulated_epoch}"
            )

    proportional = _proportional_weights()
    phase_tv = []
    for phase_name in PHASE_NAMES:
        tv = 0.5 * sum(abs(phase_weights[phase_name][domain] - proportional[domain]) for domain in DOMAIN_NAMES)
        phase_tv.append(tv)
    diagnostics = MixtureDiagnostics(
        mixture=source.key.value,
        source_csv=source.source_csv,
        phase0_sum=phase0_sum,
        phase1_sum=phase1_sum,
        aggregate_sum=aggregate_sum,
        max_simulated_epoch=max_simulated_epoch,
        q95_simulated_epoch=_q95(simulated_epochs),
        max_weight=max(max(weights.values()) for weights in phase_weights.values()),
        min_weight=min(min(weights.values()) for weights in phase_weights.values()),
        mean_phase_tv_to_proportional=sum(phase_tv) / len(phase_tv),
    )
    return phase_weights, diagnostics


def _weights_for_mixture(mixture: DelphiValidationMixture) -> tuple[dict[str, dict[str, float]], MixtureDiagnostics]:
    return _read_phase_weights(MIXTURE_SOURCES[mixture])


def _validate_runtime_phase_weights(phase_weights: dict[str, dict[str, float]], *, run_name: str) -> None:
    if set(phase_weights) != set(PHASE_NAMES):
        raise ValueError(f"{run_name} phase names mismatch: {sorted(phase_weights)}")
    for phase_name, weights in phase_weights.items():
        if set(weights) != set(DOMAIN_NAMES):
            raise ValueError(f"{run_name}/{phase_name} domain names mismatch")
        total = sum(float(weight) for weight in weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"{run_name}/{phase_name} weights sum to {total}, expected 1.0")
        negative = {domain: weight for domain, weight in weights.items() if weight < 0}
        if negative:
            raise ValueError(f"{run_name}/{phase_name} has negative weights: {negative}")


def _build_mixture_data(
    mixture: DelphiValidationMixture,
    train_tokens: int,
    model_config,
    batch_size: int,
    train_steps: int,
):
    phase_weights, _ = _weights_for_mixture(mixture)
    _validate_runtime_phase_weights(phase_weights, run_name=mixture.value)
    experiment = MixtureExperiment(
        name=EXPERIMENT_NAME,
        domains=build_top_level_domains(runtime_cache_region=DEFAULT_RUNTIME_CACHE_REGION),
        phase_schedule=PHASE_SCHEDULE,
        model_config=model_config,
        batch_size=batch_size,
        seq_len=SEQ_LEN_DELPHI,
        num_train_steps=train_steps,
        target_budget=None,
        resources=ResourceConfig.with_tpu("v5p-8", regions=[DEFAULT_TPU_REGION], zone=DEFAULT_TPU_ZONE),
        eval_harness_tasks=(),
        optimizer_config=None,
        eval_datasets_cache_path=None,
        hierarchical_runtime_domains=True,
    )
    data = experiment.create_mixture_config(WeightConfig(run_id=0, phase_weights=phase_weights))
    if train_tokens > SIMULATED_EPOCH_TARGET_BUDGET:
        raise ValueError(
            f"Delphi train_tokens={train_tokens} exceeds simulated-epoch target budget "
            f"{SIMULATED_EPOCH_TARGET_BUDGET}; simulated epoching would be ill-defined."
        )
    return (
        replace(
            data,
            target_budget=SIMULATED_EPOCH_TARGET_BUDGET,
            experiment_budget=train_tokens,
            simulated_epoch_subset_seed=None,
        ),
        phase_weights,
    )


def run_delphi_optimized_training(config: DelphiOptimizedTrainingConfig) -> None:
    """Run one Delphi optimized-mixture training job."""
    scaling_fits = _read_scaling_fits(config.analysis_output_path)
    candidate = _candidate_for_budget(
        scaling_fits=scaling_fits,
        target_flops=config.target_flops,
        batch_size=config.batch_size,
    )
    params = candidate.model_config.total_trainable_params(completed_adamh_heuristic.vocab_size)
    realized_train_tokens = candidate.train_steps * config.batch_size * SEQ_LEN_DELPHI
    tp = _tensor_parallel_size(candidate.model_config.hidden_dim, config.tpu_type)

    source = MIXTURE_SOURCES[config.mixture]
    logger.info(
        "Delphi optimized %s/%s: hidden_dim=%d layers=%d params=%.2e tokens=%.2e "
        "realized_tokens=%d batch_size=%d steps=%d tpu=%s tp=%d phase_boundary=%.3f",
        config.mixture.value,
        _slug(config.target_flops),
        candidate.model_config.hidden_dim,
        candidate.model_config.num_layers,
        params,
        candidate.tokens,
        realized_train_tokens,
        config.batch_size,
        candidate.train_steps,
        config.tpu_type,
        tp,
        PHASE_BOUNDARIES[0],
    )

    data, phase_weights = _build_mixture_data(
        config.mixture,
        realized_train_tokens,
        candidate.model_config,
        config.batch_size,
        candidate.train_steps,
    )
    _validate_runtime_phase_weights(phase_weights, run_name=config.run_name)
    data = _add_validation_components(data, config.validation_configs)

    inner_config = train_lm.TrainLmConfig(
        data=data,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                entity="marin-community",
                project="marin",
                tags=[
                    f"issue-{source.github_issue}",
                    "delphi-uncheatable-optimized-mixtures",
                    "completed-adamh",
                    config.mixture.value,
                    source.method,
                    f"FLOPs={config.target_flops:.1e}",
                    f"label={config.label}",
                    f"N={params:.1e}",
                    f"data_seed={config.data_seed}",
                    f"trainer_seed={config.trainer_seed}",
                ],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=candidate.batch_size,
            per_device_parallelism=-1,
            num_train_steps=candidate.train_steps,
            steps_per_eval=1000,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[dict(every=5000)],
            ),
            mesh=MeshConfig(
                axes={"data": -1, "replica": 1, "model": tp},
                compute_mapping={
                    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                },
            ),
            seed=config.trainer_seed,
            allow_nondivisible_batch_size=True,
        ),
        train_seq_len=SEQ_LEN_DELPHI,
        model=candidate.model_config,
        optimizer=candidate.optimizer_config,
        data_seed=config.data_seed,
    )

    resources = ResourceConfig.with_tpu(config.tpu_type, regions=[config.tpu_region], zone=config.tpu_zone)
    pod_config = TrainLmOnPodConfig(
        train_config=inner_config,
        resources=resources,
        output_path=config.output_path,
        env_vars={
            "MARIN_PREFIX": marin_prefix_for_region(config.tpu_region),
            SKIP_EVAL_HARNESS_ENV_VAR: "1",
        },
    )
    run_levanter_train_lm(pod_config)


def _predict_run_spec(
    *,
    scaling_fits,
    mixture: DelphiValidationMixture,
    target_flops: float,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    batch_size: int,
    run_order: int,
) -> DelphiOptimizedRunSpec:
    source = MIXTURE_SOURCES[mixture]
    candidate = _candidate_for_budget(
        scaling_fits=scaling_fits,
        target_flops=target_flops,
        batch_size=batch_size,
    )
    train_tokens = round(candidate.tokens)
    realized_train_tokens = candidate.train_steps * batch_size * SEQ_LEN_DELPHI
    phase_weights, diagnostics = _weights_for_mixture(mixture)
    run_name = f"{mixture.value}_{_slug(target_flops)}"
    _validate_runtime_phase_weights(phase_weights, run_name=run_name)
    non_embedding_params = int(candidate.model_config.total_trainable_params(0))
    total_params = int(candidate.model_config.total_trainable_params(completed_adamh_heuristic.vocab_size))
    return DelphiOptimizedRunSpec(
        run_order=run_order,
        run_id=RUN_ID_BASE + run_order,
        run_name=run_name,
        mixture=mixture.value,
        mixture_display_name=source.display_name,
        source_csv=source.source_csv,
        github_issue=source.github_issue,
        target_metric=source.target_metric,
        method=source.method,
        target_flops=target_flops,
        tpu_type=tpu_type,
        tpu_region=tpu_region,
        tpu_zone=tpu_zone,
        batch_size=batch_size,
        train_tokens=train_tokens,
        train_steps=candidate.train_steps,
        realized_train_tokens=realized_train_tokens,
        expected_checkpoint_step=candidate.train_steps - 1,
        model_hidden_dim=int(candidate.model_config.hidden_dim),
        model_layers=int(candidate.model_config.num_layers),
        non_embedding_params=non_embedding_params,
        total_trainable_params=total_params,
        tensor_parallel_size=_tensor_parallel_size(candidate.model_config.hidden_dim, tpu_type),
        data_seed=RUN_ID_BASE + run_order,
        trainer_seed=0,
        phase_boundary=PHASE_BOUNDARIES[0],
        phase_0_fraction=PHASE_FRACTIONS["phase_0"],
        phase_1_fraction=PHASE_FRACTIONS["phase_1"],
        simulated_epoch_target_budget=SIMULATED_EPOCH_TARGET_BUDGET,
        available_top_level_tokens=TOP_LEVEL_TOTAL_AVAILABLE_TOKENS,
        max_simulated_epoch=diagnostics.max_simulated_epoch,
        q95_simulated_epoch=diagnostics.q95_simulated_epoch,
        mean_phase_tv_to_proportional=diagnostics.mean_phase_tv_to_proportional,
        phase_weights=phase_weights,
    )


def save_delphi_optimized_manifest(config: SaveDelphiOptimizedManifestConfig) -> None:
    """Persist run specs as JSON and CSV artifacts."""
    target_budgets = {
        float(item["target_flops"]): (str(item["tpu_type"]), int(item["batch_size"]))
        for item in json.loads(config.target_budgets_json)
    }
    scaling_fits = _read_scaling_fits(config.analysis_output_path)
    run_specs: list[DelphiOptimizedRunSpec] = []
    diagnostics: list[MixtureDiagnostics] = []
    for mixture in config.mixtures:
        _, diag = _weights_for_mixture(mixture)
        diagnostics.append(diag)
    for target_flops, (tpu_type, batch_size) in target_budgets.items():
        for mixture in config.mixtures:
            run_specs.append(
                _predict_run_spec(
                    scaling_fits=scaling_fits,
                    mixture=mixture,
                    target_flops=target_flops,
                    tpu_type=tpu_type,
                    tpu_region=config.tpu_region,
                    tpu_zone=config.tpu_zone,
                    batch_size=batch_size,
                    run_order=config.run_order_offset + len(run_specs),
                )
            )

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fs.open(os.path.join(config.output_path, "run_specs.json"), "w") as handle:
        json.dump([asdict(run_spec) for run_spec in run_specs], handle, indent=2, sort_keys=True)
    with fs.open(os.path.join(config.output_path, "selected_mixtures.json"), "w") as handle:
        json.dump([asdict(item) for item in diagnostics], handle, indent=2, sort_keys=True)
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=[
            "run_order",
            "run_id",
            "run_name",
            "mixture",
            "mixture_display_name",
            "github_issue",
            "target_metric",
            "method",
            "target_flops",
            "tpu_type",
            "tpu_region",
            "tpu_zone",
            "batch_size",
            "train_tokens",
            "train_steps",
            "realized_train_tokens",
            "expected_checkpoint_step",
            "model_hidden_dim",
            "model_layers",
            "non_embedding_params",
            "total_trainable_params",
            "tensor_parallel_size",
            "data_seed",
            "trainer_seed",
            "phase_boundary",
            "phase_0_fraction",
            "phase_1_fraction",
            "simulated_epoch_target_budget",
            "available_top_level_tokens",
            "max_simulated_epoch",
            "q95_simulated_epoch",
            "mean_phase_tv_to_proportional",
            "source_csv",
        ],
    )
    writer.writeheader()
    for run_spec in run_specs:
        row = asdict(run_spec)
        row.pop("phase_weights")
        writer.writerow(row)
    with fs.open(os.path.join(config.output_path, "training_manifest.csv"), "w") as handle:
        handle.write(csv_buffer.getvalue())
    summary: dict[str, Any] = {
        "n_runs": len(run_specs),
        "mixtures": sorted({run_spec.mixture for run_spec in run_specs}),
        "target_flops": sorted({run_spec.target_flops for run_spec in run_specs}),
        "source_experiment": EXPERIMENT_NAME,
        "analysis_output_path": config.analysis_output_path,
        "phase_boundary": PHASE_BOUNDARIES[0],
        "phase_fractions": dict(PHASE_FRACTIONS),
        "run_order_offset": config.run_order_offset,
        "simulated_epoch_target_budget": SIMULATED_EPOCH_TARGET_BUDGET,
        "available_top_level_tokens": TOP_LEVEL_TOTAL_AVAILABLE_TOKENS,
        "diagnostics": [asdict(item) for item in diagnostics],
    }
    with fs.open(os.path.join(config.output_path, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def _selected_target_budgets(values: tuple[str, ...]) -> dict[float, tuple[str, int]]:
    if not values:
        return dict(TARGET_BUDGETS)
    selected: dict[float, tuple[str, int]] = {}
    unknown: list[str] = []
    for value in values:
        target = float(value)
        if target not in TARGET_BUDGETS:
            unknown.append(value)
            continue
        selected[target] = TARGET_BUDGETS[target]
    if unknown:
        allowed = ", ".join(f"{budget:.0e}" for budget in TARGET_BUDGETS)
        raise ValueError(f"Unknown target budget(s): {unknown}. Allowed: {allowed}")
    return selected


def _parse_mixtures(values: tuple[str, ...]) -> tuple[DelphiValidationMixture, ...]:
    if not values:
        return tuple(DelphiValidationMixture)
    return tuple(DelphiValidationMixture(value) for value in values)


def build_launch_artifacts(
    *,
    analysis_output_path: str,
    validation_configs: dict[str, DatasetComponent],
    mixtures: tuple[DelphiValidationMixture, ...],
    target_budgets: dict[float, tuple[str, int]],
    tpu_region: str,
    tpu_zone: str,
    run_order_offset: int = 0,
) -> LaunchArtifacts:
    """Build the executor graph for selected mixtures and FLOP budgets."""
    training_steps: list[ExecutorStep] = []
    for target_flops, (tpu_type, batch_size) in target_budgets.items():
        for mixture in mixtures:
            run_order = run_order_offset + len(training_steps)
            run_name = f"{mixture.value}_{_slug(target_flops)}"
            training_steps.append(
                ExecutorStep(
                    name=f"{EXPERIMENT_NAME}/{run_name}",
                    fn=run_delphi_optimized_training,
                    resources=ResourceConfig.with_tpu(tpu_type, regions=[tpu_region], zone=tpu_zone),
                    config=DelphiOptimizedTrainingConfig(
                        analysis_output_path=analysis_output_path,
                        target_flops=target_flops,
                        tpu_type=tpu_type,
                        tpu_region=tpu_region,
                        tpu_zone=tpu_zone,
                        batch_size=batch_size,
                        mixture=mixture,
                        label=LABEL,
                        output_path=this_output_path(),
                        run_id=RUN_ID_BASE + run_order,
                        run_name=run_name,
                        data_seed=RUN_ID_BASE + run_order,
                        trainer_seed=0,
                        validation_configs=validation_configs,
                    ),
                )
            )

    manifest_step = ExecutorStep(
        name=f"{EXPERIMENT_NAME}/manifest",
        fn=save_delphi_optimized_manifest,
        config=SaveDelphiOptimizedManifestConfig(
            output_path=this_output_path(),
            analysis_output_path=analysis_output_path,
            mixtures=mixtures,
            target_budgets_json=json.dumps(
                [
                    {"target_flops": target_flops, "tpu_type": tpu_type, "batch_size": batch_size}
                    for target_flops, (tpu_type, batch_size) in target_budgets.items()
                ],
                sort_keys=True,
            ),
            tpu_region=tpu_region,
            tpu_zone=tpu_zone,
            run_order_offset=run_order_offset,
        ),
    )
    return LaunchArtifacts(manifest_step=manifest_step, training_steps=training_steps)


def _write_local_static_manifest(
    *,
    mixtures: tuple[DelphiValidationMixture, ...],
    target_budgets: dict[float, tuple[str, int]],
    tpu_region: str,
    tpu_zone: str,
    run_order_offset: int = 0,
) -> None:
    """Write a local dependency-light manifest validating static mixture semantics."""
    LOCAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    diagnostics: list[MixtureDiagnostics] = []
    rows: list[dict[str, Any]] = []
    for mixture in mixtures:
        _, diag = _weights_for_mixture(mixture)
        diagnostics.append(diag)
    run_order = run_order_offset
    for target_flops, (tpu_type, batch_size) in target_budgets.items():
        for mixture in mixtures:
            source = MIXTURE_SOURCES[mixture]
            diag = next(item for item in diagnostics if item.mixture == mixture.value)
            rows.append(
                {
                    "run_order": run_order,
                    "run_id": RUN_ID_BASE + run_order,
                    "run_name": f"{mixture.value}_{_slug(target_flops)}",
                    "mixture": mixture.value,
                    "mixture_display_name": source.display_name,
                    "github_issue": source.github_issue,
                    "target_metric": source.target_metric,
                    "method": source.method,
                    "target_flops": target_flops,
                    "tpu_type": tpu_type,
                    "tpu_region": tpu_region,
                    "tpu_zone": tpu_zone,
                    "batch_size": batch_size,
                    "phase_boundary": PHASE_BOUNDARIES[0],
                    "phase_0_fraction": PHASE_FRACTIONS["phase_0"],
                    "phase_1_fraction": PHASE_FRACTIONS["phase_1"],
                    "simulated_epoch_target_budget": SIMULATED_EPOCH_TARGET_BUDGET,
                    "max_simulated_epoch": diag.max_simulated_epoch,
                    "q95_simulated_epoch": diag.q95_simulated_epoch,
                    "mean_phase_tv_to_proportional": diag.mean_phase_tv_to_proportional,
                    "source_csv": source.source_csv,
                }
            )
            run_order += 1

    with (LOCAL_ARTIFACT_DIR / "selected_mixtures.json").open("w") as handle:
        json.dump([asdict(item) for item in diagnostics], handle, indent=2, sort_keys=True)
    with (LOCAL_ARTIFACT_DIR / "run_specs_static.json").open("w") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
    with (LOCAL_ARTIFACT_DIR / "training_manifest_static.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "n_runs": len(rows),
        "mixtures": [mixture.value for mixture in mixtures],
        "target_flops": sorted(target_budgets),
        "phase_boundary": PHASE_BOUNDARIES[0],
        "phase_fractions": dict(PHASE_FRACTIONS),
        "run_order_offset": run_order_offset,
        "simulated_epoch_target_budget": SIMULATED_EPOCH_TARGET_BUDGET,
        "available_top_level_tokens": TOP_LEVEL_TOTAL_AVAILABLE_TOKENS,
        "diagnostics": [asdict(item) for item in diagnostics],
        "note": "Static local manifest; remote executor manifest adds model/token predictions from scaling fits.",
    }
    with (LOCAL_ARTIFACT_DIR / "summary_static.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixtures", nargs="*", default=[])
    parser.add_argument("--target-budgets", nargs="*", default=[])
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument(
        "--run-order-offset",
        type=int,
        default=0,
        help=(
            "Offset added to generated run_order/run_id/data_seed. Keep 0 for normal launches; "
            "use only for scoped retries that need to preserve the original manifest row seed."
        ),
    )
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help=(
            "Do not include the manifest step in the executor graph. Use only for scoped retries after "
            "the full local dry-run manifest has been captured; this avoids overwriting shared manifest "
            "provenance with a one-row retry manifest."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analysis-output-path", default=DEFAULT_ANALYSIS_OUTPUT_PATH)
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if args.tpu_region != DEFAULT_TPU_REGION or args.tpu_zone != DEFAULT_TPU_ZONE:
        raise ValueError(f"This launcher is pinned to {DEFAULT_TPU_REGION}/{DEFAULT_TPU_ZONE}")
    expected_prefix = marin_prefix_for_region(args.tpu_region)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != expected_prefix:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match required east5 prefix {expected_prefix!r}")
    os.environ["MARIN_PREFIX"] = expected_prefix

    mixtures = _parse_mixtures(tuple(args.mixtures))
    target_budgets = _selected_target_budgets(tuple(args.target_budgets))
    if not args.analysis_output_path:
        raise ValueError("--analysis-output-path must be set; do not rerun isoflop analysis in this parent")
    if args.run_order_offset < 0:
        raise ValueError("--run-order-offset must be nonnegative")

    if args.dry_run:
        _write_local_static_manifest(
            mixtures=mixtures,
            target_budgets=target_budgets,
            tpu_region=args.tpu_region,
            tpu_zone=args.tpu_zone,
            run_order_offset=args.run_order_offset,
        )
        logger.info("Wrote static dry-run specs under %s", LOCAL_ARTIFACT_DIR)
        return

    validation_steps = default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
    }

    artifacts = build_launch_artifacts(
        analysis_output_path=args.analysis_output_path,
        validation_configs=validation_configs,
        mixtures=mixtures,
        target_budgets=target_budgets,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
        run_order_offset=args.run_order_offset,
    )
    if os.getenv("CI") is not None:
        logger.info(
            "Built Delphi optimized-mixture graph with %d training steps; skipping executor launch.",
            len(artifacts.training_steps),
        )
        return

    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=artifacts.training_steps if args.skip_manifest else artifacts.steps,
        description=f"{EXPERIMENT_NAME}: issues #6602/#6608 Delphi scaling validation",
    )


if __name__ == "__main__":
    main()
