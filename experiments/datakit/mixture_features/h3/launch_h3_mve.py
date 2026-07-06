# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the 3 H3 mixing-via-embeddings mixtures at 300M / 6B over 40 domains.

The 40 domains are the 39 qsplit240 top-level Dolma3/Dolmino domains plus
``dolma_starcoder`` (existing tokenized cache in us-east5). Phase weights come
from scratch/mixture_features/h3/surrogate_result.json (embedded below so the
script is self-contained). Modeled on launch_two_phase_many_qsplit240_300m_6b.py.

Dry run (no TPU jobs, no W&B):
    MARIN_PREFIX=gs://marin-us-east5 python experiments/domain_phase_mix/launch_h3_mve.py \
        --dry_run true --prefix gs://marin-us-east5
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import replace

from fray.cluster import ResourceConfig
from marin.evaluation.eval_dataset_cache import create_cache_eval_datasets_step
from marin.execution.executor import executor_main
from marin.execution.remote import RemoteCallable, remote
from marin.processing.tokenize.data_configs import ExistingTokenizedCacheConfig

from experiments.domain_phase_mix.config import DatasetComponent, Domain, PhaseSchedule, WeightConfig
from experiments.domain_phase_mix.experiment import MixtureExperiment
from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_300m_6b import QSPLIT240_300M_EVAL_TASKS
from experiments.domain_phase_mix.proxy_sweep import (
    get_num_train_steps,
    regmix_300m_muonh_base,
    regmix_300m_proxy,
)
from experiments.domain_phase_mix.qsplit240_replay import (
    add_eval_cache_dependency_to_training_step,
    resolve_qsplit240_eval_cache_path_for_regions,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    BATCH_SIZE,
    PHASE_BOUNDARIES,
    PHASE_NAMES,
    SEQ_LEN,
    TARGET_BUDGET,
    build_top_level_domains,
    create_two_phase_dolma3_dolmino_top_level_optimizer_config,
)
from experiments.marin_tokenizer import marin_tokenizer

logger = logging.getLogger(__name__)

NAME_PREFIX = "rav/mixing_via_embeddings/h3_300m_6b"
EXPERIMENT_BUDGET = 6_000_000_000
NUM_TRAIN_STEPS = get_num_train_steps(EXPERIMENT_BUDGET, BATCH_SIZE, SEQ_LEN)
DEFAULT_TPU_TYPE = "v5p-8"
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"
DEFAULT_MAX_CONCURRENT = 3

STARCODER_DOMAIN_NAME = "dolma_starcoder"
STARCODER_CACHE_PATH = "gs://marin-us-east5/tokenized/dolma/starcoder-8b6089"
STARCODER_TOKEN_COUNT = 216_567_300_822

N_DOMAINS = 40

# (run_name, mixture key in H3_FULL_WEIGHTS, run_id/data_seed)
H3_RUNS = (
    ("rav_mve_h3_proposal", "PROPOSAL", 0),
    ("rav_mve_h3_olmix", "OLMIX_REUSE", 1),
    ("rav_mve_h3_tokprop", "TOKEN_PROPORTIONAL", 2),
)


def _starcoder_cache_config() -> ExistingTokenizedCacheConfig:
    return ExistingTokenizedCacheConfig(
        cache_path=STARCODER_CACHE_PATH,
        tokenizer=marin_tokenizer,
        tags=[STARCODER_DOMAIN_NAME, "source_runtime_cache"],
    )


def build_h3_domains(*, runtime_cache_region: str = DEFAULT_TPU_REGION) -> list[Domain]:
    """Return the 39 qsplit240 top-level domains plus dolma_starcoder."""
    domains = build_top_level_domains(runtime_cache_region=runtime_cache_region)
    domains.append(
        Domain(
            name=STARCODER_DOMAIN_NAME,
            components=[
                DatasetComponent(
                    name=STARCODER_DOMAIN_NAME,
                    step_fn=_starcoder_cache_config,
                    weight=STARCODER_TOKEN_COUNT,
                )
            ],
            description="StarCoder singleton domain backed directly by its original tokenized cache.",
        )
    )
    assert len(domains) == N_DOMAINS, f"Expected {N_DOMAINS} domains, got {len(domains)}"
    return domains


def create_h3_experiment(
    *,
    tpu_type: str = DEFAULT_TPU_TYPE,
    tpu_region: str = DEFAULT_TPU_REGION,
    tpu_zone: str = DEFAULT_TPU_ZONE,
    eval_datasets_cache_path: str | None = None,
) -> MixtureExperiment:
    """Create the 300M / 6B two-phase experiment over the 40 H3 domains."""
    phase_schedule = PhaseSchedule.from_boundaries(boundaries=PHASE_BOUNDARIES, names=list(PHASE_NAMES))
    optimizer_config = create_two_phase_dolma3_dolmino_top_level_optimizer_config(
        experiment_budget=EXPERIMENT_BUDGET,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        phase_schedule=phase_schedule,
        optimizer_config=regmix_300m_muonh_base,
    )
    resolved_eval_cache_path = resolve_qsplit240_eval_cache_path_for_regions(
        (tpu_region,),
        eval_datasets_cache_path,
    )
    return MixtureExperiment(
        name=NAME_PREFIX,
        domains=build_h3_domains(runtime_cache_region=tpu_region),
        phase_schedule=phase_schedule,
        model_config=regmix_300m_proxy,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_train_steps=NUM_TRAIN_STEPS,
        target_budget=TARGET_BUDGET,
        resources=ResourceConfig.with_tpu(tpu_type, regions=[tpu_region], zone=tpu_zone),
        eval_harness_tasks=QSPLIT240_300M_EVAL_TASKS,
        optimizer_config=optimizer_config,
        eval_datasets_cache_path=resolved_eval_cache_path,
        hierarchical_runtime_domains=True,
    )


def _validate_phase_weights(run_name: str, phase_weights: dict[str, dict[str, float]], domain_names: set[str]) -> None:
    assert set(phase_weights) == set(PHASE_NAMES), f"{run_name}: phases {sorted(phase_weights)} != {PHASE_NAMES}"
    for phase_name, weights in phase_weights.items():
        assert set(weights) == domain_names, (
            f"{run_name}/{phase_name}: domain mismatch: "
            f"missing={domain_names - set(weights)}, extra={set(weights) - domain_names}"
        )
        total = sum(weights.values())
        assert math.isclose(total, 1.0, abs_tol=1e-6), f"{run_name}/{phase_name}: weights sum to {total!r}, not 1"


def _merge_unique_strings(left: list[str], right: tuple[str, ...]) -> list[str]:
    merged = list(left)
    for item in right:
        if item not in merged:
            merged.append(item)
    return merged


def _add_training_dependency_groups(training_step, *, groups=("eval",)):
    """Carry the ``eval`` uv extra into child training jobs.

    Old-style TPU training steps infer only the ``tpu`` extra from resources, but the inline
    lm-eval harness imports ``lm_eval`` inside each child, so the ``eval`` group must be added.
    Mirrors ``launch_starcoder_heteroskedastic_snr._add_training_dependency_groups``.
    """
    if isinstance(training_step.fn, RemoteCallable):
        return replace(
            training_step,
            fn=replace(
                training_step.fn,
                pip_dependency_groups=_merge_unique_strings(training_step.fn.pip_dependency_groups, groups),
            ),
        )
    return replace(
        training_step,
        fn=remote(training_step.fn, resources=training_step.resources, pip_dependency_groups=list(groups)),
    )


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Launch the 3 H3 mixing-via-embeddings runs at 300M / 6B.")
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--eval-datasets-cache-path", default=None)
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if os.getenv("CI") is not None:
        logger.info("Skipping H3 mixing-via-embeddings launch in CI environment")
        return

    experiment = create_h3_experiment(
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
        eval_datasets_cache_path=args.eval_datasets_cache_path,
    )
    domain_names = set(experiment.experiment_config.domain_names)
    assert len(domain_names) == N_DOMAINS, f"Expected {N_DOMAINS} domains, got {len(domain_names)}"
    assert experiment.num_train_steps == 22888, experiment.num_train_steps

    cache_eval_datasets_step = create_cache_eval_datasets_step(
        eval_tasks=QSPLIT240_300M_EVAL_TASKS,
        gcs_path=resolve_qsplit240_eval_cache_path_for_regions((args.tpu_region,), args.eval_datasets_cache_path),
        name_prefix=NAME_PREFIX,
    )

    training_steps = []
    for run_name, mixture_key, run_id in H3_RUNS:
        phase_weights = H3_FULL_WEIGHTS[mixture_key]
        _validate_phase_weights(run_name, phase_weights, domain_names)
        training_step = experiment.create_training_step(
            weight_config=WeightConfig(run_id=run_id, phase_weights=phase_weights),
            name_prefix=NAME_PREFIX,
            run_name=run_name,
            data_seed=run_id,
        )
        training_step = add_eval_cache_dependency_to_training_step(training_step, cache_eval_datasets_step)
        training_step = _add_training_dependency_groups(training_step, groups=("eval",))
        training_steps.append(training_step)

    logger.info(
        "Launching %d H3 runs on %s in %s/%s.", len(training_steps), args.tpu_type, args.tpu_region, args.tpu_zone
    )
    executor_main(
        steps=[cache_eval_datasets_step, *training_steps],
        description=f"{NAME_PREFIX}: H3 mixing-via-embeddings mixtures (proposal, olmix reuse, token proportional).",
        max_concurrent=DEFAULT_MAX_CONCURRENT,
    )


# Phase weights for the 3 H3 mixtures, copied verbatim from
# scratch/mixture_features/h3/surrogate_result.json (key: full_weights).
H3_FULL_WEIGHTS: dict[str, dict[str, dict[str, float]]] = json.loads(
    r"""
{
  "OLMIX_REUSE": {
    "phase_0": {
      "dolma3_arxiv": 0.02586206985912246,
      "dolma3_cc/art_and_design_high": 0.013636285806653655,
      "dolma3_cc/art_and_design_low": 0.02015319554514029,
      "dolma3_cc/crime_and_law_high": 0.026501149361851748,
      "dolma3_cc/crime_and_law_low": 0.0080093548370152,
      "dolma3_cc/education_and_jobs_high": 0.009433050972350147,
      "dolma3_cc/education_and_jobs_low": 0.04237236505914563,
      "dolma3_cc/electronics_and_hardware_high": 0.05142278760551233,
      "dolma3_cc/electronics_and_hardware_low": 0.011486502495120598,
      "dolma3_cc/entertainment_high": 0.010023216519939563,
      "dolma3_cc/entertainment_low": 0.02305006205827584,
      "dolma3_cc/finance_and_business_high": 0.06519701051202151,
      "dolma3_cc/finance_and_business_low": 0.011784790162060585,
      "dolma3_cc/food_and_dining_high": 0.02731241623677609,
      "dolma3_cc/food_and_dining_low": 0.0082686123675375,
      "dolma3_cc/games_high": 0.014592810987896889,
      "dolma3_cc/games_low": 0.0025208383290731134,
      "dolma3_cc/health_high": 0.016812519065854667,
      "dolma3_cc/health_low": 0.01661651447866988,
      "dolma3_cc/history_and_geography_high": 0.010896898848887705,
      "dolma3_cc/history_and_geography_low": 0.039320100707997616,
      "dolma3_cc/industrial_high": 0.007097565898727158,
      "dolma3_cc/industrial_low": 0.015720924470374032,
      "dolma3_cc/literature_high": 0.02390557839820223,
      "dolma3_cc/literature_low": 0.03218330190662535,
      "dolma3_cc/science_math_and_technology_high": 0.004600532890236823,
      "dolma3_cc/science_math_and_technology_low": 0.03700302498325248,
      "dolma3_finemath_3plus": 0.026531650003820104,
      "dolma3_stack_edu": 0.06481730881697713,
      "dolma3_wikipedia": 0.004361166217587244,
      "dolma_starcoder": 0.030066268737441462,
      "dolmino_common_crawl_hq": 0.06929183346930323,
      "dolmino_olmocr_pdfs_hq": 0.02544396157567286,
      "dolmino_stack_edu_fim": 0.01749797901869016,
      "dolmino_stem_heavy_crawl": 0.0042617633195254296,
      "dolmino_synth_code": 0.01769726049487597,
      "dolmino_synth_instruction": 0.008670771215012994,
      "dolmino_synth_math": 0.02971564026568483,
      "dolmino_synth_qa": 0.12006874024223571,
      "dolmino_synth_thinking": 0.005792176258851783
    },
    "phase_1": {
      "dolma3_arxiv": 0.05011890480609187,
      "dolma3_cc/art_and_design_high": 0.019604751281089863,
      "dolma3_cc/art_and_design_low": 0.022459431063761096,
      "dolma3_cc/crime_and_law_high": 0.007952931244435356,
      "dolma3_cc/crime_and_law_low": 0.0272698958461217,
      "dolma3_cc/education_and_jobs_high": 0.040140841493236684,
      "dolma3_cc/education_and_jobs_low": 0.00396894444291538,
      "dolma3_cc/electronics_and_hardware_high": 0.0025673996781075017,
      "dolma3_cc/electronics_and_hardware_low": 0.06396330482052971,
      "dolma3_cc/entertainment_high": 0.012037457089174758,
      "dolma3_cc/entertainment_low": 0.05612829313010303,
      "dolma3_cc/finance_and_business_high": 0.004561080806183236,
      "dolma3_cc/finance_and_business_low": 0.014065835931954812,
      "dolma3_cc/food_and_dining_high": 0.0002352211748777962,
      "dolma3_cc/food_and_dining_low": 0.01021884514703046,
      "dolma3_cc/games_high": 0.0026832510111710855,
      "dolma3_cc/games_low": 0.009552942249882514,
      "dolma3_cc/health_high": 0.05369580020953843,
      "dolma3_cc/health_low": 0.03745199734983985,
      "dolma3_cc/history_and_geography_high": 0.05464453804495805,
      "dolma3_cc/history_and_geography_low": 0.0029236730038265397,
      "dolma3_cc/industrial_high": 0.030487990433956978,
      "dolma3_cc/industrial_low": 0.0058742829924808395,
      "dolma3_cc/literature_high": 0.023660870338332484,
      "dolma3_cc/literature_low": 0.05585203784801987,
      "dolma3_cc/science_math_and_technology_high": 0.09256328651905854,
      "dolma3_cc/science_math_and_technology_low": 0.0009549068099094569,
      "dolma3_finemath_3plus": 0.01230215242033274,
      "dolma3_stack_edu": 0.0061033272664691805,
      "dolma3_wikipedia": 0.0029013457647514966,
      "dolma_starcoder": 0.030066268737441462,
      "dolmino_common_crawl_hq": 0.05487464141665394,
      "dolmino_olmocr_pdfs_hq": 0.027765344199672852,
      "dolmino_stack_edu_fim": 0.0706680588710828,
      "dolmino_stem_heavy_crawl": 0.010811751332689546,
      "dolmino_synth_code": 0.0302676250732695,
      "dolmino_synth_instruction": 0.0006134036592242288,
      "dolmino_synth_math": 0.013344238089053663,
      "dolmino_synth_qa": 0.019793744401595578,
      "dolmino_synth_thinking": 0.01484938400117516
    }
  },
  "PROPOSAL": {
    "phase_0": {
      "dolma3_arxiv": 0.008649739510215658,
      "dolma3_cc/art_and_design_high": 0.01955830177441803,
      "dolma3_cc/art_and_design_low": 0.0039925690339360454,
      "dolma3_cc/crime_and_law_high": 0.012997835299621584,
      "dolma3_cc/crime_and_law_low": 0.0039652427754644035,
      "dolma3_cc/education_and_jobs_high": 0.024463226587281842,
      "dolma3_cc/education_and_jobs_low": 0.0182789249187922,
      "dolma3_cc/electronics_and_hardware_high": 0.01727132167140113,
      "dolma3_cc/electronics_and_hardware_low": 0.009361649080648583,
      "dolma3_cc/entertainment_high": 0.04437897941552612,
      "dolma3_cc/entertainment_low": 0.02693539782453171,
      "dolma3_cc/finance_and_business_high": 0.07028181946296426,
      "dolma3_cc/finance_and_business_low": 0.02148438134595461,
      "dolma3_cc/food_and_dining_high": 0.02305547176757964,
      "dolma3_cc/food_and_dining_low": 0.006700080872769007,
      "dolma3_cc/games_high": 0.020218821897789865,
      "dolma3_cc/games_low": 0.002626109217555715,
      "dolma3_cc/health_high": 0.03822041380008505,
      "dolma3_cc/health_low": 0.0157290703987874,
      "dolma3_cc/history_and_geography_high": 0.02124912181354224,
      "dolma3_cc/history_and_geography_low": 0.006504748311488722,
      "dolma3_cc/industrial_high": 0.01631210129557027,
      "dolma3_cc/industrial_low": 0.005575324672695228,
      "dolma3_cc/literature_high": 0.018396025580486267,
      "dolma3_cc/literature_low": 0.014162526219181508,
      "dolma3_cc/science_math_and_technology_high": 0.04449085914650558,
      "dolma3_cc/science_math_and_technology_low": 0.02151790627982969,
      "dolma3_finemath_3plus": 0.007069453258906666,
      "dolma3_stack_edu": 0.056798998088396294,
      "dolma3_wikipedia": 0.00049204457608191,
      "dolma_starcoder": 0.0518915000813468,
      "dolmino_common_crawl_hq": 0.17168401582005538,
      "dolmino_olmocr_pdfs_hq": 0.04405230508307269,
      "dolmino_stack_edu_fim": 0.043666530644859015,
      "dolmino_stem_heavy_crawl": 0.0004642159726818925,
      "dolmino_synth_code": 0.0023065674142534395,
      "dolmino_synth_instruction": 0.0031313065252181675,
      "dolmino_synth_math": 0.011864811264064317,
      "dolmino_synth_qa": 0.06254564488213409,
      "dolmino_synth_thinking": 0.007654636414307025
    },
    "phase_1": {
      "dolma3_arxiv": 0.012214995530074992,
      "dolma3_cc/art_and_design_high": 0.009728269032525725,
      "dolma3_cc/art_and_design_low": 0.005483426788812609,
      "dolma3_cc/crime_and_law_high": 0.01136421314301646,
      "dolma3_cc/crime_and_law_low": 0.009223510984145934,
      "dolma3_cc/education_and_jobs_high": 0.017100688171365203,
      "dolma3_cc/education_and_jobs_low": 0.003991231888094974,
      "dolma3_cc/electronics_and_hardware_high": 0.0068860042103059535,
      "dolma3_cc/electronics_and_hardware_low": 0.015585154031617188,
      "dolma3_cc/entertainment_high": 0.02731036791524469,
      "dolma3_cc/entertainment_low": 0.014747963049619211,
      "dolma3_cc/finance_and_business_high": 0.04580006517168007,
      "dolma3_cc/finance_and_business_low": 0.015416783546905432,
      "dolma3_cc/food_and_dining_high": 0.010018346997425308,
      "dolma3_cc/food_and_dining_low": 0.004860509312077318,
      "dolma3_cc/games_high": 0.01736008168059591,
      "dolma3_cc/games_low": 0.004031721709398332,
      "dolma3_cc/health_high": 0.030378395209501308,
      "dolma3_cc/health_low": 0.013000673497189948,
      "dolma3_cc/history_and_geography_high": 0.01567451183998861,
      "dolma3_cc/history_and_geography_low": 0.0022872107764666167,
      "dolma3_cc/industrial_high": 0.00625281002446179,
      "dolma3_cc/industrial_low": 0.00130805980841565,
      "dolma3_cc/literature_high": 0.04200779600158413,
      "dolma3_cc/literature_low": 0.014177151256991148,
      "dolma3_cc/science_math_and_technology_high": 0.044095437547519205,
      "dolma3_cc/science_math_and_technology_low": 0.0230553792381477,
      "dolma3_finemath_3plus": 0.003436936645989533,
      "dolma3_stack_edu": 0.08810392840333366,
      "dolma3_wikipedia": 0.00032062508854887825,
      "dolma_starcoder": 0.05630642062031407,
      "dolmino_common_crawl_hq": 0.1318316292591911,
      "dolmino_olmocr_pdfs_hq": 0.072871026770704,
      "dolmino_stack_edu_fim": 0.12940317292403203,
      "dolmino_stem_heavy_crawl": 0.0011502224348048284,
      "dolmino_synth_code": 0.005215507414586453,
      "dolmino_synth_instruction": 0.0035093066123248204,
      "dolmino_synth_math": 0.004151065636465272,
      "dolmino_synth_qa": 0.0737514369052619,
      "dolmino_synth_thinking": 0.006587962921272182
    }
  },
  "TOKEN_PROPORTIONAL": {
    "phase_0": {
      "dolma3_arxiv": 0.003920251599600697,
      "dolma3_cc/art_and_design_high": 0.01585036607982536,
      "dolma3_cc/art_and_design_low": 0.006433483374358941,
      "dolma3_cc/crime_and_law_high": 0.021334820548967248,
      "dolma3_cc/crime_and_law_low": 0.009583728944191527,
      "dolma3_cc/education_and_jobs_high": 0.034882814666569374,
      "dolma3_cc/education_and_jobs_low": 0.017086415643936335,
      "dolma3_cc/electronics_and_hardware_high": 0.018405161186455334,
      "dolma3_cc/electronics_and_hardware_low": 0.008651088352306352,
      "dolma3_cc/entertainment_high": 0.05878282538330399,
      "dolma3_cc/entertainment_low": 0.021391802833895898,
      "dolma3_cc/finance_and_business_high": 0.07551908502389648,
      "dolma3_cc/finance_and_business_low": 0.03598421324951779,
      "dolma3_cc/food_and_dining_high": 0.023717815870374904,
      "dolma3_cc/food_and_dining_low": 0.011674326757492082,
      "dolma3_cc/games_high": 0.031367838821708245,
      "dolma3_cc/games_low": 0.011760417933417037,
      "dolma3_cc/health_high": 0.06045654273845028,
      "dolma3_cc/health_low": 0.02576663627013856,
      "dolma3_cc/history_and_geography_high": 0.018757703697866782,
      "dolma3_cc/history_and_geography_low": 0.005162040050741883,
      "dolma3_cc/industrial_high": 0.01133868092628134,
      "dolma3_cc/industrial_low": 0.0060034932433818905,
      "dolma3_cc/literature_high": 0.03586571270590475,
      "dolma3_cc/literature_low": 0.009562894385702854,
      "dolma3_cc/science_math_and_technology_high": 0.0337411324938567,
      "dolma3_cc/science_math_and_technology_low": 0.01540234212159164,
      "dolma3_finemath_3plus": 0.004720513733089685,
      "dolma3_stack_edu": 0.018613227076727863,
      "dolma3_wikipedia": 0.0005093903672490581,
      "dolma_starcoder": 0.030066268737441462,
      "dolmino_common_crawl_hq": 0.1829219022364885,
      "dolmino_olmocr_pdfs_hq": 0.028585647143263525,
      "dolmino_stack_edu_fim": 0.018587128637112035,
      "dolmino_stem_heavy_crawl": 0.0007238309076637704,
      "dolmino_synth_code": 0.002618466151286209,
      "dolmino_synth_instruction": 0.0024962431278630183,
      "dolmino_synth_math": 0.003029417977275237,
      "dolmino_synth_qa": 0.07318525947741508,
      "dolmino_synth_thinking": 0.0055390695233902875
    },
    "phase_1": {
      "dolma3_arxiv": 0.003920251599600697,
      "dolma3_cc/art_and_design_high": 0.01585036607982536,
      "dolma3_cc/art_and_design_low": 0.006433483374358941,
      "dolma3_cc/crime_and_law_high": 0.021334820548967248,
      "dolma3_cc/crime_and_law_low": 0.009583728944191527,
      "dolma3_cc/education_and_jobs_high": 0.034882814666569374,
      "dolma3_cc/education_and_jobs_low": 0.017086415643936335,
      "dolma3_cc/electronics_and_hardware_high": 0.018405161186455334,
      "dolma3_cc/electronics_and_hardware_low": 0.008651088352306352,
      "dolma3_cc/entertainment_high": 0.05878282538330399,
      "dolma3_cc/entertainment_low": 0.021391802833895898,
      "dolma3_cc/finance_and_business_high": 0.07551908502389648,
      "dolma3_cc/finance_and_business_low": 0.03598421324951779,
      "dolma3_cc/food_and_dining_high": 0.023717815870374904,
      "dolma3_cc/food_and_dining_low": 0.011674326757492082,
      "dolma3_cc/games_high": 0.031367838821708245,
      "dolma3_cc/games_low": 0.011760417933417037,
      "dolma3_cc/health_high": 0.06045654273845028,
      "dolma3_cc/health_low": 0.02576663627013856,
      "dolma3_cc/history_and_geography_high": 0.018757703697866782,
      "dolma3_cc/history_and_geography_low": 0.005162040050741883,
      "dolma3_cc/industrial_high": 0.01133868092628134,
      "dolma3_cc/industrial_low": 0.0060034932433818905,
      "dolma3_cc/literature_high": 0.03586571270590475,
      "dolma3_cc/literature_low": 0.009562894385702854,
      "dolma3_cc/science_math_and_technology_high": 0.0337411324938567,
      "dolma3_cc/science_math_and_technology_low": 0.01540234212159164,
      "dolma3_finemath_3plus": 0.004720513733089685,
      "dolma3_stack_edu": 0.018613227076727863,
      "dolma3_wikipedia": 0.0005093903672490581,
      "dolma_starcoder": 0.030066268737441462,
      "dolmino_common_crawl_hq": 0.1829219022364885,
      "dolmino_olmocr_pdfs_hq": 0.028585647143263525,
      "dolmino_stack_edu_fim": 0.018587128637112035,
      "dolmino_stem_heavy_crawl": 0.0007238309076637704,
      "dolmino_synth_code": 0.002618466151286209,
      "dolmino_synth_instruction": 0.0024962431278630183,
      "dolmino_synth_math": 0.003029417977275237,
      "dolmino_synth_qa": 0.07318525947741508,
      "dolmino_synth_thinking": 0.0055390695233902875
    }
  }
}
"""
)


if __name__ == "__main__":
    main()
