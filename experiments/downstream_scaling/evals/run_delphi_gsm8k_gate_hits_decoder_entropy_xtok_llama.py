# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gate-hit counts for the delphi GSM8K joint-decode-decoder-entropy-xtok sweep.

Per (slug, threshold, problem, sample): how many of the completion's recorded
decision steps carried a decoder entropy strictly below the threshold. That is
the decoder's share of the completion, which separates "how much llama help did
this model get" from the accuracy curve it produced.

Nothing is regenerated. The sweep's completions steps are recomposed so the
executor resolves their output paths and sees them as already done; each
statistic step reads the ``token_paths.jsonl.gz`` sidecar those runs wrote.
Launch with ``--prefix mirror://`` so the sidecars resolve wherever they live.
"""

from __future__ import annotations

from fray.cluster import ANY_REGION
from rigging.log_setup import configure_logging
from thalas.execution.executor import ExecutorStep, executor_main, output_path_of

from experiments.downstream_scaling.evals import run_delphi_gsm8k_joint_decode_decoder_entropy_xtok_llama as sweep
from experiments.downstream_scaling.evals.algorithms.joint_decode_entropy_xtok import joint_decode_pool_configs
from experiments.downstream_scaling.evals.framework.core import make_eval_step
from experiments.downstream_scaling.evals.framework.schema import PROMPTS_FILENAME
from experiments.downstream_scaling.evals.measurements.gate_hits import GateHits
from experiments.downstream_scaling.models.delphi import DELPHI_HF_DOWNLOADS

# The per-step signal key in this sweep's token-path sidecar. The key is the
# same as the advisor-gated sweep's; under EntropySource.DECODER it holds the
# decoder's top-k entropy rather than the advisor's.
SIGNAL_FIELD = "entropy"

# The sweep runs GateDirection.ADVISOR_AT_OR_ABOVE — llama takes the round when
# the signal is >= threshold — so the strictly-below count GateHits emits is
# delphi's share of the completion.


def build_steps() -> list[ExecutorStep]:
    # Placeholder pools: worker-pool config is unversioned, so any placement
    # resolves the same completions-step hashes as the production run.
    pools = sweep.make_worker_pools(
        tpu_types=list(sweep.TPU_TYPES),
        worker_regions=[ANY_REGION],
        num_workers=sweep.WORKERS_PER_TPU_TYPE,
    )
    prompts_path = output_path_of(sweep.make_task().make_prompts_step()) / PROMPTS_FILENAME
    advisor_model_path = output_path_of(sweep.LLAMA_3_1_8B)

    steps: list[ExecutorStep] = []
    for slug in DELPHI_HF_DOWNLOADS:
        if slug in sweep.SKIP_DELPHI_KEYS:
            continue
        completions = make_eval_step(
            name=f"{sweep.EVAL_NAME_BASE}/{slug}",
            model_path=output_path_of(DELPHI_HF_DOWNLOADS[slug]),
            task=sweep.make_task(),
            alg=sweep.make_algorithm(
                worker_pools=joint_decode_pool_configs(slug, pools, sweep.PLACEMENT_OVERRIDES),
                microbatch_size=sweep.MICROBATCH_SIZE_BY_DELPHI_KEY.get(slug),
                advisor_model_path=advisor_model_path,
            ),
            skip_grades=True,
        )
        steps.extend(
            GateHits(field=SIGNAL_FIELD, threshold=threshold).make_statistic_step(
                # Indexed rather than formatted from the float: path-safe and
                # unambiguous. The value lives in the versioned config and in
                # every emitted row's metadata.
                name=f"{sweep.EVAL_NAME_BASE}/{slug}/hits_t{index}",
                prompts_path=prompts_path,
                alg_output_path=output_path_of(completions),
            )
            for index, threshold in enumerate(sweep.ENTROPY_THRESHOLDS)
        )
    return steps


if __name__ == "__main__":
    configure_logging()
    executor_main(
        steps=build_steps(),
        description=(
            "Per-completion gate-hit counts for the delphi GSM8K decoder-entropy-gated "
            "joint-decode sweep (one statistic step per slug and entropy threshold)."
        ),
    )
