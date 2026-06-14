# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inline-execution variant of ``run_delphi_masked_gsm8k_iid_v2``.

Same eval as the v2 script (per-rollout masked GSM8K), but the completion step
runs in-process via :class:`InlineIIDCompletionAlgorithm` rather than being
dispatched as a Fray TPU job. Intended to be launched as N replica TPU tasks
(``iris job run --tpu … --replicas N`` with no ``--region`` and ``--prefix
mirror://``) so each replica runs the executor on its own TPU and the work lands
wherever iris schedules it; ``mirror://`` dedups across replicas.

The TPU is requested once on the iris job (``--tpu``); the script itself names
no TPU. Step names match the dispatched v2 script, so the two share output paths.
"""

from __future__ import annotations

from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main, output_path_of

from experiments.downstream_scaling.evals.algorithms.iid import (
    IIDConfig,
    IIDExecutionConfig,
    IIDSamplingConfig,
)
from experiments.downstream_scaling.evals.algorithms.iid_inline import InlineIIDCompletionAlgorithm
from experiments.downstream_scaling.evals.framework.core import make_eval_step
from experiments.downstream_scaling.evals.tasks.gsm8k_masked_iid import MaskedGSM8KIIDTask, MaskedGSM8KIIDTaskConfig
from experiments.downstream_scaling.models.delphi import DELPHI_HF_DOWNLOADS
from experiments.llama import llama3_tokenizer

N_SAMPLES = 64
N_PROBLEMS = 256
NUM_WORKERS = 1

TEMPERATURE = 0.6
TOP_P = 1.0
TOP_K = 1000
MAX_TOKENS = 512
SEED = 42
STOP_TOKENS = ("Question:", "</s>", "<|im_end|>")

NUM_FEWSHOT = 5
FEWSHOT_SEED = 1234
MASK_FRACTIONS = tuple(i / 10 for i in range(11))
MASK_TEXT = "<mask>"


def make_task(mask_fraction: float) -> MaskedGSM8KIIDTask:
    return MaskedGSM8KIIDTask(
        config=MaskedGSM8KIIDTaskConfig(
            tokenizer_path=llama3_tokenizer,
            num_fewshot=NUM_FEWSHOT,
            fewshot_seed=FEWSHOT_SEED,
            n_problems=N_PROBLEMS,
            n_masks=N_SAMPLES,
            mask_fraction=mask_fraction,
            mask_text=MASK_TEXT,
        )
    )


def make_algorithm() -> InlineIIDCompletionAlgorithm:
    return InlineIIDCompletionAlgorithm(
        config=IIDConfig(
            sampling=IIDSamplingConfig(
                n_samples=1,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                max_tokens=MAX_TOKENS,
                seed=SEED,
                stop=STOP_TOKENS,
            ),
            execution=IIDExecutionConfig(
                num_workers=NUM_WORKERS,
                # Inert in inline mode: the step runs in-process on the replica task's own
                # TPU (from the iris job's --tpu); worker_resources is never read or hashed.
                worker_resources=ResourceConfig.with_cpu(),
            ),
        )
    )


def build_steps():
    return [
        make_eval_step(
            name=f"downstream_scaling/evals/delphi/masked_gsm8k/iid_v2/mask_{i:02d}/{slug}",
            model_path=output_path_of(checkpoint),
            task=make_task(mask_fraction),
            alg=make_algorithm(),
        )
        for i, mask_fraction in enumerate(MASK_FRACTIONS)
        for slug, checkpoint in DELPHI_HF_DOWNLOADS.items()
    ]


if __name__ == "__main__":
    executor_main(
        steps=build_steps(),
        description="Inline per-rollout masked GSM8K IID evals (Delphi ladder), run from replica TPU tasks.",
    )
