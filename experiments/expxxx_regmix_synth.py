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

import math
import random

from dataclasses import replace

from experiments.defaults import default_train, default_tokenize
from experiments.evals.task_configs import BEYOND_WEB_TASKS
from experiments.llama import llama3_tokenizer
from levanter.data.text import TextLmDatasetFormat
from levanter.models.llama import LlamaConfig
from marin.execution.executor import ExecutorStep, executor_main, InputName
from marin.processing.tokenize.data_configs import lm_mixture_data_config
from marin.resources import TpuPodConfig
from experiments.simple_train_config import SimpleTrainConfig

wrapqa = InputName.hardcoded("gs://marin-us-central2/documents/synthetic-dclm-subset-chunk-qa-1024-wrapqa-eb7854")
regular_text_tokenized = default_tokenize(
    name="dclm-regular-text-1024",
    dataset=wrapqa,
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(text_key="text"),  # text key is the normal text key
)

wrapqa_tokenized = default_tokenize(
    name="dclm-wrap-qa-1024",
    dataset=wrapqa,
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(text_key="generated_text"),
)

nemo_qa = InputName.hardcoded("gs://marin-us-central2/documents/synthetic-dclm-subset-chunk-qa-1024-nemo-qa-f36058")
nemo_qa_tokenized = default_tokenize(
    name="dclm-nemo-qa-1024",
    dataset=nemo_qa,
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(text_key="generated_text"),
)
wrapmed = InputName.hardcoded("gs://marin-us-central2/documents/synthetic-dclm-subset-chunk-qa-1024-wrapmed-3e0d3e")
wrapmed_tokenized = default_tokenize(
    name="dclm-wrap-med-1024",
    dataset=wrapmed,
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(text_key="generated_text"),
)

mcq = InputName.hardcoded("gs://marin-us-central2/documents/synthetic-dclm-subset-chunk-qa-1024-mcq-ad37b6")
mcq_tokenized = default_tokenize(
    name="dclm-mcq-1024",
    dataset=mcq,
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(text_key="generated_text"),
)

# Five dataset sources and their token counts (used to shape the Dirichlet prior)
TOKEN_COUNTS = {
    "regular_text": 2_692_565_446,
    "wrap_qa": 3_649_246_345,
    "wrap_med": 2_336_223_848,
    "mcq": 1_640_002_900,
    "nemo_qa": 3_626_282_543,
}

llama_130m = LlamaConfig(
    seq_len=4096,
    hidden_dim=512,
    intermediate_dim=2048,
    num_heads=8,
    num_kv_heads=8,
    num_layers=32,
)


def _dirichlet_weights_from_counts(
    counts: dict[str, int], concentration_scale: float, rng: random.Random
) -> dict[str, float]:
    total = float(sum(counts.values()))
    alphas = [max(1e-3, concentration_scale * (c / total)) for c in counts.values()]
    samples = _dirichlet_sample(alphas, rng)
    return {k: v for k, v in zip(counts.keys(), samples, strict=False)}


def _dirichlet_sample(alphas: list[float], rng: random.Random) -> list[float]:
    gammas = [rng.gammavariate(a, 1.0) for a in alphas]
    s = sum(gammas)
    if s <= 0:
        n = len(gammas)
        return [1.0 / n for _ in gammas]
    return [g / s for g in gammas]


def _compute_num_steps(total_tokens: int, batch_size: int, seq_len: int) -> int:
    tokens_per_step = batch_size * seq_len
    return math.ceil(total_tokens / tokens_per_step)


def _make_train_config(total_tokens: int = 2_600_000_000) -> SimpleTrainConfig:
    # Keep batch/seq modest and use small TPU for economical trials
    # for the 150m trials
    # batch_size = 256
    # seq_len = 1024

    original_batch_size = 128  # from the pretraning optimizers paper
    batch_size = 64
    # seq_len = 4096
    # seq_len = llama_150m.seq_len
    seq_len = llama_130m.seq_len
    num_steps = _compute_num_steps(total_tokens, batch_size, seq_len)

    return SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type="v4-8"),
        train_batch_size=batch_size,
        num_train_steps=num_steps,
        learning_rate=0.008 * math.sqrt(batch_size) / math.sqrt(original_batch_size),
        beta1=0.9,
        beta2=0.98 ** (original_batch_size / batch_size),  # beta2 tuning
        weight_decay=0.1,
        max_grad_norm=1.0,
        warmup=0.05,
        steps_per_task_eval=1000,
    )


def _trial_steps(
    num_trials: int = 25,
    seed: int = 20250919,
    vary_concentration: bool = True,
    concentration_min: float = 0.1,  # from regmix
    concentration_max: float = 5.0,  # from regmix
    log_uniform: bool = False,
    fixed_concentration: float = 10.0,
) -> list[ExecutorStep]:
    rng = random.Random(seed)

    components = {
        "regular_text": regular_text_tokenized,
        "wrap_qa": wrapqa_tokenized,
        "wrap_med": wrapmed_tokenized,
        "mcq": mcq_tokenized,
        "nemo_qa": nemo_qa_tokenized,
    }

    steps: list[ExecutorStep] = []

    base_train_cfg = _make_train_config()

    for i in range(num_trials):
        # Choose Dirichlet concentration scale per trial. When varying, we sample
        # either linearly or log-uniformly between [concentration_min, concentration_max].
        if vary_concentration:
            if log_uniform:
                # sample log-uniform in [min, max]
                c = 10 ** rng.uniform(math.log10(concentration_min), math.log10(concentration_max))
            else:
                c = rng.uniform(concentration_min, concentration_max)
        else:
            c = fixed_concentration

        weights = _dirichlet_weights_from_counts(TOKEN_COUNTS, concentration_scale=c, rng=rng)

        data_cfg = lm_mixture_data_config(
            components=components,
            weights=weights,
        )

        trial_name = f"llama-130m-regmix-v3p1-t{i:02d}"

        train_cfg = replace(base_train_cfg)

        train_step = default_train(
            name=trial_name,
            tokenized=data_cfg,
            model_config=llama_130m,
            train_config=train_cfg,
            tags=[
                "regmix",
                "130m",
                "2.6b",
                f"lr={train_cfg.learning_rate:.3f}",
                f"alpha_scale={c:.3f}",
                *[f"{k}={weights[k]:.3f}" for k in components.keys()],
            ],
            only_return_config=False,
            eval_harness_tasks=BEYOND_WEB_TASKS,
        )

        # eval_step = evaluate_levanter_lm_evaluation_harness(
        #     model_name=trial_name,
        #     model_path=output_path_of(train_step),
        #     evals=BEYOND_WEB_TASKS,
        #     resource_config=SINGLE_TPU_V4_8,
        # )

        steps.append(train_step)

    return steps


def chunk_steps(steps):
    return [steps[i : i + 120] for i in range(0, len(steps), 120)]


if __name__ == "__main__":
    trials = _trial_steps(
        num_trials=200,
        vary_concentration=True,
        concentration_min=0.1,
        concentration_max=5.0,
        log_uniform=False,
        seed=20250922,
    )

    for chunk_step in chunk_steps(trials):
        executor_main(steps=chunk_step)
