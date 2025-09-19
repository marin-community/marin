import math
import random

from dataclasses import replace

from experiments.defaults import default_train, default_tokenize
from experiments.evals.task_configs import BEYOND_WEB_TASKS
from experiments.llama import llama_150m, llama3_tokenizer
from levanter.data.text import TextLmDatasetFormat
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


def _make_train_config(total_tokens: int = 2_400_000_000) -> SimpleTrainConfig:
    # Keep batch/seq modest and use small TPU for economical trials
    batch_size = 256
    seq_len = llama_150m.seq_len
    num_steps = _compute_num_steps(total_tokens, batch_size, seq_len)

    return SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type="v4-8"),
        train_batch_size=batch_size,
        num_train_steps=num_steps,
        learning_rate=1e-4,
        weight_decay=1e-7,
        warmup=0.05,
        steps_per_task_eval=1000,
    )


def _trial_steps(num_trials: int = 25, seed: int = 20250919) -> list[ExecutorStep]:
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
        weights = _dirichlet_weights_from_counts(TOKEN_COUNTS, concentration_scale=10.0, rng=rng)

        data_cfg = lm_mixture_data_config(
            components=components,
            weights=weights,
        )

        trial_name = f"llama-150m-regmix-t{i:02d}"

        train_cfg = replace(base_train_cfg)

        train_step = default_train(
            name=trial_name,
            tokenized=data_cfg,
            model_config=llama_150m,
            train_config=train_cfg,
            tags=["regmix", "150m", "2.4b", *[f"{k}={weights[k]:.3f}" for k in components.keys()]],
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


if __name__ == "__main__":
    trials = _trial_steps(num_trials=25)
    executor_main(steps=trials)
