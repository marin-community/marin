# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate from isoflop_v2 checkpoints using held-out Nemotron prompts.

Samples prompts from the validation split of Nemotron CC hq_actual. The
training runs hold out 1024 sequences via num_validation_sequences; this
script reconstructs the same validation split using the same LmDataConfig
code path (Feistel shuffle with PRNGKey(0), last 1024 sequences) and
samples prompts from those held-out sequences.

Pipeline:
  1. Reconstruct the validation split of hq_actual and sample prompts.
  2. For each training checkpoint from the IID and rewarm experiments,
     generate completions using vLLM.
  3. Optionally, score generations under a reference model using
     Levanter's eval pipeline (log-likelihood).
"""

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass

import fsspec
import numpy as np

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer
from experiments.models import ModelConfig as HFModelConfig, download_model_step
from experiments.pretraining_datasets.nemotron import tokenize_nemotron
from fray.cluster import ResourceConfig
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.data.text import LmDataConfig
from marin.evaluation.log_probs import default_lm_log_probs
from marin.execution.executor import (
    ExecutorStep,
    InputName,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)
from marin.execution.remote import remote
from marin.processing.tokenize import lm_mixture_data_config
from marin.processing.tokenize.data_configs import mixture_for_evaluation
from marin.utils import fsspec_exists
from marin.evaluation.utils import discover_hf_checkpoints

logger = logging.getLogger("ray")

# Must match the training experiments
SEQ_LEN = 1024
NUM_VALIDATION_SEQUENCES = 1024

NUM_PROMPTS = 512
PROMPT_MAX_TOKENS = 128
PROMPT_SEED = 137

# ---------------------------------------------------------------------------
# Validation data config: same split as training
# ---------------------------------------------------------------------------

nemotron_steps = tokenize_nemotron()
high_q = nemotron_steps["nemotron_cc/hq_actual"]

val_data_config = lm_mixture_data_config(
    components={"high": high_q},
    weights={"high": 1.0},
    num_validation_sequences={"high": NUM_VALIDATION_SEQUENCES},
)


@dataclass(frozen=True)
class SamplePromptsConfig:
    """Config for sampling prompts from the hq_actual validation split."""

    data_config: LmDataConfig
    tokenizer: str
    output_path: str
    seq_len: int = SEQ_LEN
    num_prompts: int = NUM_PROMPTS
    prompt_max_tokens: int = PROMPT_MAX_TOKENS
    seed: int = PROMPT_SEED


def sample_held_out_prompts(config: SamplePromptsConfig):
    """Sample prompts from the validation split of the tokenized hq_actual data."""
    from transformers import AutoTokenizer

    val_datasets = config.data_config.validation_grug_sets(seq_len=config.seq_len)
    sync_ds = val_datasets["high"].as_sync_dataset()

    num_available = len(sync_ds)
    if num_available < config.num_prompts:
        raise RuntimeError(f"Only {num_available} validation sequences available, need {config.num_prompts}")

    rng = random.Random(config.seed)
    indices = rng.sample(range(num_available), k=config.num_prompts)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)

    output_file = os.path.join(config.output_path, "prompts.jsonl.gz")
    with fsspec.open(output_file, "wt", compression="gzip") as f:
        for i, idx in enumerate(indices):
            tokens = np.asarray(sync_ds[idx].tokens)
            prefix = tokens[: config.prompt_max_tokens]
            text = tokenizer.decode(prefix, skip_special_tokens=True)
            f.write(json.dumps({"id": i, "prompt": text}) + "\n")

    logger.info(f"Saved {config.num_prompts} prompts to {output_file}")


sample_prompts_step = ExecutorStep(
    name="dq-iso-generate/held-out-prompts",
    fn=remote(sample_held_out_prompts, resources=ResourceConfig.with_cpu()),
    config=SamplePromptsConfig(
        data_config=val_data_config,
        tokenizer=versioned(llama3_tokenizer),
        output_path=this_output_path(),
    ),
)


# ---------------------------------------------------------------------------
# Generation from checkpoints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerateConfig:
    """Config for generating completions from a checkpoint."""

    model_path: str | InputName
    prompt_path: str | InputName
    output_path: str
    max_tokens: int = 256
    temperature: float = 0.8
    n_samples: int = 1
    batch_size: int = 50


def generate(config: GenerateConfig):
    """Generate completions from a checkpoint using vLLM."""
    from vllm import LLM, SamplingParams

    prompt_file = os.path.join(config.prompt_path, "prompts.jsonl.gz")
    prompts: list[dict] = []
    with fsspec.open(prompt_file, "rt", compression="gzip") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))

    logger.info(f"Loaded {len(prompts)} prompts")

    hf_checkpoints = discover_hf_checkpoints(config.model_path)
    if not hf_checkpoints:
        raise RuntimeError(f"No HF checkpoints found under {config.model_path}")
    latest_model_path = sorted(hf_checkpoints, key=lambda p: int(p.rsplit("step-", 1)[-1]))[-1]
    logger.info(f"Using HF checkpoint: {latest_model_path}")

    llm = LLM(
        model=latest_model_path,
        trust_remote_code=True,
        load_format="runai_streamer",
        hf_overrides={"architectures": ["Qwen3ForCausalLM"]},
    )

    sampling_params = SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        n=config.n_samples,
    )

    all_prompt_texts = [p["prompt"] for p in prompts]

    for batch_start in range(0, len(all_prompt_texts), config.batch_size):
        batch_file = os.path.join(config.output_path, f"results-batch-{batch_start}.json.gz")
        if fsspec_exists(batch_file):
            continue

        batch_prompts = all_prompt_texts[batch_start : batch_start + config.batch_size]
        batch_meta = prompts[batch_start : batch_start + config.batch_size]

        outputs = llm.generate(batch_prompts, sampling_params)

        results = []
        for i, output in enumerate(outputs):
            samples = [{"generated": completion.text} for completion in output.outputs]
            results.append(
                {
                    "id": batch_meta[i]["id"],
                    "prompt": batch_prompts[i],
                    "samples": samples,
                }
            )

        with fsspec.open(batch_file, "wt", compression="gzip") as f:
            json.dump(results, f, indent=2)

    all_results: list[dict] = []
    for batch_start in range(0, len(all_prompt_texts), config.batch_size):
        batch_file = os.path.join(config.output_path, f"results-batch-{batch_start}.json.gz")
        with fsspec.open(batch_file, "rt", compression="gzip") as f:
            all_results.extend(json.load(f))

    with fsspec.open(os.path.join(config.output_path, "results.json.gz"), "wt", compression="gzip") as f:
        json.dump({"results": all_results}, f, indent=2)

    logger.info(f"Generated completions for {len(all_results)} prompts")


# ---------------------------------------------------------------------------
# Convert generations to JSONL for scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvertGenerationsConfig:
    results_path: str | InputName
    output_path: str


def convert_generations_to_jsonl(config: ConvertGenerationsConfig):
    """Convert generation results to JSONL for downstream tokenization and scoring."""
    with fsspec.open(os.path.join(config.results_path, "results.json.gz"), "rt", compression="gzip") as f:
        data = json.load(f)

    output_file = os.path.join(config.output_path, "generations.jsonl.gz")
    with fsspec.open(output_file, "wt", compression="gzip") as f:
        for result in data["results"]:
            for sample in result["samples"]:
                text = result["prompt"] + sample["generated"]
                f.write(json.dumps({"text": text}) + "\n")


# ---------------------------------------------------------------------------
# Step construction
# ---------------------------------------------------------------------------

DEFAULT_TPU_TYPE = "v4-8"


def make_generate_step(
    train_step: ExecutorStep,
    name: str,
    tpu_type: str = DEFAULT_TPU_TYPE,
) -> ExecutorStep:
    """Create a generation step for a training checkpoint."""
    return ExecutorStep(
        name=os.path.join("dq-iso-generate", name),
        fn=remote(generate, resources=ResourceConfig.with_tpu(tpu_type), pip_dependency_groups=["vllm"]),
        config=GenerateConfig(
            model_path=output_path_of(train_step, "hf"),
            prompt_path=output_path_of(sample_prompts_step),
            output_path=this_output_path(),
            temperature=versioned(0.8),
            max_tokens=versioned(256),
            n_samples=versioned(1),
        ),
    )


def tokenize_generations_for_scoring(
    gen_steps: dict[str, ExecutorStep],
    tokenizer: str,
) -> dict[str, ExecutorStep]:
    """Convert and tokenize generation outputs for log-probability scoring."""
    tokenized: dict[str, ExecutorStep] = {}
    for name, gen_step in gen_steps.items():
        convert_step = ExecutorStep(
            name=os.path.join("dq-iso-score", name, "jsonl"),
            fn=remote(convert_generations_to_jsonl, resources=ResourceConfig.with_cpu()),
            config=ConvertGenerationsConfig(
                results_path=output_path_of(gen_step),
                output_path=this_output_path(),
            ),
        )
        tokenized[name] = default_tokenize(
            name=os.path.join("dq-iso-score", name),
            dataset=output_path_of(convert_step),
            tokenizer=tokenizer,
            is_validation=True,
        )
    return tokenized


def build_steps(tpu_type: str = DEFAULT_TPU_TYPE, scoring_model: str | None = None) -> list[ExecutorStep]:
    """Build generation steps for all isoflop_v2 training checkpoints."""
    # Import training modules here to avoid their module-level side effects
    # (GCS globs and argparse) from running at import time of this module.
    import experiments.data_quality_scaling.isoflop_v2.exp_data_quality_scaling_warc_sweep_iid as iid_exp
    import experiments.data_quality_scaling.isoflop_v2.exp_data_quality_scaling_warc_sweep_rewarm as rewarm_exp

    all_steps: list[ExecutorStep] = [sample_prompts_step]
    all_gen_steps: dict[str, ExecutorStep] = {}

    for step in iid_exp.all_steps:
        if "dq-iso-warc-iid-train-" in step.name:
            gen_step = make_generate_step(step, step.name, tpu_type=tpu_type)
            all_steps.append(gen_step)
            all_gen_steps[step.name] = gen_step

    for step in rewarm_exp.all_steps:
        if "dq-iso-warc-rewarm-train-" in step.name:
            gen_step = make_generate_step(step, step.name, tpu_type=tpu_type)
            all_steps.append(gen_step)
            all_gen_steps[step.name] = gen_step

    if scoring_model:
        model_instance = download_model_step(HFModelConfig(hf_repo_id=scoring_model, hf_revision="main"))
        scoring_model_config = HFCheckpointConverter.from_hf(scoring_model).config_from_hf_checkpoint(scoring_model)

        tokenized_dict = tokenize_generations_for_scoring(all_gen_steps, scoring_model)
        eval_data = mixture_for_evaluation(tokenized_dict)

        all_steps.append(
            default_lm_log_probs(
                checkpoint=output_path_of(model_instance),
                model=scoring_model_config,
                data=eval_data,
                resource_config=ResourceConfig.with_tpu(tpu_type),
                checkpoint_is_hf=True,
                name="dq-iso-score-all",
            )
        )

    return all_steps


parser = argparse.ArgumentParser(description="Generate from isoflop_v2 checkpoints.")
parser.add_argument(
    "--tpu-type",
    type=str,
    default=DEFAULT_TPU_TYPE,
    help=f"TPU type for generation steps (default {DEFAULT_TPU_TYPE}).",
)
parser.add_argument(
    "--scoring-model",
    type=str,
    default="marin-community/marin-8b-base",
    help="HF repo ID for scoring model (computes log-likelihoods of generations).",
)
args, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0], *remaining]

all_steps = build_steps(tpu_type=args.tpu_type, scoring_model=args.scoring_model)

if __name__ == "__main__":
    executor_main(steps=all_steps)
