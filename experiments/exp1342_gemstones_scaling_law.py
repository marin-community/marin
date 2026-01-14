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

"""
Gemstone Models Configuration

This file contains configurations for all 22 Gemstone models from the tomg-group-umd collection.
These models range from 50M to 2B parameters, spanning 11 widths (256 to 3072) and 18 depths (3 to 80).

Usage:
1. Import the model step you want to use
2. Run executor_main([model_step]) to download
3. Use the model step's output path for downstream jobs

Example:
```
from gemstones import gemstone_768x45
from marin.execution.executor import executor_main

executor_main([gemstone_768x45])
model_path = gemstone_768x45  # Use step directly or step / "subpath"
```
"""

# nodryrun

import re
from dataclasses import dataclass

from datasets import get_dataset_config_names, get_dataset_split_names
from huggingface_hub import get_collection, list_repo_refs
from levanter.compat.hf_checkpoints import HFCheckpointConverter

from experiments.defaults import default_tokenize, default_validation_sets
from experiments.models import ModelConfig, download_model_step
from fray.cluster import ResourceConfig
from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.evaluation.log_probs import default_lm_log_probs
from marin.execution.executor import executor_main, versioned
from marin.execution import step, StepContext, StepRef
from marin.processing.tokenize.data_configs import mixture_for_evaluation
from marin.transform.huggingface.dataset_to_eval import DatasetConversionConfig, OutputFormatOptions, hf_dataset_to_jsonl

# Unfortunately, the international corpus of english is not publicly accessible and cannot be redistributed.
# By default, this experiment runs without ICE since it cannot be shared more widely.
CAN_ACCESS_ICE = False


@dataclass(frozen=True)
class GemstoneConfig:
    """Hashable configuration for Gemstone models."""

    width: int
    depth: int
    model_id: str
    revision: str
    variant: str  # "main", "lr_ablation", or "cooldown"
    step: int | None = None
    cooldown_start_step: int | None = None

    @classmethod
    def from_model_revision(cls, model_id: str, revision: str) -> "GemstoneConfig":
        """Parse model ID and revision into config."""
        # Parse model name: tomg-group-umd/Gemstone-{width}x{depth}[_variant]
        model_match = re.match(r"tomg-group-umd/Gemstone-(\d+)x(\d+)(?:_(.+))?", model_id)
        if not model_match:
            raise ValueError(f"Invalid model ID format: {model_id}")

        width = int(model_match.group(1))
        depth = int(model_match.group(2))
        variant_suffix = model_match.group(3) or ""

        # Determine variant
        if variant_suffix == "lr_ablation":
            variant = "lr_ablation"
        elif variant_suffix == "cooldown":
            variant = "cooldown"
        else:
            variant = "main"

        # Parse revision
        step = None
        cooldown_start_step = None

        if revision == "main":
            # Main branch - no step info
            pass
        elif revision.startswith("step_"):
            if "_cooldown_" in revision:
                # step_{x}_cooldown_{y}
                cooldown_match = re.match(r"step_(\d+)_cooldown_(\d+)", revision)
                if cooldown_match:
                    cooldown_start_step = int(cooldown_match.group(1))
                    step = int(cooldown_match.group(2))
            else:
                # step_{x}
                step_match = re.match(r"step_(\d+)", revision)
                if step_match:
                    step = int(step_match.group(1))

        return cls(
            width=width,
            depth=depth,
            variant=variant,
            step=step,
            cooldown_start_step=cooldown_start_step,
            model_id=model_id,
            revision=revision,
        )

    def __hash__(self):
        return hash((self.width, self.depth, self.variant, self.step, self.cooldown_start_step))

    def __str__(self):
        base = f"{self.width}x{self.depth}"
        if self.variant != "main":
            base += f"_{self.variant}"

        if self.step is not None:
            if self.cooldown_start_step is not None:
                base += f"@step_{self.cooldown_start_step}_cooldown_{self.step}"
            else:
                base += f"@step_{self.step}"
        elif self.variant == "cooldown":
            base += "@main"

        return base


def get_gemstone_model_ids():
    """
    Get all Gemstone model repository IDs from the tomg-group-umd collection.

    Returns:
        list: All Gemstone model repository IDs
    """

    collection_id = "tomg-group-umd/gemstone-models-679408ee3f19f1d4d00e8b10"
    collection = get_collection(collection_id)

    model_ids = []
    for item in collection.items:
        if hasattr(item, "item_id") and hasattr(item, "item_type") and item.item_type == "model":
            model_ids.append(item.item_id)

    return model_ids


def get_all_revisions(repo_id):
    """
    Get all revision names (branches and tags) for a Hugging Face repository.

    Args:
        repo_id (str): Repository ID (e.g., 'microsoft/DialoGPT-medium')
        repo_type (str): Type of repo ('model', 'dataset', 'space')
        token (str, optional): HF API token for private repos

    Returns:
        list: List of revision names (branch names and tag names)
    """

    refs = list_repo_refs(repo_id, repo_type="model")
    branches = [branch.name for branch in refs.branches]
    tags = [tag.name for tag in refs.tags]
    combined = [tag for tag in branches + tags if "step" in tag]

    combined = sorted(combined, key=lambda x: int(x.split("step_")[-1].split("cooldown_")[-1]))
    return combined


def distributional_eval_sets(tokenizer):
    eval_sets = default_validation_sets(tokenizer=versioned(tokenizer))

    @step(name="raw/WillHeld/MD3", fn=download_hf)
    def md3_raw_step(ctx: StepContext):
        return DownloadConfig(
            hf_dataset_id="WillHeld/MD3",
            revision=versioned("7c74e59"),
            gcs_output_path=ctx.output,
            wait_for_completion=True,
            hf_urls_glob=["**/*.parquet"],
        )

    md3_raw = md3_raw_step().with_output_path("raw/WillHeld/MD3-Trunc").cd("7c74e59/huggingface.co/datasets/WillHeld/MD3/resolve/7c74e59")

    tokenized_domains = {}
    for split in get_dataset_split_names("WillHeld/MD3"):
        @step(name=f"hf_dataset_to_jsonl/md3/{split}", fn=hf_dataset_to_jsonl)
        def json_step(ctx: StepContext, md3_raw=md3_raw, split=split):
            return DatasetConversionConfig(
                dataset_name="WillHeld/MD3",
                subsets=["*"],
                splits=[split],
                input_path=ctx.require(md3_raw),
                hf_path="WillHeld/MD3",
                output_path=ctx.output,
                output_format=OutputFormatOptions("decontamination"),
                prompt_key="transcript",
            )

        tokenized_json = default_tokenize(f"tokenized/md3-{split}", json_step(), tokenizer, is_validation=True)
        tokenized_domains[f"md3/{split}"] = tokenized_json

    if CAN_ACCESS_ICE:
        @step(name="raw/WillHeld/ICE", fn=download_hf)
        def ice_raw_step(ctx: StepContext):
            return DownloadConfig(
                hf_dataset_id="WillHeld/ICE_Cleaned",
                revision=versioned("4c09dd9"),
                gcs_output_path=ctx.output,
                wait_for_completion=True,
                hf_urls_glob=["**/*.parquet"],
            )

        ice_raw = ice_raw_step().with_output_path("raw/WillHeld/ICE_Cleaned").cd("4c09dd9/huggingface.co/datasets/WillHeld/ICE_Cleaned/resolve/4c09dd9")

        for split in get_dataset_split_names("WillHeld/ICE_Cleaned"):
            @step(name=f"hf_dataset_to_jsonl/ICE/{split}", fn=hf_dataset_to_jsonl)
            def json_step(ctx: StepContext, ice_raw=ice_raw, split=split):
                return DatasetConversionConfig(
                    dataset_name="WillHeld/ICE_Cleaned",
                    subsets=["*"],
                    splits=[split],
                    input_path=ctx.require(ice_raw),
                    hf_path="WillHeld/ICE_Cleaned",
                    output_path=ctx.output,
                    output_format=OutputFormatOptions("decontamination"),
                    prompt_key="text",
                )

            tokenized_json = default_tokenize(f"tokenized/ICE-{split}", json_step(), tokenizer, is_validation=True)
            tokenized_domains[f"ICE/{split}"] = tokenized_json

    @step(name="raw/WillHeld/paloma_subreddits", fn=download_hf)
    def subreddits_raw_step(ctx: StepContext):
        return DownloadConfig(
            hf_dataset_id="WillHeld/paloma_subreddits",
            revision=versioned("9561a2b"),
            gcs_output_path=ctx.output,
            wait_for_completion=True,
            hf_urls_glob=["**/*.parquet", "README.md"],
        )

    subreddits_raw = subreddits_raw_step().with_output_path("raw/WillHeld/paloma_subreddits").cd("9561a2b/huggingface.co/datasets/WillHeld/paloma_subreddits/resolve/9561a2b")

    for subset in get_dataset_config_names("WillHeld/paloma_subreddits"):
        @step(name=f"hf_dataset_to_jsonl/paloma_subreddits/{subset}", fn=hf_dataset_to_jsonl)
        def json_step(ctx: StepContext, subreddits_raw=subreddits_raw, subset=subset):
            return DatasetConversionConfig(
                dataset_name="WillHeld/paloma_subreddits",
                subsets=[subset],
                splits=["train"],
                input_path=ctx.require(subreddits_raw),
                hf_path="WillHeld/paloma_subreddits",
                output_path=ctx.output,
                output_format=OutputFormatOptions("decontamination"),
                prompt_key="text",
            )

        tokenized_json = default_tokenize(f"tokenized/paloma_subreddits-{subset}", json_step(), tokenizer, is_validation=True)
        tokenized_domains[f"paloma_subreddits/{subset}"] = tokenized_json

    @step(name="raw/WillHeld/paloma_programming_languages", fn=download_hf)
    def pls_raw_step(ctx: StepContext):
        return DownloadConfig(
            hf_dataset_id="WillHeld/paloma_programming_languages",
            revision=versioned("6c08b5f"),
            gcs_output_path=ctx.output,
            wait_for_completion=True,
            hf_urls_glob=["**/*.parquet", "README.md"],
        )

    pls_raw = pls_raw_step().with_output_path("raw/WillHeld/paloma_programming_languages").cd("6c08b5f/huggingface.co/datasets/WillHeld/paloma_programming_languages/resolve/6c08b5f")

    for subset in get_dataset_config_names("WillHeld/paloma_programming_languages"):
        @step(name=f"hf_dataset_to_jsonl/paloma_programming_languages/{subset}", fn=hf_dataset_to_jsonl)
        def json_step(ctx: StepContext, pls_raw=pls_raw, subset=subset):
            return DatasetConversionConfig(
                dataset_name="WillHeld/paloma_programming_languages",
                subsets=[subset],
                splits=["train"],
                input_path=ctx.require(pls_raw),
                hf_path="WillHeld/paloma_programming_languages",
                output_path=ctx.output,
                output_format=OutputFormatOptions("decontamination"),
                prompt_key="text",
            )

        tokenized_json = default_tokenize(
            f"tokenized/paloma_programming_languages-{subset}", json_step(), tokenizer, is_validation=True
        )
        tokenized_domains[f"paloma_programming_languages/{subset}"] = tokenized_json

    eval_sets = {**eval_sets, **tokenized_domains}
    evaluation_mixture = mixture_for_evaluation(eval_sets)

    return evaluation_mixture


def roughly_equals(real, target):
    return real == target or real == target + 1 or real == target - 1


model_revision_pairs = []

models = get_gemstone_model_ids()
for model in models:
    for revision in get_all_revisions(model):
        if revision != "main":  # Skip Main since it changes, so isn't a well-behaved version
            model_revision_pairs.append((model, revision))

gemstone_splits = {"main": {}, "cooldown": {}, "lr_ablation": {}}
gemstone_tokenizer = "tomg-group-umd/Gemstone-1280x15"

evaluation_mixture = distributional_eval_sets(gemstone_tokenizer)

for model, revision in model_revision_pairs:
    config = GemstoneConfig.from_model_revision(model, revision)

    gemstone_model = download_model_step(ModelConfig(hf_repo_id=model, hf_revision=revision))
    gemstone_splits[config.variant][config] = gemstone_model


if __name__ == "__main__":
    for config in gemstone_splits["cooldown"]:
        if roughly_equals(config.step, int(config.cooldown_start_step + (0.1 * config.cooldown_start_step))):
            try:
                model = config.model_id
                revision = config.revision
                model_config = HFCheckpointConverter.from_hf(f"{model}@{revision}").config_from_hf_checkpoint(
                    f"{model}@{revision}"
                )
                gemstone_model = gemstone_splits["cooldown"][config]

                eval_step = default_lm_log_probs(
                    gemstone_model,
                    model_config,
                    evaluation_mixture,
                    checkpoint_is_hf=True,
                    resource_config=ResourceConfig.with_tpu("v5p-8"),
                    name=versioned(f"Domain-Scaling-Laws-{model}@{revision}"),
                    wandb_tags=[
                        f"M={model_config.model_type}",
                        "eval=domain-scaling-laws",
                    ],
                )
                executor_main(
                    [eval_step],
                    description="Compute logprobs for all Gemstone Model Checkpoints",
                )
            except ValueError as e:
                print(f"Skipping {model}/{revision}: {e}")
    baselines = [
        ("allenai/OLMo-2-1124-7B", "stage1-step928646-tokens3896B"),
        ("allenai/OLMo-2-1124-13B", "stage1-step596000-tokens5000B"),
    ]
    for model, revision in baselines:
        local_evaluation_mixture = distributional_eval_sets(model)
        model_instance = download_model_step(ModelConfig(hf_repo_id=model, hf_revision=revision))
        model_config = HFCheckpointConverter.from_hf(f"{model}@{revision}").config_from_hf_checkpoint(
            f"{model}@{revision}"
        )
        eval_step = default_lm_log_probs(
            model_instance,
            model_config,
            local_evaluation_mixture,
            checkpoint_is_hf=True,
            name=versioned(f"Domain-Scaling-Laws-tokenizer-fix-{model}@{revision}"),
        )
        executor_main(
            [eval_step],
            description="Compute logprobs for all Baseline Model Checkpoints",
        )
