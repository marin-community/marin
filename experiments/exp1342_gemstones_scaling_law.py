"""
Gemstone Models Configuration

This file contains configurations for all 22 Gemstone models from the tomg-group-umd collection.
These models range from 50M to 2B parameters, spanning 11 widths (256 to 3072) and 18 depths (3 to 80).

Usage:
1. Import the model step you want to use
2. Run executor_main([model_step]) to download
3. Use get_model_local_path(model_step) to get the local path

Example:
```
from gemstones import gemstone_768x45
from marin.execution.executor import executor_main
from experiments.models import get_model_local_path

executor_main([gemstone_768x45])
local_path = get_model_local_path(gemstone_768x45)
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
from marin.download.huggingface.download import DownloadConfig, download
from marin.evaluation.log_probs import default_lm_log_probs
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.processing.tokenize.data_configs import mixture_for_evaluation
from marin.raw2json.huggingface.qa.raw2json import DatasetConversionConfig, OutputFormatOptions, raw2json

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
    combined = [tag for tag in (branches + tags) if "step" in tag]

    combined = sorted(combined, key=lambda x: int(x.split("step_")[-1].split("cooldown_")[-1]))
    return combined


def roughly_equals(real, target):
    return real == target or real == target + 1 or real == target - 1


model_revision_pairs = []

models = get_gemstone_model_ids()
for model in models:
    for revision in get_all_revisions(model):
        if revision != "main":  # Skip Main since it changes, so isn't a well-behaved version
            model_revision_pairs.append((model, revision))

gemstone_splits = {"main": {}, "cooldown": {}, "lr_ablation": {}}
# gemstone_tokenizer = "tomg-group-umd/Gemstone-1280x15"
gemstone_tokenizer = "allenai/OLMo-2-1124-7B"
eval_sets = default_validation_sets(tokenizer=versioned(gemstone_tokenizer))
md3_raw = ExecutorStep(
    name="raw/WillHeld/MD3",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id="WillHeld/MD3",
        revision=versioned("7c74e59"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet"],
    ),
    override_output_path="raw/WillHeld/MD3-Trunc",
).cd("7c74e59/huggingface.co/datasets/WillHeld/MD3/resolve/7c74e59")

tokenized_domains = {}
for split in get_dataset_split_names("WillHeld/MD3"):
    json = ExecutorStep(
        name=f"raw2json/md3/{split}",
        fn=raw2json,
        config=DatasetConversionConfig(
            dataset_name="WillHeld/MD3",
            subsets=["*"],
            splits=[split],
            input_path=md3_raw,
            hf_path="WillHeld/MD3",
            output_path=this_output_path(),
            output_format=OutputFormatOptions("decontamination"),
            prompt_key="transcript",
        ),
    )
    tokenized_json = default_tokenize(f"tokenized/md3-{split}", json, gemstone_tokenizer, is_validation=True)
    tokenized_domains[f"md3/{split}"] = tokenized_json

if CAN_ACCESS_ICE:
    ice_raw = ExecutorStep(
        name="raw/WillHeld/ICE",
        fn=download,
        config=DownloadConfig(
            hf_dataset_id="WillHeld/ICE_Cleaned",
            revision=versioned("4c09dd9"),
            gcs_output_path=this_output_path(),
            wait_for_completion=True,
            hf_urls_glob=["**/*.parquet"],
        ),
        override_output_path="raw/WillHeld/ICE_Cleaned",
    ).cd("4c09dd9/huggingface.co/datasets/WillHeld/ICE_Cleaned/resolve/4c09dd9")

    for split in get_dataset_split_names("WillHeld/ICE_Cleaned"):
        json = ExecutorStep(
            name=f"raw2json/ICE/{split}",
            fn=raw2json,
            config=DatasetConversionConfig(
                dataset_name="WillHeld/ICE_Cleaned",
                subsets=["*"],
                splits=[split],
                input_path=ice_raw,
                hf_path="WillHeld/ICE_Cleaned",
                output_path=this_output_path(),
                output_format=OutputFormatOptions("decontamination"),
                prompt_key="text",
            ),
        )
        tokenized_json = default_tokenize(f"tokenized/ICE-{split}", json, gemstone_tokenizer, is_validation=True)
        tokenized_domains[f"ICE/{split}"] = tokenized_json

subreddits_raw = ExecutorStep(
    name="raw/WillHeld/paloma_subreddits",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id="WillHeld/paloma_subreddits",
        revision=versioned("9561a2b"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet", "README.md"],
    ),
    override_output_path="raw/WillHeld/paloma_subreddits",
).cd("9561a2b/huggingface.co/datasets/WillHeld/paloma_subreddits/resolve/9561a2b")

for subset in get_dataset_config_names("WillHeld/paloma_subreddits"):
    json = ExecutorStep(
        name=f"raw2json/paloma_subreddits/{subset}",
        fn=raw2json,
        config=DatasetConversionConfig(
            dataset_name="WillHeld/paloma_subreddits",
            subsets=[subset],
            splits=["train"],
            input_path=subreddits_raw,
            hf_path="WillHeld/paloma_subreddits",
            output_path=this_output_path(),
            output_format=OutputFormatOptions("decontamination"),
            prompt_key="text",
        ),
    )
    tokenized_json = default_tokenize(
        f"tokenized/paloma_subreddits-{subset}", json, gemstone_tokenizer, is_validation=True
    )
    tokenized_domains[f"paloma_subreddits/{subset}"] = tokenized_json

pls_raw = ExecutorStep(
    name="raw/WillHeld/paloma_programming_languages",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id="WillHeld/paloma_programming_languages",
        revision=versioned("6c08b5f"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet", "README.md"],
    ),
    override_output_path="raw/WillHeld/paloma_programming_languages",
).cd("6c08b5f/huggingface.co/datasets/WillHeld/paloma_programming_languages/resolve/6c08b5f")

for subset in get_dataset_config_names("WillHeld/paloma_programming_languages"):
    json = ExecutorStep(
        name=f"raw2json/paloma_programming_languages/{subset}",
        fn=raw2json,
        config=DatasetConversionConfig(
            dataset_name="WillHeld/paloma_programming_languages",
            subsets=[subset],
            splits=["train"],
            input_path=pls_raw,
            hf_path="WillHeld/paloma_programming_languages",
            output_path=this_output_path(),
            output_format=OutputFormatOptions("decontamination"),
            prompt_key="text",
        ),
    )
    tokenized_json = default_tokenize(
        f"tokenized/paloma_programming_languages-{subset}", json, gemstone_tokenizer, is_validation=True
    )
    tokenized_domains[f"paloma_programming_languages/{subset}"] = tokenized_json

eval_sets = {**eval_sets, **tokenized_domains}
evaluation_mixture = mixture_for_evaluation(eval_sets)


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
                    output_path_of(gemstone_model),
                    model_config,
                    evaluation_mixture,
                    checkpoint_is_hf=True,
                    name=versioned(f"Domain-Scaling-Laws-{model}@{revision}"),
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
        model_instance = download_model_step(ModelConfig(hf_repo_id=model, hf_revision=revision))
        model_config = HFCheckpointConverter.from_hf(f"{model}@{revision}").config_from_hf_checkpoint(
            f"{model}@{revision}"
        )
        eval_step = default_lm_log_probs(
            output_path_of(model_instance),
            model_config,
            evaluation_mixture,
            checkpoint_is_hf=True,
            name=versioned(f"Domain-Scaling-Laws-tokenizer-fix-{model}@{revision}"),
        )
        executor_main(
            [eval_step],
            description="Compute logprobs for all Baseline Model Checkpoints",
        )
