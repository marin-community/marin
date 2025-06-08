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

import re
from dataclasses import dataclass

from huggingface_hub import get_collection, list_repo_refs
from levanter.compat.hf_checkpoints import HFCheckpointConverter

from experiments.defaults import default_validation_sets
from experiments.models import ModelConfig, download_model_step
from marin.evaluation.log_probs import default_lm_log_probs
from marin.execution.executor import executor_main, output_path_of, versioned
from marin.processing.tokenize.data_configs import mixture_for_evaluation


@dataclass(frozen=True)
class GemstoneConfig:
    """Hashable configuration for Gemstone models."""

    width: int
    depth: int
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

        return cls(width=width, depth=depth, variant=variant, step=step, cooldown_start_step=cooldown_start_step)

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
gemstone_tokenizer = "tomg-group-umd/Gemstone-1280x15"
eval_sets = default_validation_sets(tokenizer=versioned(gemstone_tokenizer))
evaluation_mixture = mixture_for_evaluation(eval_sets)
eval_steps = []
for model, revision in model_revision_pairs:
    try:
        config = GemstoneConfig.from_model_revision(model, revision)

        gemstone_model = download_model_step(ModelConfig(hf_repo_id=model, hf_revision=revision))
        gemstone_splits[config.variant][config] = gemstone_model
        if config.variant == "cooldown" and roughly_equals(
            config.step, int(config.cooldown_start_step + (0.1 * config.cooldown_start_step))
        ):
            model_config = HFCheckpointConverter.from_hf(f"{model}@{revision}").config_from_hf_checkpoint(
                f"{model}@{revision}"
            )

            eval_steps.append(
                default_lm_log_probs(
                    output_path_of(gemstone_model), model_config, evaluation_mixture, checkpoint_is_hf=True
                )
            )
    except ValueError as e:
        print(f"Skipping {model}/{revision}: {e}")

if __name__ == "__main__":
    executor_main(
        eval_steps,
        description="Compute logprobs for all Gemstone Model Checkpoints",
    )
