from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from skyrlrecipe import SkyRLRecipe

CONFIG_DIR = Path(__file__).parent  # skyrl-train/config
DEFAULT_CONFIG_NAME = "ppo_base_config.yaml"


def generation_context_limit(config: DictConfig) -> int:
    """Return the sequence bound shared by generation and loss normalization."""
    configured = int(config.max_input_length) + int(config.sampling_params.max_generate_length)
    model_limit = OmegaConf.select(config, "engine_init_kwargs.max_model_len")
    return configured if model_limit is None else min(configured, int(model_limit))


def get_default_config():
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=DEFAULT_CONFIG_NAME)
    return cfg


def compose_recipe(recipe: SkyRLRecipe) -> DictConfig:
    """Compose a local training recipe with Hydra's base and its request token limits."""
    cfg = get_default_config()
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, recipe.model_dump(mode="json", exclude_unset=True, exclude={"context_budget"}))
    for path, value in recipe.context_budget.derived_values(None).items():
        if (
            path == "generator.engine_init_kwargs.max_model_len"
            and "serving_window_tokens" not in recipe.context_budget.model_fields_set
        ):
            continue
        OmegaConf.update(cfg, path, value, force_add=True)
    return cfg


if __name__ == "__main__":
    cfg = get_default_config()
