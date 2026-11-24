from omegaconf import DictConfig, OmegaConf
from skyrlrecipe import SkyRLRecipe

from skyrl_train.config.utils import get_default_config


def compose_recipe(recipe: SkyRLRecipe) -> DictConfig:
    """Compose a local recipe without entrypoint, config_groups, extra_env or model_num_attention_heads."""
    cfg = get_default_config()
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(
        cfg,
        recipe.model_dump(
            mode="json",
            exclude_unset=True,
            exclude={"context_budget", "entrypoint", "config_groups", "extra_env", "model_num_attention_heads"},
        ),
    )
    for path, value in recipe.context_budget.derived_values(None).items():
        if (
            path == "generator.engine_init_kwargs.max_model_len"
            and "serving_window_tokens" not in recipe.context_budget.model_fields_set
        ):
            continue
        OmegaConf.update(cfg, path, value, force_add=True)
    return cfg
