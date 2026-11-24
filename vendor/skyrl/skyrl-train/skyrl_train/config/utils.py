from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

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


def model_expert_count(config: object) -> int | None:
    """Return a model's declared expert count, including its text configuration."""
    for candidate in (config, getattr(config, "text_config", None)):
        if candidate is not None:
            for name in ("num_experts", "num_local_experts", "n_routed_experts"):
                value = getattr(candidate, name, None)
                if isinstance(value, int) and value > 0:
                    return value
    return None


if __name__ == "__main__":
    cfg = get_default_config()
