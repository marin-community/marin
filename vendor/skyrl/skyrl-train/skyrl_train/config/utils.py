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


if __name__ == "__main__":
    cfg = get_default_config()
