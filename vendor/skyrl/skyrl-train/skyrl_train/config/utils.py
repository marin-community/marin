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


def sequences_per_step(config: DictConfig) -> int:
    """Return the training rows one optimizer step consumes, counting rollouts per prompt.

    Rollout training draws ``train_batch_size`` prompt groups of ``n_samples_per_prompt``
    rollouts each; fixed-data training sets ``n_samples_per_prompt=1`` so one row is one
    sequence. Matching methods set equal products (control C2).
    """
    return int(config.trainer.train_batch_size) * int(config.generator.n_samples_per_prompt)


def response_token_cap(config: DictConfig) -> int:
    """Return the longest response any training method may consume, in tokens.

    Rollout training truncates generation at this cap; fixed-data rows must fit it.
    """
    return int(config.generator.sampling_params.max_generate_length)


def sequence_token_cap(config: DictConfig) -> int:
    """Return the configured prompt-plus-response sequence budget, in tokens.

    The matched budget is the configured one, not the engine's context limit: fixed-data
    runs launch no engine, so ``engine_init_kwargs.max_model_len`` must not enter the match.
    """
    return int(config.generator.max_input_length) + response_token_cap(config)


def get_default_config():
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=DEFAULT_CONFIG_NAME)
    return cfg


if __name__ == "__main__":
    cfg = get_default_config()
