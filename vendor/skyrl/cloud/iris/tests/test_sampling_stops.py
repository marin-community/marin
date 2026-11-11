"""Explicit EOS and special-token settings survive recipe composition and vLLM translation."""

from pathlib import Path

import yaml

from cloud.iris.launch_config import LaunchTopology
from cloud.iris.rl_config_translation import compose_skyrl_config, parse_rl_config
from skyrl_train.inference_engines.utils import get_vllm_sampling_params


def test_multi_eos_and_special_tokens_reach_vllm(tmp_path):
    recipe = Path(__file__).resolve().parents[1] / "configs/qwen_megatron_smoke.yaml"
    raw = yaml.safe_load(recipe.read_text())
    for field in ("sampling_params", "eval_sampling_params"):
        raw["generator"].setdefault(field, {}).update(
            stop_token_ids=[128001, 128009], skip_special_tokens=False
        )
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = compose_skyrl_config(
        parse_rl_config(str(path)), {}, LaunchTopology(num_nodes=1, gpus_per_node=8, gpu_variant="H100")
    ).config
    for params in (config.generator.sampling_params, config.generator.eval_sampling_params):
        actual = get_vllm_sampling_params(params)
        assert actual["stop_token_ids"] == [128001, 128009]
        assert actual["skip_special_tokens"] is False
