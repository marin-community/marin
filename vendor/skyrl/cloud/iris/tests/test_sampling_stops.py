"""Recipe stop IDs and special-token settings reach backend requests."""

from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from cloud.iris.launch_config import LaunchTopology
from cloud.iris.rl_config_translation import compose_skyrl_config, parse_rl_config
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend


RECIPE = Path(__file__).resolve().parents[1] / "configs/qwen_megatron_smoke.yaml"
TOPOLOGY = LaunchTopology(num_nodes=1, gpus_per_node=8, gpu_variant="H100")


@pytest.mark.parametrize("field", ["sampling_params", "eval_sampling_params"])
@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("skip_special_tokens", [False, True])
def test_recipe_explicit_stops_reach_backend(tmp_path, field, backend, skip_special_tokens):
    raw = yaml.safe_load(RECIPE.read_text())
    raw["generator"].setdefault(field, {}).update(
        stop_token_ids=[128001, 128009], skip_special_tokens=skip_special_tokens
    )
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(raw))

    config = compose_skyrl_config(parse_rl_config(str(path)), {}, TOPOLOGY).config
    actual = get_sampling_params_for_backend(backend, config.generator[field])

    assert actual["stop_token_ids"] == [128001, 128009]
    assert isinstance(actual["stop_token_ids"], list)
    assert actual["skip_special_tokens"] is skip_special_tokens


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("field", ["sampling_params", "eval_sampling_params"])
def test_recipe_omitted_stops_keep_backend_defaults(backend, field):
    config = compose_skyrl_config(parse_rl_config(str(RECIPE)), {}, TOPOLOGY).config
    actual = get_sampling_params_for_backend(backend, config.generator[field])

    assert actual["skip_special_tokens"] is True
    assert actual.get("stop_token_ids") is None
    assert actual["stop"] is None


@pytest.mark.parametrize("skip_special_tokens", [None, False, True])
@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_backend_special_tokens_with_legacy_sampling_config(skip_special_tokens, backend):
    config = OmegaConf.create(
        dict(max_generate_length=8, temperature=0.0, top_p=1.0, top_k=-1, min_p=0.0, logprobs=None)
    )
    if skip_special_tokens is not None:
        config.skip_special_tokens = skip_special_tokens

    actual = get_sampling_params_for_backend(backend, config)

    assert actual["skip_special_tokens"] is (True if skip_special_tokens is None else skip_special_tokens)
    assert actual.get("stop_token_ids") is None
    assert actual.get("stop") is None
