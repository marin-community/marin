from pathlib import Path

import pytest
import yaml
from skyrlrecipe import SkyRLRecipe

from cloud.iris.launch_config import LaunchTopology
from cloud.iris.rl_config_translation import compose_skyrl_config, parse_rl_config
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend


@pytest.mark.parametrize("field", ["sampling_params", "eval_sampling_params"])
def test_recipe_explicit_stops_reach_vllm(tmp_path, field):
    source = Path(__file__).resolve().parents[1] / "configs/qwen_megatron_smoke.yaml"
    raw = yaml.safe_load(source.read_text())
    raw["generator"].setdefault(field, {}).update(stop_token_ids=[128001, 128009], skip_special_tokens=False)
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(SkyRLRecipe.from_document(raw).to_skyrl()))

    topology = LaunchTopology(num_nodes=1, gpus_per_node=8, gpu_variant="H100")
    config = compose_skyrl_config(parse_rl_config(str(path)), {}, topology).config
    actual = get_sampling_params_for_backend("vllm", config.generator[field])

    assert actual["skip_special_tokens"] is False
    assert actual["stop_token_ids"] == [128001, 128009]
