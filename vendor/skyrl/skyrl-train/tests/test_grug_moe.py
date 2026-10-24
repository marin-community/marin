import json
from pathlib import Path

import pytest
from transformers import AutoConfig

import skyrl_train.models  # noqa: F401  registers grug_moe with AutoConfig

LEVANTER_CONFIG = Path(__file__).parent / "fixtures/grug_training_oracle/config.json"


def load(directory: Path, **changes) -> AutoConfig:
    config = {**json.loads(LEVANTER_CONFIG.read_text()), **changes}
    (directory / "config.json").write_text(json.dumps({k: v for k, v in config.items() if v is not None}))
    return AutoConfig.from_pretrained(directory)


@pytest.mark.parametrize("omitted", ["qk_mult", "sliding_window", "max_position_embeddings", "num_experts_per_tok"])
def test_grug_checkpoint_without_a_sampler_defaulted_key_fails_to_load(tmp_path, omitted):
    with pytest.raises((TypeError, ValueError), match=f"(missing|must state).*{omitted}"):
        load(tmp_path, **{omitted: None})


@pytest.mark.parametrize(
    "theta",
    [{"rope_parameters": {"rope_type": "default", "rope_theta": 500000.0}}, {"rope": {"theta": 500000.0}}],
    ids=["rope_parameters", "rope"],
)
def test_grug_checkpoint_rope_theta_aliases_resolve_to_the_stored_theta(tmp_path, theta):
    loaded = load(tmp_path, rope_theta=None, **theta)
    assert loaded.rope_theta == 500000.0
    loaded.save_pretrained(tmp_path / "exported")
    assert AutoConfig.from_pretrained(tmp_path / "exported").rope_theta == 500000.0
