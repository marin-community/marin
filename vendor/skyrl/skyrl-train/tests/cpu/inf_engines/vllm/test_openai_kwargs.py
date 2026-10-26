import pytest
from omegaconf import OmegaConf
from skyrl_train.config.weight_sync_pause import resolve_weight_sync_pause_policy


@pytest.mark.parametrize(
    "mode,options,accepted",
    [
        ("abort", {"enable_auto_tool_choice": True}, False),
        ("abort", {"tool_call_parser": "hermes"}, False),
        ("abort", {"reasoning_parser": "deepseek_r1"}, False),
        ("abort", {"enable_auto_tools": False, "enable_auto_tool_choice": True}, True),
        ("keep", {"enable_auto_tools": True, "tool_parser": "hermes", "reasoning_parser": "deepseek_r1"}, True),
    ],
)
def test_abort_requires_unparsed_generation(mode, options, accepted):
    config = OmegaConf.create(
        {
            "backend": "vllm",
            "run_engines_locally": True,
            "vllm_v1_disable_multiproc": False,
            "weight_sync_pause": {"mode": mode, "clear_cache": True},
            "engine_init_kwargs": options,
        }
    )
    if accepted:
        assert resolve_weight_sync_pause_policy(config).mode.value == mode
    else:
        with pytest.raises(ValueError, match="abort.*pars"):
            resolve_weight_sync_pause_policy(config)
    assert OmegaConf.to_container(config.engine_init_kwargs) == options
