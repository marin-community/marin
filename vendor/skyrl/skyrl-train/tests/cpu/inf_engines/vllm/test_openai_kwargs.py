import pytest
from omegaconf import OmegaConf

from skyrl_train.config.weight_sync_pause import resolve_weight_sync_pause_policy
from skyrl_train.inference_engines.vllm.utils import pop_vllm_wrapper_kwargs


def test_pop_vllm_wrapper_kwargs_normalizes_aliases_and_preserves_engine_options():
    engine_kwargs = {
        "enable_auto_tools": 1,
        "tool_parser": "json",
        "other": "keep",
    }
    wrapper_kwargs = pop_vllm_wrapper_kwargs(engine_kwargs)

    assert wrapper_kwargs == {"enable_auto_tools": True, "tool_parser": "json"}
    assert engine_kwargs == {"other": "keep"}

    engine_kwargs = {"enable_auto_tool_choice": 0, "tool_call_parser": "proto"}
    wrapper_kwargs = pop_vllm_wrapper_kwargs(engine_kwargs)

    assert wrapper_kwargs == {"enable_auto_tools": False, "tool_parser": "proto"}
    assert engine_kwargs == {}


@pytest.mark.parametrize(
    "mode,options,accepted",
    [
        ("abort", {"enable_auto_tool_choice": True}, False),
        ("abort", {"tool_call_parser": "hermes"}, False),
        ("abort", {"reasoning_parser": "deepseek_r1"}, False),
        ("abort", {"enable_auto_tools": True}, False),
        ("abort", {"tool_parser": "hermes"}, False),
        ("abort", {"enable_auto_tools": True, "enable_auto_tool_choice": False}, False),
        ("abort", {}, True),
        ("abort", {"enable_auto_tools": False, "enable_auto_tool_choice": True}, True),
        ("abort", {"tool_parser": None, "tool_call_parser": "hermes"}, True),
        ("abort", {"enable_auto_tool_choice": False, "tool_call_parser": None, "reasoning_parser": None}, True),
        ("keep", {"enable_auto_tools": True, "tool_parser": "hermes", "reasoning_parser": "deepseek_r1"}, True),
        ("wait", {"enable_auto_tools": True, "tool_parser": "hermes", "reasoning_parser": "deepseek_r1"}, True),
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
