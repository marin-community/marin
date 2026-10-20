"""Behavior tests for the managed speculative-decoding configuration."""

from pathlib import Path
import sys
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
for source_root in (_REPO_ROOT, _REPO_ROOT / "skyrl-train"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from cloud.iris.rl_config_translation import compose_skyrl_config, parse_rl_config  # noqa: E402
from marinskyrl.speculative_decoding import (  # noqa: E402
    SpeculativeDecodingConfigError,
    parse_speculative_decoding_config,
)
from skyrl_train.entrypoints.main_base import config_dir  # noqa: E402


_DRAFT_REVISION = "4bdb47c08e5b5190bea3c7a93c3e14470230e469"


def _base_config() -> dict:
    return {
        "entrypoint": "standard",
        "context_budget": {
            "request_window_tokens": 2048,
            "max_new_tokens_per_turn": 512,
            "max_turns": 1,
        },
        "trainer": {"placement": {"colocate_all": False}},
        "generator": {
            "backend": "vllm",
            "run_engines_locally": True,
            "inference_engine_tensor_parallel_size": 1,
            "speculative_decoding": {
                "method": "eagle3",
                "model": {
                    "source_uri": "hf://laion/snowball-64k-eagle3-draft-r2egym",
                    "source_identity": _DRAFT_REVISION,
                },
                "num_speculative_tokens": 3,
                "training": {
                    "interval_steps": 1,
                    "reserved_gpu_memory_gib": 8,
                },
            },
        },
    }


def _write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "speculative.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_managed_speculator_reaches_hydra_with_immutable_source_unchanged(tmp_path: Path) -> None:
    parsed = parse_rl_config(str(_write_config(tmp_path, _base_config())))
    speculator = parsed.generator["speculative_decoding"]

    assert speculator["model"]["source_uri"] == "hf://laion/snowball-64k-eagle3-draft-r2egym"
    assert speculator["model"]["source_identity"] == _DRAFT_REVISION

    cfg = compose_skyrl_config(parsed, {"num_nodes": 2}, SimpleNamespace(gpus_per_node=8)).config

    resolved = parse_speculative_decoding_config(
        OmegaConf.to_container(cfg.generator.speculative_decoding, resolve=True),
        backend=cfg.generator.backend,
        run_engines_locally=cfg.generator.run_engines_locally,
        entrypoint="skyrl_train.entrypoints.main_base",
        colocate_all=cfg.trainer.placement.colocate_all,
    )
    assert resolved is not None
    assert resolved.vllm_speculative_config() == {
        "method": "eagle3",
        "model": "laion/snowball-64k-eagle3-draft-r2egym",
        "num_speculative_tokens": 3,
        "revision": _DRAFT_REVISION,
    }
    assert resolved.training is not None
    assert resolved.training.interval_steps == 1
    assert resolved.training.reserved_gpu_memory_gib == 8


def test_local_speculator_uses_model_without_hub_resolution() -> None:
    value = _base_config()["generator"]["speculative_decoding"]
    value["model"]["source_uri"] = "/tmp/draft-model"

    resolved = parse_speculative_decoding_config(
        value,
        backend="vllm",
        run_engines_locally=True,
        entrypoint="skyrl_train.entrypoints.main_base",
        colocate_all=False,
    )

    assert resolved is not None
    assert resolved.vllm_speculative_config() == {
        "method": "eagle3",
        "model": "/tmp/draft-model",
        "num_speculative_tokens": 3,
    }


def test_null_speculator_keeps_the_default_disabled() -> None:
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config")

    assert cfg.generator.speculative_decoding is None
    assert (
        parse_speculative_decoding_config(
            None,
            backend=cfg.generator.backend,
            run_engines_locally=cfg.generator.run_engines_locally,
            entrypoint="skyrl_train.entrypoints.main_base",
            colocate_all=cfg.trainer.placement.colocate_all,
        )
        is None
    )


def test_frozen_object_store_speculator_is_supported(tmp_path: Path) -> None:
    config = _base_config()
    model = config["generator"]["speculative_decoding"]["model"]
    model["source_uri"] = "s3://models/snowball/eagle3"
    model["source_identity"] = "snowball-eagle3@step-1888"
    config["generator"]["speculative_decoding"]["training"] = None

    parsed = parse_rl_config(str(_write_config(tmp_path, config)))

    assert parsed.generator["speculative_decoding"]["training"] is None
    assert parsed.generator["speculative_decoding"]["model"] == model

    resolved = parse_speculative_decoding_config(
        parsed.generator["speculative_decoding"],
        backend="vllm",
        run_engines_locally=True,
        entrypoint="skyrl_train.entrypoints.main_base",
        colocate_all=False,
    )
    assert resolved is not None
    assert resolved.vllm_speculative_config() == {
        "method": "eagle3",
        "model": "s3://models/snowball/eagle3",
        "num_speculative_tokens": 3,
        "draft_load_config": {"load_format": "runai_streamer"},
    }


def test_gcs_alias_is_normalized_for_vllm_runai_loading(tmp_path: Path) -> None:
    config = _base_config()
    model = config["generator"]["speculative_decoding"]["model"]
    model["source_uri"] = "gcs://models/snowball/eagle3"
    model["source_identity"] = "snowball-eagle3@step-1888"

    parsed = parse_rl_config(str(_write_config(tmp_path, config)))
    resolved = parse_speculative_decoding_config(
        parsed.generator["speculative_decoding"],
        backend="vllm",
        run_engines_locally=True,
        entrypoint="skyrl_train.entrypoints.main_base",
        colocate_all=False,
    )

    assert resolved is not None
    assert resolved.vllm_speculative_config()["model"] == "gs://models/snowball/eagle3"


def test_speculator_rejects_a_mutable_hub_revision(tmp_path: Path) -> None:
    """A branch name such as ``main`` can move, so the run would not be replayable."""
    config = _base_config()
    config["generator"]["speculative_decoding"]["model"]["source_identity"] = "main"

    with pytest.raises(SpeculativeDecodingConfigError, match="full 40-character"):
        parse_rl_config(str(_write_config(tmp_path, config)))
