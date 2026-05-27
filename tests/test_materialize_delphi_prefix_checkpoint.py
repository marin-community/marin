# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
from levanter.checkpoint import CheckpointerConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.adamh import AdamHConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.executor import MirroredValue
from rigging.filesystem import collect_gcs_paths

from experiments.delphi_models import get_delphi_model
from scripts.materialize_delphi_prefix_checkpoint import (
    SOURCE_CHECKPOINT_MIRROR_BUDGET_GB,
    MaterializeRequest,
    build_executor_step,
    build_materialization_plan,
    decode_train_config_from_executor_info,
    levanter_stop_step_for_checkpoint_step,
    print_plan,
    regionalize_readonly_marin_uris,
)


def source_train_config(num_train_steps: int = 37_001) -> TrainLmConfig:
    return TrainLmConfig(
        trainer=TrainerConfig(
            num_train_steps=num_train_steps,
            checkpointer=CheckpointerConfig(),
            tracker=WandbConfig(tags=["source"], replicate_path="gs://marin-us-central2/original"),
        ),
        optimizer=AdamHConfig(
            learning_rate=0.0027599905274620106,
            adam_lr=0.00033154735825338737,
            beta2=0.9999,
            epsilon=3.6604122149949323e-08,
            warmup=0.1,
            decay=0.2,
            lr_schedule="linear",
            min_lr_ratio=0.0,
        ),
    )


def request(
    *,
    source_step: int = 20_000,
    target_step: int = 25_900,
    output_root: str = "gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-3e18-step25900",
) -> MaterializeRequest:
    return MaterializeRequest(
        base="3e18",
        source_step=source_step,
        target_step=target_step,
        output_root=output_root,
        tpu="v5p-8",
        ram="128g",
        regions=("us-east5",),
    )


def test_decode_delphi_executor_info_defaults_untyped_model_to_qwen3():
    executor_info = {
        "config": {
            "train_config": {
                "model": {
                    "max_seq_len": 4096,
                    "hidden_dim": 1024,
                    "intermediate_dim": 4096,
                    "num_layers": 11,
                    "num_heads": 8,
                    "head_dim": None,
                    "num_kv_heads": 8,
                    "activation_function": "silu",
                    "initializer_range": 0.02,
                    "layer_norm_epsilon": 1e-5,
                    "tie_word_embeddings": False,
                    "hybrid_norm": False,
                    "use_qk_norm": False,
                    "input_embedding_norm": False,
                    "upcast_attn": False,
                    "attn_backend": None,
                    "flash_attention_block_size": None,
                    "gradient_checkpointing": True,
                    "scan_layers": True,
                    "use_bias": False,
                    "use_layer_norm_weight": True,
                    "rope": {
                        "theta": 500000,
                        "factor": 8.0,
                        "low_freq_factor": 1.0,
                        "high_freq_factor": 4.0,
                        "original_max_position_embeddings": 8192,
                    },
                    "reference_checkpoint": "NousResearch/Llama-2-7b-hf",
                    "tokenizer": None,
                    "use_sliding_window": False,
                    "sliding_window": 4096,
                },
                "optimizer": {
                    "learning_rate": 0.0027599905274620106,
                    "adam_lr": 0.00033154735825338737,
                    "beta2": 0.9999,
                    "epsilon": 3.6604122149949323e-08,
                    "warmup": 0.1,
                    "decay": 0.2,
                    "lr_schedule": "linear",
                    "min_lr_ratio": 0.0,
                },
                "trainer": {"num_train_steps": 37335},
            }
        }
    }

    train_config = decode_train_config_from_executor_info(
        executor_info,
        source_path="gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+18/.executor_info",
    )

    assert isinstance(train_config.model, Qwen3Config)
    assert train_config.model.hidden_dim == 1024


def test_rejects_output_under_original_delphi_root():
    model = get_delphi_model("3e18")
    with pytest.raises(ValueError, match="original Delphi root"):
        build_materialization_plan(
            request(output_root=f"{model.gcs_run_root}/materialized-step25900"),
            model,
            source_train_config(),
        )


def test_rejects_target_before_source():
    model = get_delphi_model("3e18")
    with pytest.raises(ValueError, match="target_step must be greater than source_step"):
        build_materialization_plan(
            request(source_step=20_000, target_step=19_999),
            model,
            source_train_config(),
        )


def test_materialized_config_preserves_original_schedule_length():
    model = get_delphi_model("3e18")
    with (
        patch("rigging.filesystem.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict("os.environ", {"MARIN_PREFIX": "gs://marin-us-east5/scratch"}),
    ):
        plan = build_materialization_plan(request(), model, source_train_config())

    assert plan.train_config.trainer.num_train_steps == 37_001
    assert plan.original_num_train_steps == 37_001


def test_stop_target_is_separate_from_lr_schedule_length():
    model = get_delphi_model("3e18")
    req = request()
    with (
        patch("rigging.filesystem.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict("os.environ", {"MARIN_PREFIX": "gs://marin-us-east5/scratch"}),
    ):
        plan = build_materialization_plan(req, model, source_train_config())

    trainer = plan.train_config.trainer
    assert trainer.num_train_steps == 37_001
    assert trainer.stop_step == req.target_step + 1
    assert levanter_stop_step_for_checkpoint_step(req.target_step) == req.target_step + 1
    assert isinstance(trainer.initialize_from, MirroredValue)
    assert trainer.initialize_from.value == (
        "checkpoints/isoflop/isoflop-3e+18-d1024-L11-B8-adamh_scaling_v6/checkpoints/step-20000"
    )
    assert trainer.initialize_from.budget_gb == SOURCE_CHECKPOINT_MIRROR_BUDGET_GB
    assert trainer.load_checkpoint is False

    checkpointer = trainer.checkpointer
    assert checkpointer.base_path == f"{req.output_root}/checkpoints"
    assert checkpointer.append_run_id_to_base_path is False
    assert checkpointer.keep == []
    assert checkpointer.metadata["target_step"] == req.target_step
    assert checkpointer.metadata["source_checkpoint_path"].startswith("mirror://")
    collected_gcs_paths = [path for _, path in collect_gcs_paths(plan.train_config)]
    assert not any("marin-us-central2" in path for path in collected_gcs_paths)
    step = build_executor_step(plan)
    assert step.config.resources.regions == ("us-east5",)

    tracker = trainer.tracker
    assert isinstance(tracker, WandbConfig)
    assert "delphi_prefix_checkpoint" in tracker.tags
    assert f"target_step:{req.target_step}" in tracker.tags
    assert tracker.replicate_path == req.output_root


def test_regionalizes_tensorstore_data_paths_without_mirror_scheme():
    rendered = regionalize_readonly_marin_uris(
        {
            "cache_dir": "gs://marin-us-central2/tokenized/nemotron_cc/hq_actual-5af4cc/train",
            "external": "gs://not-marin-bucket/path",
        },
        target_prefix="gs://marin-us-east5",
    )

    assert rendered["cache_dir"] == "gs://marin-us-east5/tokenized/nemotron_cc/hq_actual-5af4cc/train"
    assert not rendered["cache_dir"].startswith("mirror://")
    assert rendered["external"] == "gs://not-marin-bucket/path"


def test_dry_run_plan_prints_operator_safety_summary(capsys):
    model = get_delphi_model("3e18")
    with (
        patch("rigging.filesystem.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict("os.environ", {"MARIN_PREFIX": "gs://marin-us-east5/scratch"}),
    ):
        plan = build_materialization_plan(request(), model, source_train_config())

    print_plan(plan)
    out = capsys.readouterr().out

    assert "original schedule length: 37,001 steps" in out
    assert "source step:" in out
    assert "target step:" in out
    assert "steps to train:           5,900" in out
    assert f"destination checkpoint:   {plan.destination_checkpoint_path}" in out
    assert "training load path:       mirror://checkpoints/isoflop/" in out
    assert "source mirror budget:     40 GB" in out
    assert "regions:                  us-east5" in out
    assert "original Delphi root will not be modified" in out


# ---------------------------------------------------------------------------
# --also-save-step / extra_target_steps
# ---------------------------------------------------------------------------


def _multi_target_request(
    *,
    source_step: int = 20_000,
    extras: tuple[int, ...] = (23_000,),
    target_step: int = 25_900,
) -> MaterializeRequest:
    return MaterializeRequest(
        base="3e18",
        source_step=source_step,
        target_step=target_step,
        output_root="gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-3e18-prefixes-qwen3",
        tpu="v5p-8",
        ram="128g",
        regions=("us-east5",),
        extra_target_steps=extras,
    )


def test_rejects_extra_target_step_at_or_before_source():
    model = get_delphi_model("3e18")
    with pytest.raises(ValueError, match="must be > source_step"):
        build_materialization_plan(
            _multi_target_request(extras=(20_000,)),
            model,
            source_train_config(),
        )
    with pytest.raises(ValueError, match="must be > source_step"):
        build_materialization_plan(
            _multi_target_request(extras=(19_999,)),
            model,
            source_train_config(),
        )


def test_rejects_extra_target_step_at_or_after_target():
    model = get_delphi_model("3e18")
    with pytest.raises(ValueError, match="must be < target_step"):
        build_materialization_plan(
            _multi_target_request(extras=(25_900,), target_step=25_900),
            model,
            source_train_config(),
        )
    with pytest.raises(ValueError, match="must be < target_step"):
        build_materialization_plan(
            _multi_target_request(extras=(26_000,), target_step=25_900),
            model,
            source_train_config(),
        )


def test_rejects_duplicate_extra_target_steps():
    model = get_delphi_model("3e18")
    with pytest.raises(ValueError, match="must be distinct"):
        build_materialization_plan(
            _multi_target_request(extras=(22_000, 22_000)),
            model,
            source_train_config(),
        )


def test_multi_target_keep_policies_are_sorted_and_well_formed():
    # Levanter's Checkpointer constructor asserts step_policies sorted
    # ascending by ``until`` (lib/levanter/src/levanter/checkpoint.py:436).
    # The helper must emit ``keep`` in ascending order regardless of the
    # input ordering of ``extra_target_steps``.
    model = get_delphi_model("3e18")
    with (
        patch("rigging.filesystem.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict("os.environ", {"MARIN_PREFIX": "gs://marin-us-east5/scratch"}),
    ):
        plan = build_materialization_plan(
            _multi_target_request(extras=(24_000, 22_000, 23_500)),
            model,
            source_train_config(),
        )
    keep = plan.train_config.trainer.checkpointer.keep
    # One policy per extra step, sorted ascending, every == until == step.
    # The final target_step is NOT in keep — it's covered by Levanter's
    # forced save at training end (trainer.py:568,581).
    assert keep == [
        {"every": 22_000, "until": 22_000},
        {"every": 23_500, "until": 23_500},
        {"every": 24_000, "until": 24_000},
    ]


def test_multi_target_destinations_cover_every_save_point():
    model = get_delphi_model("3e18")
    with (
        patch("rigging.filesystem.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict("os.environ", {"MARIN_PREFIX": "gs://marin-us-east5/scratch"}),
    ):
        plan = build_materialization_plan(
            _multi_target_request(extras=(23_000, 24_500), target_step=25_900),
            model,
            source_train_config(),
        )
    dests = plan.destination_checkpoint_paths
    # Sorted ascending; final is the target_step.
    assert len(dests) == 3
    assert dests[0].endswith("/checkpoints/step-23000")
    assert dests[1].endswith("/checkpoints/step-24500")
    assert dests[2].endswith("/checkpoints/step-25900")
    assert plan.destination_checkpoint_path == dests[-1]  # back-compat shortcut
    assert plan.all_target_steps == (23_000, 24_500, 25_900)


def test_multi_target_metadata_lists_all_target_steps():
    # Each save's metadata.json carries ``all_target_steps`` so an
    # intermediate ckpt's metadata can be distinguished from "only save".
    # The legacy ``target_step`` field still equals the FINAL target.
    model = get_delphi_model("3e18")
    with (
        patch("rigging.filesystem.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict("os.environ", {"MARIN_PREFIX": "gs://marin-us-east5/scratch"}),
    ):
        plan = build_materialization_plan(
            _multi_target_request(extras=(23_000, 24_500), target_step=25_900),
            model,
            source_train_config(),
        )
    meta = plan.train_config.trainer.checkpointer.metadata
    assert meta["target_step"] == 25_900  # legacy: the final
    assert meta["all_target_steps"] == [23_000, 24_500, 25_900]  # new: every save
    assert meta["materialization"] == "delphi_prefix_checkpoint"


def test_single_target_keeps_empty_keep_policy():
    # Backward compatibility: when no extras are requested, ``keep`` must
    # stay empty (the original behavior pre-extension). Only the forced
    # final save fires.
    model = get_delphi_model("3e18")
    with (
        patch("rigging.filesystem.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict("os.environ", {"MARIN_PREFIX": "gs://marin-us-east5/scratch"}),
    ):
        plan = build_materialization_plan(request(), model, source_train_config())
    assert plan.train_config.trainer.checkpointer.keep == []
    assert plan.destination_checkpoint_paths == (plan.destination_checkpoint_path,)
    assert plan.all_target_steps == (25_900,)
    assert plan.train_config.trainer.checkpointer.metadata["all_target_steps"] == [25_900]


def test_multi_target_dry_run_plan_lists_every_destination(capsys):
    model = get_delphi_model("3e18")
    with (
        patch("rigging.filesystem.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict("os.environ", {"MARIN_PREFIX": "gs://marin-us-east5/scratch"}),
    ):
        plan = build_materialization_plan(
            _multi_target_request(extras=(23_000, 24_500), target_step=25_900),
            model,
            source_train_config(),
        )
    print_plan(plan)
    out = capsys.readouterr().out
    assert "extra target steps:" in out
    assert "23,000" in out and "24,500" in out
    assert "destination checkpoints:" in out
    for dest in plan.destination_checkpoint_paths:
        assert dest in out
