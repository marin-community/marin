# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import jax
from jax._src import config as jax_config
from jax.sharding import use_abstract_mesh

from experiments.grug.moe.eval_logprob import (
    CURRENT_CHECKPOINT_LAYOUT,
    GRUG_LOGPROB_TASKS,
    LEGACY_MOE_FLAT_CHECKPOINT_LAYOUT,
    _fill_missing_task_names,
    _lm_eval_spec,
    _transformer_class_for_layout,
    build_grug_logprob_eval_step,
    task_key,
)
from experiments.grug.moe.legacy_model import LegacyTransformer
from experiments.grug.moe.model import GrugModelConfig, Transformer, debug_mesh_and_token_pspec


class _reset_abstract_mesh:
    def __enter__(self):
        self._prev = jax_config.abstract_mesh_context_manager.swap_local(jax_config.config_ext.unset)
        return self

    def __exit__(self, exc_type, exc, tb):
        jax_config.abstract_mesh_context_manager.set_local(self._prev)
        return False


def _small_grug_config() -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=256,
        hidden_dim=32,
        intermediate_dim=64,
        shared_expert_intermediate_dim=64,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=16,
        sliding_window=16,
    )


def test_legacy_transformer_matches_flat_moe_checkpoint_layout():
    mesh, _ = debug_mesh_and_token_pspec(num_devices=4)
    key = jax.random.PRNGKey(0)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        transformer_shape = eqx.filter_eval_shape(LegacyTransformer.init, _small_grug_config(), key=key)

    legacy_mlp = transformer_shape.blocks[0].mlp
    assert hasattr(legacy_mlp, "w_gate_up")
    assert hasattr(legacy_mlp, "w_down")
    assert not hasattr(legacy_mlp, "expert_mlp")


def test_current_transformer_keeps_nested_moe_checkpoint_layout():
    mesh, _ = debug_mesh_and_token_pspec(num_devices=4)
    key = jax.random.PRNGKey(0)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        transformer_shape = eqx.filter_eval_shape(Transformer.init, _small_grug_config(), key=key)

    current_mlp = transformer_shape.blocks[0].mlp
    assert hasattr(current_mlp, "expert_mlp")
    assert not hasattr(current_mlp, "w_gate_up")
    assert not hasattr(current_mlp, "w_down")


def test_build_grug_logprob_eval_step_threads_checkpoint_layout():
    task = GRUG_LOGPROB_TASKS[0]

    step = build_grug_logprob_eval_step(
        run_id="grug_moe_mix_v4_path_r1_t025_d512-2.19e-17",
        hidden_dim=512,
        budget=2.19e17,
        checkpoint_subpath="grug/path/checkpoints",
        task=task,
        max_eval_instances=1,
        checkpoint_layout=LEGACY_MOE_FLAT_CHECKPOINT_LAYOUT,
    )

    assert step.config.checkpoint_layout == LEGACY_MOE_FLAT_CHECKPOINT_LAYOUT
    assert step.config.max_eval_instances == 1
    assert step.config.checkpoint_path.name == "grug/path/checkpoints"
    assert LEGACY_MOE_FLAT_CHECKPOINT_LAYOUT not in step.name
    assert dataclasses.is_dataclass(step.config)


def test_transformer_class_for_layout_selects_matching_checkpoint_tree():
    assert _transformer_class_for_layout(CURRENT_CHECKPOINT_LAYOUT) is Transformer
    assert _transformer_class_for_layout(LEGACY_MOE_FLAT_CHECKPOINT_LAYOUT) is LegacyTransformer


def test_wsc273_task_uses_promptsource_dataset_without_legacy_hf_script():
    wsc_task = next(task for task in GRUG_LOGPROB_TASKS if task.task_alias == "wsc273_0shot")

    assert wsc_task.name == "wsc273_promptsource"
    assert wsc_task.task_kwargs["dataset_path"] == "marcov/winograd_wsc_wsc273_promptsource"
    assert wsc_task.task_kwargs["dataset_name"] == "default"


def test_fill_missing_task_names_repairs_dynamic_lm_eval_configs():
    @dataclasses.dataclass
    class FakeTaskConfig:
        task: str | None = None

    class FakeTask:
        def __init__(self):
            self.config = FakeTaskConfig()

    task = FakeTask()
    nested_task = FakeTask()

    _fill_missing_task_names({"custom_task": task, "group": {"nested_task": nested_task}})

    assert task.config.task == "custom_task"
    assert nested_task.config.task == "nested_task"


def test_grug_logprob_tasks_resolve_registered_lm_eval_task_names():
    from lm_eval.tasks import TaskManager

    task_manager = TaskManager()
    for task in GRUG_LOGPROB_TASKS:
        spec = _lm_eval_spec(task)
        if isinstance(spec, dict) and "dataset_path" in spec:
            continue
        lm_eval_task_name = spec["task"] if isinstance(spec, dict) else spec
        assert task_manager.match_tasks([lm_eval_task_name]), task_key(task)


def test_grug_non_mmlu_sl_verb_aliases_use_valid_base_tasks():
    specs = {task_key(task): _lm_eval_spec(task) for task in GRUG_LOGPROB_TASKS}

    assert specs["boolq_sl_verb_10shot"] == {"task": "boolq", "task_alias": "boolq_sl_verb_10shot"}
    assert specs["csqa_sl_verb_5shot"]["task"] == "commonsense_qa"
    assert specs["csqa_sl_verb_5shot"]["task_alias"] == "csqa_sl_verb_5shot"
    assert "choices['text'][0]" in specs["csqa_sl_verb_5shot"]["doc_to_choice"]
    assert specs["csqa_sl_verb_5shot"]["doc_to_target"] == "{{choices['label'].index(answerKey)}}"
    assert specs["medmcqa_sl_verb_5shot"]["task"] == "medmcqa"
    assert specs["medmcqa_sl_verb_5shot"]["task_alias"] == "medmcqa_sl_verb_5shot"
    assert "opa" in specs["medmcqa_sl_verb_5shot"]["doc_to_choice"]


def test_grug_dynamic_sl_verb_task_kwargs_render_choices_and_targets():
    from lm_eval.tasks import TaskManager, get_task_dict

    specs = {task_key(task): _lm_eval_spec(task) for task in GRUG_LOGPROB_TASKS}
    task_dict = get_task_dict(
        [specs["boolq_sl_verb_10shot"], specs["csqa_sl_verb_5shot"], specs["medmcqa_sl_verb_5shot"]],
        task_manager=TaskManager(),
    )

    assert task_dict["boolq"].doc_to_choice({"label": 1}) == ["no", "yes"]
    assert task_dict["boolq"].doc_to_target({"label": 1}) == 1

    csqa_doc = {
        "choices": {
            "label": ["A", "B", "C", "D", "E"],
            "text": ["bank", "library", "store", "mall", "new york"],
        },
        "answerKey": "C",
    }
    assert task_dict["commonsense_qa"].doc_to_choice(csqa_doc) == [
        "A. bank",
        "B. library",
        "C. store",
        "D. mall",
        "E. new york",
    ]
    assert task_dict["commonsense_qa"].doc_to_target(csqa_doc) == 2

    medmcqa_doc = {
        "opa": "alpha",
        "opb": "beta",
        "opc": "gamma",
        "opd": "delta",
        "cop": 3,
    }
    assert task_dict["medmcqa"].doc_to_choice(medmcqa_doc) == [
        "A. alpha",
        "B. beta",
        "C. gamma",
        "D. delta",
    ]
    assert task_dict["medmcqa"].doc_to_target(medmcqa_doc) == 3


def test_grug_custom_loglikelihood_tasks_emit_numeric_metrics():
    from lm_eval.tasks import TaskManager, get_task_dict

    specs = {
        task_key(task): _lm_eval_spec(task)
        for task in GRUG_LOGPROB_TASKS
        if task.task_alias in {"logprob_gsm8k_5shot", "logprob_humaneval_10shot"}
    }
    task_dict = get_task_dict(list(specs.values()), task_manager=TaskManager())
    _fill_missing_task_names(task_dict)

    gsm_metrics = task_dict["logprob_gsm8k_5shot"].process_results(
        {"answer": "42"},
        [(-10.0, False)],
    )
    humaneval_metrics = task_dict["logprob_humaneval_10shot"].process_results(
        {"canonical_solution": "return 42\n"},
        [(-8.0, False)],
    )

    assert set(gsm_metrics) == {"bpb", "nll"}
    assert set(humaneval_metrics) == {"bpb", "nll"}
    assert gsm_metrics["nll"] == 10.0
    assert humaneval_metrics["nll"] == 8.0
    assert gsm_metrics["bpb"] > 0
    assert humaneval_metrics["bpb"] > 0
