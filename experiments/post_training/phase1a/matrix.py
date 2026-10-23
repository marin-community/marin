# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 1a's run matrix: every run's identity, role plan, and SkyRL recipe.

Phase 1a compares SFT on expert samples (M1) with on-policy distillation (M4) on
Snowball at router level R1. Thirty runs cover two methods x five conditions x
three seeds; two procedure nulls train each method on B's own samples. Phase 0c
sweeps four learning rates over both methods on ``theta_X``.

The two arms of a cell differ only in the training signal. Both read the same
prompts in the same order with ``data.shuffle=false`` (see
:mod:`experiments.post_training.phase1a.data`): an M4 step rolls out 8 prompts x 4
samples, and the matching M1 step trains on those 8 prompts' 4 expert completions.
Optimizer, schedule, steps, sequences per step, response caps, packing, checkpoint
schedule, router policy and seed are rendered from the constants below for both
arms; nothing is inherited silently from a base recipe.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

import yaml
from marin.rl.skyrl import SkyRLRolePlan, SkyRLTopology

# M4 starts from the Snowball OPD recipe (WORK_BRIEF WP10).
M4_BASE_RECIPE = Path("vendor/skyrl/cloud/iris/configs/snowball_mopd_ultra_32k.yaml")


class Method(StrEnum):
    M1 = "m1"
    M4 = "m4"


class Task(StrEnum):
    """A training task, named by its rows' ``data_source``."""

    X = "task_X"
    R = "task_R"
    GENERAL = "general"


class Condition(StrEnum):
    X = "theta_X"
    R = "theta_R"
    XR = "theta_XR"
    X_PRIME = "theta_X_prime"
    R_PRIME = "theta_R_prime"
    PROCEDURE_NULL = "procedure_null"


# Sequences per step by task (WORK_BRIEF fixed parameters; control C6).
SEQUENCES_PER_STEP: MappingProxyType[Condition, MappingProxyType[Task, int]] = MappingProxyType(
    {
        Condition.X: MappingProxyType({Task.X: 32}),
        Condition.R: MappingProxyType({Task.R: 32}),
        Condition.XR: MappingProxyType({Task.X: 32, Task.R: 32}),
        Condition.X_PRIME: MappingProxyType({Task.X: 64}),
        Condition.R_PRIME: MappingProxyType({Task.R: 64}),
        Condition.PROCEDURE_NULL: MappingProxyType({Task.GENERAL: 32}),
    }
)
# Response caps, thinking included, for teacher samples, expert samples and rollouts alike.
RESPONSE_CAP_TOKENS: MappingProxyType[Task, int] = MappingProxyType({Task.X: 4_096, Task.R: 16_384, Task.GENERAL: 4_096})
PROMPT_CAP_TOKENS = 4_096
# The teacher route of each task's rows; WP7 writes these on the M1 rows.
TEACHER_ROUTES: MappingProxyType[Task, str] = MappingProxyType({Task.X: "E_X", Task.R: "E_R", Task.GENERAL: "B"})
SAMPLES_PER_PROMPT = 4
ROLLOUT_TEMPERATURE = 1.0
STEPS = 100
WARMUP_STEPS = 5
ADAM_BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0
CHECKPOINT_STEPS = (10, 20, 30, 40, 50, 75, 100)
OPTIMIZER_CHECKPOINT_STEPS = (10, 50, 100)
SEEDS = (0, 1, 2)
PHASE1A_CONDITIONS = (Condition.X, Condition.R, Condition.XR, Condition.X_PRIME, Condition.R_PRIME)
# Phase 0c sweeps {0.5, 1, 2, 4} x 1e-6 over M1 and M4 on theta_X (control C10).
LR_SWEEP = (5e-7, 1e-6, 2e-6, 4e-6)

# Snowball learner geometry from the OPD recipe: 4 nodes, PP2 x EP8.
GPUS_PER_NODE = 8
POLICY_NODES = 4
M4_ROLLOUT_ENGINES = 2


@dataclass(frozen=True)
class TeacherModel:
    """A teacher checkpoint at a pinned revision, as SkyRL's teacher spec takes it."""

    path: str
    revision: str


@dataclass(frozen=True)
class RunPlan:
    method: Method
    condition: Condition
    seed: int
    learning_rate: float

    @property
    def name(self) -> str:
        return f"phase1a-{self.method}-{self.condition}-lr{self.learning_rate:g}-seed{self.seed}"

    @property
    def tasks(self) -> tuple[Task, ...]:
        return tuple(SEQUENCES_PER_STEP[self.condition])

    @property
    def sequences_per_step(self) -> int:
        return sum(SEQUENCES_PER_STEP[self.condition].values())

    @property
    def prompts_per_step(self) -> dict[Task, int]:
        """Prompts per step by task; M1 trains on each prompt's 4 expert completions."""
        return {task: count // SAMPLES_PER_PROMPT for task, count in SEQUENCES_PER_STEP[self.condition].items()}


def phase1a_runs(learning_rate: float) -> list[RunPlan]:
    """Thirty cells plus one procedure null per method, all at one common learning rate."""
    runs = [
        RunPlan(method, condition, seed, learning_rate)
        for method in Method
        for condition in PHASE1A_CONDITIONS
        for seed in SEEDS
    ]
    return runs + [RunPlan(method, Condition.PROCEDURE_NULL, SEEDS[0], learning_rate) for method in Method]


def phase0c_lr_sweep() -> list[RunPlan]:
    """Four learning rates x both methods on theta_X."""
    return [RunPlan(method, Condition.X, SEEDS[0], rate) for rate in LR_SWEEP for method in Method]


def role_plan(run: RunPlan) -> SkyRLRolePlan:
    """One optimizer step per training step: the mini-batch is the whole batch.

    M4's batch counts prompts and samples 4 rollouts each; M1's batch counts rows.
    M1 plans no inference engines: fixed-data runs claim only the learner nodes.
    """
    if run.method is Method.M4:
        prompts = run.sequences_per_step // SAMPLES_PER_PROMPT
        return SkyRLRolePlan(
            colocate_all=False,
            policy_num_nodes=POLICY_NODES,
            policy_num_gpus_per_node=GPUS_PER_NODE,
            num_inference_engines=M4_ROLLOUT_ENGINES,
            inference_engine_tensor_parallel_size=1,
            inference_engine_pipeline_parallel_size=1,
            inference_engine_data_parallel_size=GPUS_PER_NODE,
            inference_engine_expert_parallel_size=GPUS_PER_NODE,
            train_batch_size=prompts,
            policy_mini_batch_size=prompts,
            micro_train_batch_size_per_gpu=1,
            n_samples_per_prompt=SAMPLES_PER_PROMPT,
        )
    return SkyRLRolePlan(
        colocate_all=False,
        policy_num_nodes=POLICY_NODES,
        policy_num_gpus_per_node=GPUS_PER_NODE,
        num_inference_engines=0,
        inference_engine_tensor_parallel_size=1,
        inference_engine_pipeline_parallel_size=1,
        inference_engine_data_parallel_size=GPUS_PER_NODE,
        inference_engine_expert_parallel_size=GPUS_PER_NODE,
        train_batch_size=run.sequences_per_step,
        policy_mini_batch_size=run.sequences_per_step,
        micro_train_batch_size_per_gpu=1,
        n_samples_per_prompt=1,
    )


def topology(run: RunPlan) -> SkyRLTopology:
    """M4 adds two rollout nodes and one pinned node per teacher it routes to."""
    plan = role_plan(run)
    nodes = POLICY_NODES if run.method is Method.M1 else POLICY_NODES + M4_ROLLOUT_ENGINES + len(run.tasks)
    return SkyRLTopology(num_nodes=nodes, gpus_per_node=GPUS_PER_NODE, gpu_variant="H100", role_plan=plan)


_ROLE_PLAN_KEYS = {
    "trainer": ("train_batch_size", "policy_mini_batch_size", "micro_train_batch_size_per_gpu", "placement"),
    "generator": (
        "num_inference_engines",
        "inference_engine_tensor_parallel_size",
        "inference_engine_pipeline_parallel_size",
        "inference_engine_data_parallel_size",
        "inference_engine_expert_parallel_size",
        "n_samples_per_prompt",
    ),
}
# Harbor and agent settings of the OPD recipe that single-turn X and R rows never use.
_UNUSED_SECTIONS = ("config_groups", "terminal_bench")


def _base_recipe() -> dict:
    recipe = yaml.safe_load(M4_BASE_RECIPE.read_text())
    for section in _UNUSED_SECTIONS:
        del recipe[section]
    for section, keys in _ROLE_PLAN_KEYS.items():
        for key in keys:
            recipe[section].pop(key, None)
    del recipe["entrypoint"]
    return recipe


def _teacher_name(task: Task) -> str:
    return TEACHER_ROUTES[task].lower()


def skyrl_recipe(run: RunPlan, teachers: dict[Task, TeacherModel]) -> dict:
    """The ``skyrl`` subtree for one run, with every matched field rendered explicitly.

    ``teachers`` maps each task to its expert (E_X, E_R) or, for the procedure null,
    to B. M4 scores rollouts with them; M1's rows were sampled from them.
    """
    if set(teachers) != set(run.tasks):
        raise ValueError(f"{run.name} needs teachers for {sorted(run.tasks)}, got {sorted(teachers)}")
    recipe = _base_recipe()
    caps = {str(task): RESPONSE_CAP_TOKENS[task] for task in run.tasks}
    longest = max(caps.values())
    trainer = recipe["trainer"]
    trainer.update(
        use_sample_packing=False,
        epochs=1,
        max_steps=STEPS,
        update_epochs_per_batch=1,
        seed=run.seed,
        checkpoint_steps=list(CHECKPOINT_STEPS),
        optimizer_checkpoint_steps=list(OPTIMIZER_CHECKPOINT_STEPS),
        ckpt_interval=CHECKPOINT_STEPS[0],
        resume_mode="latest",
        micro_forward_batch_size_per_gpu=1,
        evaluation_runner="training",
        router_telemetry={"enabled": True},
    )
    trainer["policy"]["optimizer_config"].update(
        optimizer="AdamW",
        lr=run.learning_rate,
        adam_betas=list(ADAM_BETAS),
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        num_warmup_steps=WARMUP_STEPS,
        scheduler="constant_with_warmup",
    )
    recipe["data"] = {"kind": "parquet", "train_data": [], "val_data": [], "shuffle": False, "sampling": {"kind": None}}
    recipe["context_budget"] = {
        "request_window_tokens": PROMPT_CAP_TOKENS + longest,
        "max_new_tokens_per_turn": longest,
        "max_turns": 1,
        "generated_budget_fraction": 1.0,
    }
    generator = recipe["generator"]
    generator["max_generate_length_by_data_source"] = caps
    generator["sampling_params"]["temperature"] = ROLLOUT_TEMPERATURE
    generator["engine_init_kwargs"].pop("enable_auto_tool_choice")
    generator["engine_init_kwargs"].pop("tool_call_parser")

    if run.method is Method.M4:
        recipe["environment"] = {"env_class": "prompt_only"}
        template = next(iter(recipe["teachers"].values()))
        recipe["teachers"] = {
            _teacher_name(task): {
                **copy.deepcopy(template),
                "model": {"path": teachers[task].path, "revision": teachers[task].revision},
                "resources": {**template["resources"], "colocation_group": f"teacher_{_teacher_name(task)}"},
            }
            for task in run.tasks
        }
        recipe["teacher_routing"] = {
            "opd": {
                "revision": f"phase1a-{run.condition}",
                "routes": {TEACHER_ROUTES[task]: {"teacher": _teacher_name(task), "weight": 1.0} for task in run.tasks},
            }
        }
        return recipe

    del recipe["teachers"], recipe["teacher_routing"]
    recipe["environment"] = {"env_class": "fixed_completion"}
    recipe["data"]["fixed_completion_format"] = "tokenized"
    algorithm = trainer["algorithm"]
    del algorithm["distillation"]
    algorithm.update(policy_loss_type="sft", advantage_estimator="uniform")
    return recipe
