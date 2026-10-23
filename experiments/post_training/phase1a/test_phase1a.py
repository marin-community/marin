# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 1a tests: the matrix's arithmetic, the shared prompt order, and M1/M4
agreement on every matched field after SkyRL composes both launch configs."""

import json
import subprocess

import pytest
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import ArtifactStep, StepContext
from marin.rl.skyrl import _launcher_command as launcher_command_for_test
from marin.rl.skyrl import skyrl_step
from marin.training.training import LevanterCheckpoint

from experiments.post_training.phase1a.data import eligible_samples, layout
from experiments.post_training.phase1a.launch import execution, run_spec
from experiments.post_training.phase1a.matrix import (
    LR_SWEEP,
    PHASE1A_CONDITIONS,
    SAMPLES_PER_PROMPT,
    STEPS,
    TEACHER_ROUTES,
    Condition,
    Method,
    RunPlan,
    Task,
    TeacherModel,
    phase0c_lr_sweep,
    phase1a_runs,
    role_plan,
)

TEACHERS = {
    Task.X: TeacherModel("open-athena/e-x", "1" * 40),
    Task.R: TeacherModel("open-athena/e-r", "2" * 40),
    Task.GENERAL: TeacherModel("open-athena/b", "3" * 40),
}


def test_matrix_holds_thirty_cells_two_nulls_and_an_eight_run_sweep():
    runs = phase1a_runs(1e-6)
    assert len(runs) == 32
    assert sum(run.condition is Condition.PROCEDURE_NULL for run in runs) == 2
    assert {run.learning_rate for run in runs} == {1e-6}
    sweep = phase0c_lr_sweep()
    assert len(sweep) == 8 and {run.condition for run in sweep} == {Condition.X}
    assert {(run.method, run.learning_rate) for run in sweep} == {(m, rate) for m in Method for rate in LR_SWEEP}


@pytest.mark.parametrize("condition", [*PHASE1A_CONDITIONS, Condition.PROCEDURE_NULL])
def test_both_arms_consume_the_same_sequences_per_step(condition):
    m1 = role_plan(RunPlan(Method.M1, condition, 0, 1e-6))
    m4 = role_plan(RunPlan(Method.M4, condition, 0, 1e-6))
    assert m1.train_batch_size * m1.n_samples_per_prompt == m4.train_batch_size * m4.n_samples_per_prompt
    assert m1.policy_mini_batch_size == m1.train_batch_size and m4.policy_mini_batch_size == m4.train_batch_size
    expected = {Condition.XR: 64, Condition.X_PRIME: 64, Condition.R_PRIME: 64}.get(condition, 32)
    assert m4.train_batch_size * SAMPLES_PER_PROMPT == expected


def _samples(task: Task, task_ids: list[str], per_prompt: int) -> list[dict]:
    return [
        {
            "task_id": task_id,
            "data_source": str(task),
            "teacher_route": TEACHER_ROUTES[task],
            "input_ids": [index, sample],
            "assistant_masks": [0, 1],
        }
        for index, task_id in enumerate(task_ids)
        for sample in range(per_prompt)
    ]


def test_m1_step_trains_on_the_m4_steps_prompts_in_order():
    run = RunPlan(Method.M4, Condition.XR, 1, 1e-6)
    ids = {task: [f"{task}-{index}" for index in range(STEPS * 8 + 5)] for task in (Task.X, Task.R)}
    rows = {task: _samples(task, ids[task], 4) + _samples(task, [f"{task}-short"], 3) for task in ids}
    samples = {task: eligible_samples(rows[task], task) for task in ids}
    assert all(f"{task}-short" not in samples[task] for task in ids), "a prompt needs 4 expert samples"
    prompts = {
        task: {task_id: {"prompt": [{"role": "user", "content": task_id}]} for task_id in ids[task]} for task in ids
    }
    m4_rows, m1_rows = layout(run, prompts, samples)

    assert len(m4_rows) == STEPS * 16 and len(m1_rows) == STEPS * 64
    for step in range(STEPS):
        m4_step = [row["task_id"] for row in m4_rows[step * 16 : (step + 1) * 16]]
        m1_step = [row["task_id"] for row in m1_rows[step * 64 : (step + 1) * 64]]
        assert m1_step == [task_id for task_id in m4_step for _ in range(4)]
        assert [row["teacher_route"] for row in m4_rows[step * 16 : (step + 1) * 16]] == ["E_X"] * 8 + ["E_R"] * 8
    assert len({row["task_id"] for row in m4_rows}) == len(m4_rows), "each prompt is used once"
    assert {row["env_class"] for row in m4_rows} == {"prompt_only"}


def test_a_short_prompt_pool_raises():
    run = RunPlan(Method.M1, Condition.X_PRIME, 0, 1e-6)
    ids = [f"x-{index}" for index in range(STEPS * 16 - 1)]
    samples = {Task.X: eligible_samples(_samples(Task.X, ids, 4), Task.X)}
    with pytest.raises(ValueError, match="needs 1600"):
        layout(run, {Task.X: {}}, samples)


# --- matched fields through SkyRL's own launcher ---

COMPOSE = """
import json
import sys
from pathlib import Path

from cloud.iris.launch_config import load_launch_config
from skyrl_train.config.training_budget import training_budget

def matched(path):
    cfg = load_launch_config(Path(path)).skyrl
    budget = training_budget(cfg)
    optimizer = cfg.trainer.policy.optimizer_config
    return {
        "sequences_per_step": budget.sequences_per_step,
        "sequences_per_mini_batch": budget.sequences_per_mini_batch,
        "max_input_tokens": budget.max_input_tokens,
        "response_caps": dict(budget.response_caps.by_data_source),
        "optimizer": [optimizer.optimizer, optimizer.lr, list(optimizer.adam_betas), optimizer.weight_decay,
                      optimizer.max_grad_norm, optimizer.num_warmup_steps, optimizer.scheduler],
        "steps": [cfg.trainer.max_steps, cfg.trainer.epochs, cfg.trainer.update_epochs_per_batch],
        "seed": cfg.trainer.seed,
        "packing": cfg.trainer.use_sample_packing,
        "order": [cfg.data.shuffle, cfg.data.sampling.kind],
        "checkpoints": [list(cfg.trainer.checkpoint_steps), list(cfg.trainer.optimizer_checkpoint_steps)],
        "router_telemetry": cfg.trainer.router_telemetry.enabled,
        "strategy": cfg.trainer.strategy,
        "env_class": cfg.environment.env_class,
    }

print(json.dumps([matched(path) for path in sys.argv[1:]]))
"""


@pytest.mark.timeout(300)  # A cold cache installs the independently locked launcher.
def test_m1_and_m4_agree_on_every_matched_field_after_composition(tmp_path):
    model = ArtifactStep.adopt("models/phase1a-test-b", "2026.10.23", "s3://bucket/b", kind=LevanterCheckpoint)
    data = ArtifactStep.adopt("documents/phase1a-test-data", "2026.10.23", "s3://bucket/data")
    cells = [(condition, 1) for condition in PHASE1A_CONDITIONS] + [(Condition.PROCEDURE_NULL, 0)]
    paths = []
    with build_context(BuildContext(versions=VersionCodex(default="2026.10.23"))):
        for condition, seed in cells:
            for method in Method:
                run = RunPlan(method, condition, seed, 2e-6)
                teachers = {task: TEACHERS[task] for task in run.tasks}
                spec = run_spec(run, model=model, run_data=data, teachers=teachers)
                step = skyrl_step(spec, execution(), export_hf=True)
                config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
                path = tmp_path / f"{run.name}.yaml"
                path.write_text(config.launch_config_yaml)
                paths.append(str(path))
    command = launcher_command_for_test(paths[0])
    bootstrap = command[: command.index("marinskyrl")]
    result = subprocess.run([*bootstrap, "python", "-c", COMPOSE, *paths], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr[-4000:]
    composed = json.loads(result.stdout)
    for index, (condition, _) in enumerate(cells):
        m1, m4 = composed[2 * index], composed[2 * index + 1]
        assert m1.pop("env_class") == "fixed_completion" and m4.pop("env_class") == "prompt_only"
        assert m1 == m4, condition
        assert m4["order"] == [False, None]
        assert m4["optimizer"] == ["AdamW", 2e-6, [0.9, 0.999], 0.0, 1.0, 5, "constant_with_warmup"]
