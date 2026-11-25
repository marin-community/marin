# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mochi loop guards (#12264) stage presets: baseline panel, guard arms, and the antithrash pair.

Every stage pins the same panel selection, the same start checkpoint, and the frozen eval
protocol, so the arms differ only in their intervention. The stall stop lives in
``RolloutLimits.repeat_stall_batches``; the decoding arm sets ``repetition_penalty``; the
antithrash pair sets ``generator.trajectory_reward_shaping.antithrash``. See
``cohorts/manifest.json`` for the frozen panel and DESIGN.md §4 for the stage plan.
"""

from dataclasses import replace

from marin.execution.lazy import ArtifactStep
from marin.rl.skyrl import ArtifactHfModel, MegatronGeometry, SkyRLGenerationPlan, SkyRLRetentionPolicy
from skyrlrecipe import ContextBudget, RecipePatch, sections
from skyrlrecipe.declared import EvaluationProfile
from skyrlrecipe.sampling import EvalSamplingParams

from experiments.post_training import mochi_rl
from experiments.post_training.rl_presets import (
    STOP_TOKEN_IDS,
    Evaluation,
    Generation,
    Layout,
    Preset,
    Recipe,
)

VERSION = "2026.11.25.1"
PANEL_SELECTION = f"users/benfeuer/mochi-loop-guards/panel@{VERSION}"
POOL_SELECTION = f"users/benfeuer/mochi-loop-guards/snowball-100k-agentic-ho-excluded@{VERSION}"
"""The RL task pool: snowball-100k-agentic minus the 128 swesmith-ho held-out row hashes."""

PANEL_ROOT = "s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards"
START_CHECKPOINT = (
    "s3://marin-us-east-08a/marin/models/mochi-sft/hero-sft-fullab-step393k-v2/step-400407/2af3f828aa-cpu-a1"
)
"""Mochi SFT step 400407, the default start unless Dr Doom's export seals with success (#12264 §7)."""

MODEL = ArtifactHfModel(
    ArtifactStep.adopt(
        "models/mochi-sft/hero-sft-fullab-step393k-v2/step-400407",
        "2026.11.13",
        START_CHECKPOINT,
    ),
    relative_path="",
)

# The frozen eval protocol (cohorts/manifest.json): the agent gets 1,800 s and the verifier
# 300 s (CAMPAIGN_JOB), the input window is 32,768 tokens, and each reply is capped at 8,192.
PANEL_AGENT_TIMEOUT = 1_800
PANEL_VERIFIER_TIMEOUT = 300
PANEL_MAX_INPUT_TOKENS = 32_768
PANEL_MAX_REPLY_TOKENS = 8_192
PANEL_MODEL_TURN_TIMEOUT = 600
PANEL_ATTEMPT_TIMEOUT = 8 * 60 * 60
TERMINUS_CONTEXT = ContextBudget(
    request_window_tokens=PANEL_MAX_INPUT_TOKENS,
    max_new_tokens_per_turn=PANEL_MAX_REPLY_TOKENS,
    # Harbor sets no turn cap; the Micro210 policy's 1,024-turn budget is the native
    # equivalent, and the agent and attempt timeouts bound the trial regardless.
    max_turns=1024,
)
PANEL_SAMPLING = EvalSamplingParams(
    temperature=0.7,
    top_p=1.0,
    top_k=-1,
    max_generate_length=PANEL_MAX_REPLY_TOKENS,
    stop_token_ids=STOP_TOKEN_IDS,
    skip_special_tokens=False,
)

# Layout: 4 GB200 per task, the GB200 variant, cw-us-east-08a, excluding the faulty rack,
# and 72 h coordinator/job timeouts (#12264 §4). This mirrors Dr Doom's BASELINE_LAYOUT
# plus the mochi-gb200-cw-us-east-08a serving profile's exclusion. UNVERIFIED RESOLVED
# INFEASIBLE on main: role_topology=RoleTopologyMode.NVLINK_DOMAIN cannot plan a
# generation-only job — the native role-plan validator requires exactly one 'policy' claim
# (vendor/skyrl/cloud/iris/role_plan.py) and _launch_config_yaml reads
# plan.policy_num_nodes, which a generation plan does not carry. Dr Doom's generation
# support is not on main. The exclusion and the 72 h budgets still apply; the recorded
# proposed change is to extend the native role plan to generation-only claims. See
# PREP-STATUS.md.
GUARDS_LAYOUT = Layout(
    name="mochi-guards-generate-128",
    workload=Generation(
        SkyRLGenerationPlan(
            gpus_per_node=4,
            num_inference_engines=8,
            inference_engine_tensor_parallel_size=1,
            inference_engine_pipeline_parallel_size=1,
            inference_engine_data_parallel_size=16,
            inference_engine_expert_parallel_size=16,
            n_samples_per_prompt=1,
        )
    ),
    megatron=MegatronGeometry(1, 12, 4, 4, 1),
    cluster="cw-us-east-08a",
    host_memory="900GB",
    gpu_variant="GB200",
    host_cpu=120,
    host_disk="2TB",
    priority="production",
    coordinator_timeout_hours=72,
    job_timeout_seconds=72 * 3600,
    excluded_nvlink_domains=("dh1-r136-us-east-08a",),
)

RL_LAYOUT = replace(
    mochi_rl.ONE_RACK,
    name="mochi-guards-one-rack",
    coordinator_timeout_hours=72,
    job_timeout_seconds=72 * 3600,
    excluded_nvlink_domains=("dh1-r136-us-east-08a",),
)
"""One 60-GB200 rack (48 policy + 12 inference) for each S2 arm, with the same exclusions."""

EXPORT_LAYOUT = replace(
    mochi_rl.THREE_RACK,
    name="mochi-guards-export-96",
    coordinator_timeout_hours=72,
    job_timeout_seconds=72 * 3600,
    excluded_nvlink_domains=("dh1-r136-us-east-08a",),
)
"""The 96-GPU export grid for the S2x exports (two arms of 48 pipeline-parallel GPUs)."""

PANEL_ENVIRONMENT = RecipePatch(
    environment=sections.Environment(
        task_sessions={
            "daytona": {"api_key_env": "DAYTONA_EVAL_API_KEY", "eval_api_key_env": "DAYTONA_EVAL_API_KEY"},
            "task_spec": {
                "session": {
                    "model_turn_timeout": PANEL_MODEL_TURN_TIMEOUT,
                    "total_turn_timeout": PANEL_AGENT_TIMEOUT,
                    "attempt_timeout": PANEL_ATTEMPT_TIMEOUT,
                    "verifier_timeout": PANEL_VERIFIER_TIMEOUT,
                },
            },
        }
    ),
    context_budget=TERMINUS_CONTEXT,
    generator=sections.Generator(
        max_num_seqs=128,
        engine_init_timeout_seconds=7200,
        trajectory_retention=mochi_rl.SAMPLED_TRAJECTORIES.replace(
            phases=("eval",),
            sample_fraction=0.0,
            eval_sample_fraction=1.0,
            sample_count_per_step=0,
            required=True,
            always_retain_failures=True,
            always_retain_non_terminating=True,
            always_retain_loops=True,
            max_bytes_per_step=None,
            max_bytes_per_run=None,
        ),
    ),
)
"""One shared panel environment: the frozen protocol and full eval-trajectory retention."""


PANEL_COHORTS = (
    ("tb2", "tb2", 3),
    ("tblite", "tblite", 2),
    ("swe100", "swe100", 1),
    ("swesmith_ho", "swesmith_ho", 2),
)


def evaluation(name: str, file: str, trials: int, *, sampling: EvalSamplingParams = PANEL_SAMPLING) -> Evaluation:
    """One panel cohort: a selection file run at its frozen trial count and sampling."""
    return Evaluation(
        name=name,
        file=file,
        source="held_out",
        profile=EvaluationProfile(n_samples_per_prompt=trials, sampling_params=sampling),
    )


PANEL_EVALUATIONS = tuple(evaluation(name, file, trials) for name, file, trials in PANEL_COHORTS)


def panel_preset(
    name: str,
    *,
    model: ArtifactHfModel = MODEL,
    repeat_stall_batches: int | None = None,
    repetition_penalty: float = 1.0,
) -> Preset:
    """One panel stage: S0 baseline, S1-stop, or S1-rp, differing only in its intervention."""
    environment = mochi_rl._ONE_TURN.merge(PANEL_ENVIRONMENT)
    if repeat_stall_batches is not None:
        environment = environment.merge(
            RecipePatch(
                environment=sections.Environment(
                    task_sessions={
                        "task_spec": {"session": {"repeat_stall_batches": repeat_stall_batches}},
                    }
                )
            )
        )
    sampling = PANEL_SAMPLING.replace(repetition_penalty=repetition_penalty)
    evaluations = tuple(evaluation(name, file, trials, sampling=sampling) for name, file, trials in PANEL_COHORTS)
    return Preset(
        name=name,
        revision=VERSION,
        recipe=Recipe(layout=GUARDS_LAYOUT, fixed_reply_tokens=None, config=environment),
        updates=0,
        measured=range(1),
        tasks=PANEL_SELECTION,
        held_out_tasks=PANEL_SELECTION,
        model=model,
        evaluations=evaluations,
        accepts_init=True,
        seed=20261125,
        sample_namespace="datasets/mochi-loop-guards",
        wandb_entity="marin-community",
        checkpoints="none",
        retention=SkyRLRetentionPolicy(resume_checkpoint_count=4, temporary_storage_ttl_days=3),
        temporary_output=False,
        budget="128 GB200, one panel pass (about 4 h of trials)",
    )


def _arm_patches(antithrash: bool) -> RecipePatch:
    """The S2 arm's only difference from the shared agentic recipe: its antithrash setting.

    The stall stop stays off in both arms: a stop at three repeats would hide exactly what
    antithrash penalizes (#12264 §3, I3). The shaping section enables the antithrash charge
    with the design defaults (0.02 per repeat, cap 0.1) and the arm's scope.
    """
    return RecipePatch(
        generator=sections.Generator(
            trajectory_reward_shaping=sections.TrajectoryRewardShaping(
                enabled=True,
                antithrash=sections.Antithrash(
                    enabled=antithrash,
                    applies_to="correct_only" if antithrash else "all",
                ),
            ),
        ),
    )


def rl_arm(name: str, *, antithrash: bool, updates: int) -> Preset:
    """One S2 arm: control (antithrash off) or treatment (correct_only), on its own rack."""
    base = replace(
        mochi_rl._TRAIN_LONG_AGENTIC,
        name=name,
        revision=VERSION,
        recipe=replace(
            mochi_rl._TRAIN_LONG_AGENTIC.recipe,
            layout=RL_LAYOUT,
            config=mochi_rl._TRAIN_LONG_AGENTIC.recipe.config.merge(_arm_patches(antithrash)),
        ),
        updates=updates,
        measured=range(1, updates + 1),
        tasks=POOL_SELECTION,
        held_out_tasks=None,
        evaluations=(),
        checkpoint_evaluation=(),
        seed=20261125,
        sample_namespace="datasets/mochi-loop-guards",
        checkpoints="none",
        temporary_output=True,
        budget="60 GB200 per arm; N updates fixed at submit from S0's measured trial times",
    )
    return base


S0 = panel_preset("s0-baseline")
S1_STOP = panel_preset("s1-stop", repeat_stall_batches=3)
S1_RP = panel_preset("s1-rp", repetition_penalty=1.1)
S2_CONTROL = rl_arm("s2-antithrash-control", antithrash=False, updates=8)
S2_TREATMENT = rl_arm("s2-antithrash-treatment", antithrash=True, updates=8)


def s3_panel(name: str, export: ArtifactHfModel) -> Preset:
    """One S3 arm: the full panel on one S2 export, run exactly like S0.

    The export binds at T+34 from the S2x export artifact; the plan preflight validates the
    stage with the declared start checkpoint standing in for the export.
    """
    return panel_preset(name, model=export)


def export_command(ckpt_path: str, step: int, launch_config: str, *, export_path: str | None = None) -> str:
    """The S2x export command for one arm: 48 export GPUs derived from the run's launch config.

    The export gang restores the saved policy geometry and writes the HF export beside the
    checkpoint; the S3 panel then consumes that export as its initializer.
    """
    destination = export_path or f"{ckpt_path.rstrip('/')}/exports"
    return (
        "uv run --project vendor/skyrl python vendor/skyrl/cloud/iris/export_hf_checkpoint.py"
        f" --ckpt_path {ckpt_path} --step {step} --launch-config {launch_config}"
        f" --cluster cw-us-east-08a --priority production --num-nodes 12 --gpus-per-node 4"
        f" --gpu-variant GB200 --export_path {destination} --dry-run"
    )


if __name__ == "__main__":
    from experiments.post_training.rl_presets import ModelPresets

    GUARDS = ModelPresets(
        model="mochi",
        presets=(S0, S1_STOP, S1_RP, S2_CONTROL, S2_TREATMENT),
        namespace="mochi-loop-guards",
        control_pins={},
        needs_completed_default_export=True,
    )
    print(GUARDS.name, [preset.name for preset in GUARDS.presets])
