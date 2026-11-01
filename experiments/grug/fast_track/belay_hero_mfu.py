# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Marin Belay hero-mfu track: the one-rack GB200 MFU stack, launched exactly as it was measured.

``mfl-mp-qall`` (c6d473c243, 2026-10-21) ran 11.465 s/step wall clock over steps 6-23 (36.8% MFU) on one GB200
NVL72 rack of cw-us-east-08a, at production priority. Every argument below was recovered from the launch scripts
and confirmed field by field against the run's W&B config.

    # plan only: print the launcher arguments, submit nothing
    uv run python -m experiments.grug.fast_track.belay_hero_mfu --stack mfl-mp-qall --plan
    # first time for a program: seed the XLA autotune cache on one GPU (the timed run refuses without it)
    uv run python -m experiments.grug.fast_track.belay_hero_mfu --stack mfl-mp-qall --seed-xla-autotune-cache \\
        --run-id <id>-seed --submit
    # the timed run
    uv run python -m experiments.grug.fast_track.belay_hero_mfu --stack mfl-mp-qall [--model-set ...] \\
        --run-id <id> --submit --target-cluster cw-us-east-08a --priority production
    # acceptance and step time of finished runs
    uv run python -m experiments.grug.fast_track.belay_hero_mfu --accept <run> [<run> ...]

A run counts only if its routing is healthy (``ACCEPTANCE``): mean load-balancing loss 8.000-8.001 (no layer above
8.007), capacity drops below 1e-5, router bias within +-0.43. Compare a variant only with a control on the same
rack, at the same commit, with the same PGLE profile; a win is a step at least 0.5% faster.
"""

import dataclasses
import statistics
import sys

import wandb

from experiments.grug.fast_track import launch

PGLE_DIR = "experiments/grug/fast_track/pgle"
WANDB_PROJECTS = ("marin-community/marin_moe", "marin-community/marin")
LB_PREFIX, LB_SUFFIX = "train/router/layer_", "/load_balancing_loss"
# Router settings of the S7 stack: base hero settings, then the consolidated (S5), KDA (S6) and MLA/Inkling (S7) arms.
S7_MODEL_SETTINGS = (
    "local_kv_heads=12",
    "global_kv_heads=6",
    "num_kv_heads=12",
    "capacity_factor=1.15",
    "moe_implementation=ragged_all_to_all",
    "moe_shared_placement=overlap",
    "gpu_attention=gpu_fa4_cute_sm100",
    "moe_ungated_relu2=true",
    "moe_ungated_kernel=true",
    "shared_ungated_relu2=true",
    "intermediate_dim=4608",
    "shared_expert_intermediate_dim=4608",
    "num_shared_experts=2",
    "num_experts_per_token=8",
    "latent_dim=3072",
    "fp8_recipe=mxfp8",
    "moe_mxfp8_dispatch=false",
    "mxfp8_fused_grad_quant=true",
    "moe_ragged_chunks=2",
    "sconv_impl=triton_gpu",
    "attn_res=true",
    "scan_layers=true",
    "scan_offload_partials=true",
    "scan_save_routed_output=true",
    "attn_res_fused_source_grads=true",
    "mxfp8_expert_wgrad=true",
    "attn_res_num_blocks=8",
    "attn_res_plain_sum=false",
    "attn_res_fused_block_mix=true",
    "local_mixer=kda",
    "kda_chunk_size=32",
    "kda_onchip_decay_gate=true",
    "kda_state_block_v=64",
    "kda_state_grads=pallas",
    "local_kv_heads=6",
    "num_kv_heads=6",
    "mla=true",
    "inkling_relpos=true",
    "mla_kv_latent_dim=512",
    "mla_num_heads=48",
    "qk_norm=false",
    "mla_q_norm=true",
    "mla_k_norm=true",
    "rel_extent=512",
    "gpu_attention=gpu_pallas_fa4",
)
# The head of the 390251-step production schedule over a 30-step run: MuonH / Adam peaks x 30 / 3902, warmup over it.
HERO_OPT_SETTINGS = (
    "warmup=1.0",
    "learning_rate=1.7491280826995436e-05",
    "adam_lr=4.0364494216143314e-06",
    "epsilon=6.044604247506697e-15",
)


@dataclasses.dataclass(frozen=True)
class Stack:
    model_settings: tuple[str, ...]
    pgle_profile: str
    slop_factor: int
    steps: int = 30
    mem_fraction: float = 0.78


STACKS = {
    "mfl-mp-qall": Stack(
        model_settings=(
            *S7_MODEL_SETTINGS,
            "kda_state_primal_block_v=64",
            "kda_state_num_stages=3",
            "gated_norm_implementation=fused",
            "gpu_attention=gpu_fa4_cute_sm100",
            "scan_split_backward_layers=0,1,2,3",
            "quack_gemm_table=tuned",
            "capacity_factor=1.05",
        ),
        # From run mfl-mp-m7g (the S7 split-backward + expert_down program), committed in e78b3c005b.
        pgle_profile="mfl-mp-m7g.pb",
        slop_factor=112,
    ),
}


@dataclasses.dataclass(frozen=True)
class Acceptance:
    lb_mean: tuple[float, float] = (8.000, 8.001)
    lb_layer_max: float = 8.007
    drops_max: float = 1e-5
    bias_abs_max: float = 0.43
    steps: tuple[int, int] = (6, 20)
    """Routing is checked over steps [6, 20); step time is wall clock over [6, 24)."""


ACCEPTANCE = Acceptance()
WALL_STEPS = (6, 24)


def stack_args(name: str, *, seed_autotune: bool = False) -> list[str]:
    """Launcher arguments for ``name`` without the run id, cluster, priority and ``--submit``."""
    stack = STACKS[name]
    xla_flags = " ".join(
        (
            "--xla_gpu_enable_latency_hiding_scheduler=true",
            "--xla_gpu_enable_command_buffer=",
            f"--xla_gpu_memory_limit_slop_factor={stack.slop_factor}",
            f"--xla_gpu_pgle_profile_file_or_directory_path={PGLE_DIR}/{stack.pgle_profile}",
            "--xla_gpu_enable_host_memory_offloading=true",
        )
    )
    args = ["--size", "gb200-d6144", "--recipe", "baseline", "--version", "2026.10.18"]
    args += ["--ragged-transport", "device", "--master-param-mode", "device", "--num-steps", str(stack.steps)]
    args += ["--no-eval", "--job-env", f"XLA_FLAGS={xla_flags}"]
    args += ["--job-env", f"XLA_PYTHON_CLIENT_MEM_FRACTION={stack.mem_fraction}"]
    for setting in stack.model_settings:
        args += ["--model-set", setting]
    for setting in HERO_OPT_SETTINGS:
        args += ["--opt-set", setting]
    args += ["--opt-state-offload", "expert_down", "--pipeline-depth", "1"]
    args += ["--seed-xla-autotune-cache", "--max-retries", "3"] if seed_autotune else ["--max-retries", "0"]
    return args


@dataclasses.dataclass(frozen=True)
class RunCheck:
    run: str
    wall_step_time: float
    lb_mean: float
    lb_layer_max: float
    drops_max: float
    bias: tuple[float, float]

    @property
    def accepted(self) -> bool:
        lo, hi = ACCEPTANCE.lb_mean
        return (
            lo <= round(self.lb_mean, 3) <= hi
            and self.lb_layer_max <= ACCEPTANCE.lb_layer_max
            and self.drops_max < ACCEPTANCE.drops_max
            and max(abs(b) for b in self.bias) <= ACCEPTANCE.bias_abs_max
        )


def run_check(name: str) -> RunCheck:
    """Wall-clock step time and routing health of the W&B run ``name``."""
    api = wandb.Api()
    runs = [r for p in WANDB_PROJECTS for r in api.runs(p, filters={"display_name": name}, order="-created_at")]
    if not runs:
        raise SystemExit(f"{name}: no W&B run")
    run = runs[0]
    lb_keys = [k for k in run.summary.keys() if k.startswith(LB_PREFIX) and k.endswith(LB_SUFFIX)]
    keys = ["global_step", "_timestamp", "train/router/capacity_overflow_rate_mean"]
    keys += ["train/router/bias_max", "train/router/bias_min", *lb_keys]
    rows = {x["global_step"]: x for x in run.scan_history(keys=keys)}
    window = [s for s in range(*ACCEPTANCE.steps) if s in rows]
    wall = sorted(s for s in rows if WALL_STEPS[0] <= s < WALL_STEPS[1])
    if not window or len(wall) < 2:
        raise SystemExit(f"{name}: {run.state}, not enough logged steps")
    return RunCheck(
        run=name,
        wall_step_time=(rows[wall[-1]]["_timestamp"] - rows[wall[0]]["_timestamp"]) / (wall[-1] - wall[0]),
        lb_mean=statistics.mean(statistics.mean(rows[s][k] for k in lb_keys) for s in window),
        lb_layer_max=max(rows[s][k] for s in window for k in lb_keys),
        drops_max=max(rows[s]["train/router/capacity_overflow_rate_mean"] for s in window),
        bias=(
            min(rows[s]["train/router/bias_min"] for s in window),
            max(rows[s]["train/router/bias_max"] for s in window),
        ),
    )


def main(argv: list[str]) -> None:
    if "--accept" in argv:
        checks = [run_check(n) for n in argv[argv.index("--accept") + 1 :]]
        for c in checks:
            verdict = "accepted" if c.accepted else "REJECTED"
            print(
                f"{c.run}: {verdict}; wall {c.wall_step_time:.3f} s/step | LB {c.lb_mean:.3f} (max layer "
                f"{c.lb_layer_max:.3f}) | drops {c.drops_max:.1e} | bias [{c.bias[0]:.3f}, {c.bias[1]:.3f}]"
            )
        if not all(c.accepted for c in checks):
            raise SystemExit(1)
        return
    if "--stack" not in argv:
        raise SystemExit(f"usage: belay_hero_mfu --stack <{'|'.join(STACKS)}> [--plan] [launch args...]")
    i = argv.index("--stack")
    name, rest = argv[i + 1], argv[:i] + argv[i + 2 :]
    plan = "--plan" in rest
    rest = [a for a in rest if a != "--plan"]
    seed = "--seed-xla-autotune-cache" in rest
    rest = [a for a in rest if a != "--seed-xla-autotune-cache"]
    args = [*stack_args(name, seed_autotune=seed), *rest]
    if plan:
        print("experiments.grug.fast_track.launch " + " ".join(args))
        return
    # --submit re-runs the launcher on the cluster from sys.argv, so it must hold the expanded launcher arguments.
    sys.argv = [sys.argv[0], *args]
    launch.main.main(args=args)


if __name__ == "__main__":
    main(sys.argv[1:])
