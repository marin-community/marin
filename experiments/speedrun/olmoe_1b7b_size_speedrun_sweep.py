# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
OLMoE 1B/7B-style speedrun sweep across model sizes.

This is a lightweight analogue of
`experiments/speedrun/hackable_transformer_starter/hackable_transformer_attn_sink.py`,
but uses our Mixtral/OLMoE-style MoE model (`experiments/speedrun/custom_mixtral.py`).

The sweep varies model size and learning-rate multipliers (base-2 logspace multipliers, like the
hackable-transformer starter). Batch size / steps / optimizer base LR are functions of the size preset.
We keep the routed-expert granularity ratio fixed: `num_experts_per_tok / n_routed_experts`.

How to run (TPU v5p-8):
  1) Set env vars (WANDB_API_KEY, HF_TOKEN, etc.) as in the tutorial:
     https://marin.readthedocs.io/en/latest/tutorials/submitting-speedrun/
  2) From repo root:
       python marin/run/ray_run.py -- \
         python -m experiments.speedrun.olmoe_1b7b_size_speedrun_sweep
"""

# nodryrun

from __future__ import annotations

import argparse
import dataclasses
import logging
import os

import numpy as np
import wandb
from fray.cluster import ResourceConfig
from levanter.utils.activation import ActivationFunctionEnum
from levanter.optim import MuonConfig

from experiments.llama import llama3_tokenizer_vocab_size
from experiments.simple_train_config import SimpleTrainConfig
from experiments.speedrun.custom_mixtral import MixtralConfig
from experiments.speedrun.olmoe_1b7b_nemotron_40b import (
    nemotron_cc_mixture,
    nemotron_dclm_fineweb_mixture,
    nemotron_only_speedrun,
)
from experiments.speedrun.prebuilt_caches import fineweb_edu_subcache_10B
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main
from marin.speedrun.speedrun import Author, SpeedrunConfig

logger = logging.getLogger("ray")

AUTHOR = Author(name="TODO", affiliation="TODO", url="")  # TODO: update me

OLMOE_1B7B_REFERENCE_CHECKPOINT = "allenai/OLMoE-1B-7B-0125"


def _dataset_tag(dataset: str) -> str:
    # Keep tags short to reduce the odds of W&B name truncation collisions.
    tags = {
        "nemotron_cc": "nemo",
        "fineweb_edu_10b": "fw10",
        "nemotron_dclm_fineweb_10b": "ncfw",
    }
    try:
        return tags[dataset]
    except KeyError as e:
        raise ValueError(f"Unknown dataset preset: {dataset}") from e


def _get_num_train_steps(param_count: int, batch_size: int, seq_len: int, tpp: int = 20) -> int:
    total_tokens = param_count * tpp
    return max(1, total_tokens // (batch_size * seq_len))


def _active_params_equivalent(cfg: MixtralConfig, vocab_size: int) -> int:
    """Approximate \"active\" params/token for MoE sizing.

    MixtralConfig.total_trainable_params() counts *all* experts. For speedrun-style scaling, we
    instead estimate the dense-equivalent parameter count by counting only the routed experts
    used per token (plus always-on attention/router).
    """

    token_embedding = vocab_size * cfg.hidden_dim
    head = 0 if cfg.tie_word_embeddings else token_embedding

    head_size = cfg.hidden_dim // cfg.num_heads
    q_proj = cfg.hidden_dim * head_size * cfg.num_heads
    kv_proj = 2 * cfg.hidden_dim * head_size * cfg.num_kv_heads
    o_proj = head_size * cfg.num_heads * cfg.hidden_dim
    attn = q_proj + kv_proj + o_proj

    router = cfg.hidden_dim * cfg.n_routed_experts
    moe_active = router + 3 * cfg.num_experts_per_tok * cfg.hidden_dim * cfg.intermediate_dim

    transformer_layer = attn + moe_active + 2 * cfg.hidden_dim  # + 2 rmsnorm
    transformer = cfg.num_layers * transformer_layer + cfg.hidden_dim  # + final rmsnorm
    return int(transformer + token_embedding + head)


def _size_presets(seq_len: int) -> dict[str, MixtralConfig]:
    # Keep expert granularity fixed: 8/64 = 1/8
    return {
        "olmoe_s": MixtralConfig(
            seq_len=seq_len,
            hidden_dim=768,
            intermediate_dim=384,
            num_layers=8,
            num_heads=6,
            num_kv_heads=3,
            n_routed_experts=8,
            num_experts_per_tok=1,
            layer_norm_epsilon=1e-5,
            gradient_checkpointing=True,
            scan_layers=True,
            use_gmm=True,
            cross_entropy_block_size=32000,
            flash_attention_block_size=None,
            reference_checkpoint=OLMOE_1B7B_REFERENCE_CHECKPOINT,
            tokenizer=OLMOE_1B7B_REFERENCE_CHECKPOINT,
        ),
        "olmoe_m": MixtralConfig(
            seq_len=seq_len,
            hidden_dim=1024,
            intermediate_dim=512,
            num_layers=12,
            num_heads=8,
            num_kv_heads=4,
            n_routed_experts=16,
            num_experts_per_tok=2,
            layer_norm_epsilon=1e-5,
            gradient_checkpointing=True,
            scan_layers=True,
            use_gmm=True,
            cross_entropy_block_size=32000,
            flash_attention_block_size=None,
            reference_checkpoint=OLMOE_1B7B_REFERENCE_CHECKPOINT,
            tokenizer=OLMOE_1B7B_REFERENCE_CHECKPOINT,
        ),
        "olmoe_l": MixtralConfig(
            seq_len=seq_len,
            hidden_dim=1536,
            intermediate_dim=768,
            num_layers=16,
            num_heads=12,
            num_kv_heads=6,
            n_routed_experts=32,
            num_experts_per_tok=4,
            layer_norm_epsilon=1e-5,
            gradient_checkpointing=True,
            scan_layers=True,
            use_gmm=True,
            cross_entropy_block_size=32000,
            flash_attention_block_size=None,
            reference_checkpoint=OLMOE_1B7B_REFERENCE_CHECKPOINT,
            tokenizer=OLMOE_1B7B_REFERENCE_CHECKPOINT,
        ),
        # Roughly matches OLMoE 1B-7B active params.
        "olmoe_1b7b": MixtralConfig(
            seq_len=seq_len,
            hidden_dim=2048,
            intermediate_dim=1024,
            num_layers=16,
            num_heads=16,
            num_kv_heads=8,
            n_routed_experts=64,
            num_experts_per_tok=8,
            layer_norm_epsilon=1e-5,
            gradient_checkpointing=True,
            scan_layers=True,
            use_gmm=True,
            cross_entropy_block_size=32000,
            flash_attention_block_size=None,
            reference_checkpoint=OLMOE_1B7B_REFERENCE_CHECKPOINT,
            tokenizer=OLMOE_1B7B_REFERENCE_CHECKPOINT,
        ),
    }


def _muon_presets() -> dict[str, MuonConfig]:
    # Tuned for short speedrun-like tests; these values were used in prior W&B sweeps.
    return {
        "olmoe_s": MuonConfig(
            learning_rate=0.01,
            adam_lr=0.003,
            weight_decay=0.1,
            min_lr_ratio=0,
            warmup=0,
            momentum=0.98,
            beta1=0.8,
            beta2=0.98,
            epsilon=1e-25,
            muon_epsilon=1e-5,
            max_grad_norm=1,
            lr_schedule="linear",
            decay=0.8,
        ),
        "olmoe_m": MuonConfig(
            learning_rate=0.008,
            adam_lr=0.0024,
            weight_decay=0.1,
            min_lr_ratio=0,
            warmup=0,
            momentum=0.98,
            beta1=0.8,
            beta2=0.98,
            epsilon=1e-25,
            muon_epsilon=1e-5,
            max_grad_norm=1,
            lr_schedule="linear",
            decay=1,
        ),
        "olmoe_l": MuonConfig(
            learning_rate=0.008,
            adam_lr=0.0024,
            weight_decay=0.1,
            min_lr_ratio=0,
            warmup=0,
            momentum=0.98,
            beta1=0.8,
            beta2=0.98,
            epsilon=1e-25,
            muon_epsilon=1e-5,
            max_grad_norm=1,
            lr_schedule="linear",
            decay=1,
        ),
        "olmoe_1b7b": MuonConfig(
            learning_rate=0.004,
            adam_lr=0.0012,
            weight_decay=0.1,
            min_lr_ratio=0,
            warmup=0,
            momentum=0.98,
            beta1=0.8,
            beta2=0.98,
            epsilon=1e-15,
            muon_epsilon=1e-5,
            max_grad_norm=2,
            lr_schedule="linear",
            decay=1,
        ),
    }


def _resource_presets(tpu_type: str) -> dict[str, ResourceConfig]:
    resources = ResourceConfig.with_tpu(tpu_type)
    return {k: resources for k in _size_presets(seq_len=2048).keys()}


def _batch_sizes() -> dict[str, int]:
    # Conservative defaults: OLMoE total parameters are large (optimizer state memory).
    return {"olmoe_s": 128, "olmoe_m": 128, "olmoe_l": 64, "olmoe_1b7b": 64}


def _lr_multipliers(num: int = 5, start_exp: float = -2.0, stop_exp: float = 2.0) -> list[float]:
    # Base-2 logspace, exclude 1x (baseline is implicit).
    vals = np.logspace(start=start_exp, stop=stop_exp, num=num, base=2.0)
    filtered = [float(v) for v in vals if not np.isclose(v, 1.0)]
    seen = set()
    out: list[float] = []
    for v in filtered:
        key = f"{v:.8f}"
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def _format_multiplier_label(mult: float) -> str:
    exp = float(np.log2(mult))
    exp_round = round(exp)
    if np.isclose(exp, exp_round):
        # Avoid '-' in run names: experiments.defaults.default_train truncation logic uses '-' as a delimiter
        # and can drop the LR tag (e.g. "e-2") under W&B's 64-char limit, collapsing distinct runs.
        if exp_round < 0:
            return f"em{abs(exp_round)}"
        return f"e{exp_round}"

    s = f"{mult:.6g}"
    s = s.rstrip("0").rstrip(".") if "." in s else s
    return f"x{s.replace('.', '_')}"


def _size_tag(size: str) -> str:
    tags = {
        "olmoe_s": "s",
        "olmoe_m": "m",
        "olmoe_l": "l",
        "olmoe_1b7b": "1b7b",
    }
    try:
        return tags[size]
    except KeyError as e:
        raise ValueError(f"Unknown size preset: {size}") from e


def build_run(
    size: str,
    *,
    dataset: str,
    tokenized_dataset: object,
    seq_len: int,
    tpu_type: str,
    lr_multiplier: float | None = None,
    global_batch_size: int | None = None,
    steps_per_eval: int | None = 500,
    steps_per_task_eval: int | None = None,
    bilinear_mlp: bool = False,
    use_qk_norm: bool = False,
    router_topk_then_softmax: bool = False,
    router_fp32: bool = False,
    alf_lb_loss_scale: float = 0.0,
    dense_first_n_layers: int = 0,
    run_suffix: str | None = None,
) -> tuple[str, SpeedrunConfig]:
    sizes = _size_presets(seq_len=seq_len)
    if size not in sizes:
        raise ValueError(f"Unknown size: {size}")
    model_cfg = sizes[size]
    if bilinear_mlp:
        model_cfg = dataclasses.replace(model_cfg, activation_function=ActivationFunctionEnum.linear)

    if use_qk_norm or router_topk_then_softmax or router_fp32 or alf_lb_loss_scale > 0 or dense_first_n_layers > 0:
        model_cfg = dataclasses.replace(
            model_cfg,
            use_qk_norm=use_qk_norm,
            router_topk_then_softmax=router_topk_then_softmax,
            router_fp32=router_fp32,
            alf_lb_loss_scale=alf_lb_loss_scale,
            dense_first_n_layers=dense_first_n_layers,
        )

    batch = global_batch_size if global_batch_size is not None else _batch_sizes()[size]

    effective_params = _active_params_equivalent(model_cfg, llama3_tokenizer_vocab_size)
    steps = _get_num_train_steps(effective_params, batch, seq_len, tpp=20)

    muon = _muon_presets()[size]
    if lr_multiplier is not None:
        muon = dataclasses.replace(
            muon,
            learning_rate=muon.learning_rate * lr_multiplier,
            adam_lr=muon.adam_lr * lr_multiplier,
        )

    resources = _resource_presets(tpu_type)[size]
    train = SimpleTrainConfig(
        resources,
        train_seq_len=seq_len,
        train_batch_size=batch,
        num_train_steps=steps,
        learning_rate=muon.learning_rate,
        optimizer_config=muon,
        steps_per_eval=steps_per_eval,
        steps_per_hf_export=-1,  # disable HF exports (we can still eval from Levanter checkpoints)
        steps_per_task_eval=steps_per_task_eval,
    )

    # Keep run names short enough that `experiments.defaults.default_train` won't truncate away the LR multiplier.
    # Truncation can cause multiple distinct runs to share the same W&B name prefix (and therefore collide in
    # executor step naming / status tracking).
    lr_tag = _format_multiplier_label(lr_multiplier) if lr_multiplier is not None else ""
    bilinear_tag = "bi" if bilinear_mlp else ""
    moe_stab_enabled = use_qk_norm or router_topk_then_softmax or alf_lb_loss_scale > 0 or dense_first_n_layers > 0
    moe_stab_tag = "ms" if moe_stab_enabled else ""
    router_fp32_tag = "r32" if router_fp32 else ""
    params_m = max(1, round(effective_params / 1_000_000))
    run_name = (
        f"osz{_dataset_tag(dataset)}{_size_tag(size)}p{params_m}q{seq_len}b{batch}"
        f"{lr_tag}{bilinear_tag}{moe_stab_tag}{router_fp32_tag}"
    )
    if run_suffix:
        # Use '-' so experiments.defaults.default_train can preserve the suffix if it truncates the name to fit W&B.
        run_name = f"{run_name}-{run_suffix}"
    activation_desc = "bilinear" if bilinear_mlp else "silu"
    desc = (
        f"OLMoE-style MoE ({size}); Splash attention; activation={activation_desc}; "
        f"LR sweep multiplier={lr_multiplier or 1.0:g}"
    )
    cfg = SpeedrunConfig(
        author=AUTHOR,
        description=desc,
        model_config=model_cfg,
        train_config=train,
        tokenized_dataset=tokenized_dataset,
    )
    return run_name, cfg


def _canonicalize_wandb_base_name(
    *,
    raw_name: str,
    expected_base_names: set[str],
) -> str | None:
    """Map historical run naming schemes to the current sweep's base-name format.

    We want a stable key that matches `build_run(..., run_suffix=None)` so that we can resume older
    runs even if their W&B display name (or `trainer.tracker.name`) uses a previous format.
    """
    candidate = raw_name.split("-", 1)[0]
    if candidate in expected_base_names:
        return candidate

    # Older scheme (examples):
    #   osz_ncfw10b_mp338m_s2048_b64_em
    #   osz_ncfw10b_1b7bp1534m_s2048_b64_e1
    # The sweep key we want is:
    #   osz{dataset_tag}{size_tag}p{params_m}q{seq_len}b{batch}{lr_tag}{...}
    if not candidate.startswith("osz_"):
        return None

    # Parse the old-style token stream. We only use it to recover (dataset,size,params,lr),
    # and then match against `expected_base_names` to avoid accidental collisions.
    parts = candidate.split("_")
    if len(parts) < 5:
        return None

    # parts[0] == "osz"
    dataset_token = parts[1]
    size_token = parts[2]
    seq_token = parts[3]
    batch_token = parts[4]
    lr_token = parts[5] if len(parts) > 5 else ""

    dataset_map = {
        "nemotron_cc": "nemo",
        "fineweb_edu_10b": "fw10",
        # historical
        "ncfw10b": "ncfw",
        "nemotron_dclm_fineweb_10b": "ncfw",
    }
    dataset_tag = dataset_map.get(dataset_token)
    if not dataset_tag:
        return None

    # size_token encodes both the size family and the approx active params in millions.
    size_tag = None
    params_m = None
    if size_token.startswith("sp"):
        size_tag = "s"
        params_m = int(size_token.removeprefix("sp").removesuffix("m"))
    elif size_token.startswith("mp"):
        size_tag = "m"
        params_m = int(size_token.removeprefix("mp").removesuffix("m"))
    elif size_token.startswith("lp"):
        size_tag = "l"
        params_m = int(size_token.removeprefix("lp").removesuffix("m"))
    elif size_token.startswith("1b7bp") and size_token.endswith("m"):
        size_tag = "1b7b"
        params_m = int(size_token.removeprefix("1b7bp").removesuffix("m"))

    if size_tag is None or params_m is None:
        return None

    # seq_token like s2048; batch_token like b64 (may have been truncated in some older runs)
    seq = None
    if seq_token.startswith("s"):
        try:
            seq = int(seq_token.removeprefix("s"))
        except ValueError:
            seq = None
    batch = None
    if batch_token.startswith("b"):
        try:
            batch = int(batch_token.removeprefix("b"))
        except ValueError:
            batch = None

    # Normalize LR tag: older runs sometimes used "em" to mean 2^-2 (=0.25).
    lr_tag = lr_token
    if lr_tag == "em":
        lr_tag = "em2"

    # Build a minimal candidate and see if it matches any expected base name exactly.
    # If seq/batch were truncated, we fall back to matching any expected base name that shares
    # (dataset_tag, size_tag, params_m, lr_tag) and differs only in q/b fields.
    if seq is not None and batch is not None:
        guess = f"osz{dataset_tag}{size_tag}p{params_m}q{seq}b{batch}{lr_tag}"
        if guess in expected_base_names:
            return guess

    prefix = f"osz{dataset_tag}{size_tag}p{params_m}"
    suffix = lr_tag
    for exp in expected_base_names:
        if exp.startswith(prefix) and exp.endswith(suffix):
            return exp

    return None


def _has_checkpoint_metadata_gcs(run_id: str) -> bool:
    """Return True if the run's checkpoint directory contains any metadata.json.

    This lets us prefer resume targets that can actually load a checkpoint.
    """
    try:
        import fsspec

        ckpt_root = f"gs://marin-us-central1/checkpoints/speedrun/{run_id}/checkpoints"
        fs, _, (path,) = fsspec.get_fs_token_paths(ckpt_root)
        if not fs.exists(path):
            return False
        return bool(fs.glob(path + "/**/metadata.json"))
    except Exception:
        # Best-effort: if GCS isn't available for some reason, fall back to W&B `_step` only.
        return False


def _load_resume_run_ids_by_base_name(*, wandb_group: str, expected_base_names: set[str]) -> dict[str, str]:
    """Return a mapping {base_run_name: trainer.id} for the given W&B group.

    When multiple runs map to the same base name, prefer runs with checkpoint metadata, then choose the
    largest logged `_step` among those.
    """
    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT")
    if not entity or not project:
        raise RuntimeError("WANDB_ENTITY and WANDB_PROJECT must be set when using --resume-from-wandb.")

    api = wandb.Api(timeout=120)
    runs = list(api.runs(f"{entity}/{project}", filters={"group": wandb_group}, per_page=500))
    if not runs:
        raise RuntimeError(f"No W&B runs found for group={wandb_group!r} in {entity}/{project}.")

    def _get_config_value(config: dict, dotted_key: str):
        cur = config
        for part in dotted_key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur

    def _unwrap_wandb_value(value):
        # W&B sometimes stores config values as {"value": ...}. Normalize to the underlying value.
        if isinstance(value, dict) and "value" in value and len(value) <= 2:
            return value.get("value")
        return value

    best: dict[str, tuple[int, str]] = {}
    for run in runs:
        cfg = run.config or {}
        trainer = cfg.get("trainer") or {}
        run_id = cfg.get("trainer.id") or trainer.get("id") or getattr(run, "id", None)
        if not run_id:
            continue

        # Base key must match the value returned by build_run(..., run_suffix=None), i.e. before appending any
        # -<suffix>. We cannot reliably derive this from `trainer.id` because:
        # - older runs used `osz_...` ids (which can be parsed)
        # - newer runs use hashed/truncated ids that don't match the base-name scheme
        # So prefer the *configured* tracker name (stored in W&B config), which remains stable even if the run is
        # manually renamed in the UI.
        raw_base = None
        for key in (
            "trainer.tracker.name",
            "trainer.tracker.run_name",
            "trainer.tracker.wandb_name",
            "tracker.name",
        ):
            raw_base = _unwrap_wandb_value(_get_config_value(cfg, key))
            if raw_base:
                break
        if not raw_base:
            raw_base = getattr(run, "name", None)

        base_name = None
        if raw_base:
            base_name = _canonicalize_wandb_base_name(raw_name=str(raw_base), expected_base_names=expected_base_names)
        if not base_name:
            # Fallback for older runs: parse the trainer id (output directory) format.
            base_name = _canonicalize_wandb_base_name(raw_name=str(run_id), expected_base_names=expected_base_names)
        if not base_name:
            continue

        step = int(run.summary.get("_step") or 0)
        has_ckpt = _has_checkpoint_metadata_gcs(run_id)

        # Encode the preference (checkpoint presence first, then step) into a sortable score.
        # Any checkpoint beats any non-checkpoint, even if the latter has a larger _step due to buffering.
        score = (1_000_000_000 if has_ckpt else 0) + step

        if base_name not in best or score > best[base_name][0]:
            best[base_name] = (score, run_id)

    return {base: run_id for base, (_step, run_id) in best.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="OLMoE 1B/7B-style size sweep (speedrun).")
    parser.add_argument(
        "--dataset",
        default="nemotron_cc",
        choices=["nemotron_cc", "fineweb_edu_10b", "nemotron_dclm_fineweb_10b"],
        help="Tokenized dataset preset to use for training.",
    )
    parser.add_argument("--tpu-type", default="v5p-8")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--sizes", nargs="*", default=["olmoe_s", "olmoe_m", "olmoe_l", "olmoe_1b7b"])
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="Override the preset batch size for all sizes in the sweep.",
    )
    parser.add_argument(
        "--use-default-validation",
        action="store_true",
        help=(
            "Enable Levanter-native validation losses during training by adding the default validation sets "
            "(Paloma + uncheatable) to the data config. This is independent of --eval-suite "
            "(lm-eval-harness benchmarks)."
        ),
    )
    parser.add_argument(
        "--steps-per-eval",
        type=int,
        default=500,
        help=(
            "How often to run Levanter-native validation losses during training (TrainerConfig.steps_per_eval). "
            "Defaults to 500 to ensure at least one eval in short speedruns."
        ),
    )
    parser.add_argument(
        "--steps-per-task-eval",
        type=int,
        default=500,
        help=(
            "How often to run eval-harness tasks during training when --eval-suite-mode is during_train/both. "
            "Defaults to 500 to ensure at least one during-train eval in short speedruns."
        ),
    )
    parser.add_argument(
        "--bilinear-mlp",
        action="store_true",
        help=(
            "Use bilinear expert MLPs by setting activation_function=linear "
            "(SwiGLU -> (W1 x) * (W3 x)); HF export is not supported for this variant."
        ),
    )
    parser.add_argument(
        "--use-qk-norm",
        action="store_true",
        help="Enable QK normalization (RMSNorm) inside attention for q and k vectors.",
    )
    parser.add_argument(
        "--router-topk-then-softmax",
        action="store_true",
        help="Route by selecting top-k logits first, then softmax over the selected experts only.",
    )
    parser.add_argument(
        "--router-fp32",
        action="store_true",
        help="Compute router/gating math (logits/top-k/softmax/aux terms) in fp32 for MoE stability.",
    )
    parser.add_argument(
        "--alf-lb-loss-scale",
        type=float,
        default=0.0,
        help=("Enable DeepSeek-style auxiliary-free load balancing (bias-only) with this loss scale; " "0.0 disables."),
    )
    parser.add_argument(
        "--dense-first-n-layers",
        type=int,
        default=0,
        help="Force dense (single-expert) routing for the first N transformer layers; 0 disables.",
    )
    parser.add_argument(
        "--run-suffix",
        type=str,
        default=None,
        help=(
            "Optional suffix appended to each run name (recommended for forcing unique reruns). "
            "Tip: use `t$(date +%%Y%%m%%d_%%H%%M%%S)`."
        ),
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="Optional W&B group for all runs in the sweep (recommended for comparisons).",
    )
    parser.add_argument(
        "--resume-from-wandb",
        action="store_true",
        help=(
            "Resume previously-started runs in --wandb-group by mapping each run name prefix back to the original "
            "trainer.id/output directory. This lets you restart the sweep without reusing the exact --run-suffix."
        ),
    )
    # Executor controls (so this script can be run under ray_run without draccus CLI conflicts).
    parser.add_argument("--prefix", default=os.getenv("MARIN_PREFIX"))
    parser.add_argument("--executor-info-base-path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-run-failed", action="store_true")
    parser.add_argument("--run-only", nargs="*", default=None)
    parser.add_argument(
        "--eval-suite",
        choices=("none", "core", "core_plus_mmlu", "core_plus_leaderboard"),
        default="none",
        help="Eval-harness suite name (applies to every sweep run).",
    )
    parser.add_argument(
        "--eval-suite-mode",
        choices=("post_train", "during_train", "both"),
        default="post_train",
        help="When to run eval-harness: post_train (default), during_train, or both.",
    )
    parser.add_argument(
        "--eval-tpu-type",
        default=None,
        help="Optional TPU type for post-train eval steps (default: same as --tpu-type).",
    )
    parser.add_argument(
        "--max-eval-instances",
        type=int,
        default=None,
        help="Optional cap on max eval instances per task (post-train eval only).",
    )
    args = parser.parse_args()

    dataset_presets = {
        "nemotron_cc": nemotron_cc_mixture,
        "fineweb_edu_10b": fineweb_edu_subcache_10B,
        "nemotron_dclm_fineweb_10b": nemotron_dclm_fineweb_mixture,
    }

    expected_base_names: set[str] = set()
    for size in args.sizes:
        for mult in _lr_multipliers():
            base_name, _ = build_run(
                size,
                dataset=args.dataset,
                tokenized_dataset=dataset_presets[args.dataset],
                seq_len=args.seq_len,
                tpu_type=args.tpu_type,
                lr_multiplier=mult,
                global_batch_size=args.global_batch_size,
                steps_per_eval=args.steps_per_eval,
                steps_per_task_eval=args.steps_per_task_eval,
                bilinear_mlp=args.bilinear_mlp,
                use_qk_norm=args.use_qk_norm,
                router_topk_then_softmax=args.router_topk_then_softmax,
                router_fp32=args.router_fp32,
                alf_lb_loss_scale=args.alf_lb_loss_scale,
                dense_first_n_layers=args.dense_first_n_layers,
                run_suffix=None,
            )
            expected_base_names.add(base_name)

    resume_run_ids: dict[str, str] = {}
    if args.resume_from_wandb:
        if not args.wandb_group:
            raise ValueError("--resume-from-wandb requires --wandb-group.")
        resume_run_ids = _load_resume_run_ids_by_base_name(
            wandb_group=args.wandb_group,
            expected_base_names=expected_base_names,
        )
        logger.info("Loaded %d resume targets from W&B group=%s", len(resume_run_ids), args.wandb_group)

    steps: list[ExecutorStep] = []
    for size in args.sizes:
        for mult in _lr_multipliers():
            base_name, cfg = build_run(
                size,
                dataset=args.dataset,
                tokenized_dataset=dataset_presets[args.dataset],
                seq_len=args.seq_len,
                tpu_type=args.tpu_type,
                lr_multiplier=mult,
                global_batch_size=args.global_batch_size,
                steps_per_eval=args.steps_per_eval,
                steps_per_task_eval=args.steps_per_task_eval,
                bilinear_mlp=args.bilinear_mlp,
                use_qk_norm=args.use_qk_norm,
                router_topk_then_softmax=args.router_topk_then_softmax,
                router_fp32=args.router_fp32,
                alf_lb_loss_scale=args.alf_lb_loss_scale,
                dense_first_n_layers=args.dense_first_n_layers,
                run_suffix=None,
            )

            override_output_path = None
            name = base_name
            if args.resume_from_wandb and base_name in resume_run_ids:
                override_output_path = f"checkpoints/speedrun/{resume_run_ids[base_name]}"
            elif args.run_suffix:
                name = f"{base_name}-{args.run_suffix}"

            steps.extend(
                nemotron_only_speedrun(
                    name,
                    cfg,
                    use_default_validation=args.use_default_validation,
                    eval_suite=args.eval_suite,
                    eval_suite_mode=args.eval_suite_mode,
                    eval_tpu_type=args.eval_tpu_type,
                    max_eval_instances=args.max_eval_instances,
                    wandb_group=args.wandb_group,
                    override_output_path=override_output_path,
                )
            )

    executor_cfg = ExecutorMainConfig(
        prefix=args.prefix,
        executor_info_base_path=args.executor_info_base_path,
        dry_run=args.dry_run,
        force_run_failed=args.force_run_failed,
        run_only=args.run_only,
    )
    executor_main.__wrapped__(executor_cfg, steps=steps, description=f"OLMoE 1B/7B-style size sweep on {args.tpu_type}")


if __name__ == "__main__":
    main()
