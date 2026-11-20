# Serving Numerical Parity Reference

This reference lists the Levanter, SkyRL/Megatron, and vLLM knobs that control numerical agreement between training and served vLLM inference (and repeatability of vLLM scoring). All knobs remain off by default.

**Summary:**

- **Levanter Snowball SFT / distillation:** use `VLLM_PARITY_SPEC` in [snowball_h100.py](https://github.com/marin-community/marin/blob/main/experiments/sft/configs/snowball_h100.py) (`moe_capacity_factor`, `router_dot_fp32` + FP32 `router_bias`, `fp32_activations`; add `router_combine_fp32` once [PR 10686](https://github.com/marin-community/marin/pull/10686) merges). Leave `torch_order` attention and `--xla_allow_excess_precision=false` off outside diagnostics.
- **SkyRL / Megatron RL:** use captured-route router replay (`moe_router_replay` + `enable_return_routed_experts`) and batch invariance (`trainer.algorithm.batch_invariant`, `VLLM_BATCH_INVARIANT=1`). Leave FP32 LM heads opt-in: paired FP32 heads cut the Snowball train–serve gap by only 0.6% ([issue 11561](https://github.com/marin-community/marin/issues/11561)).
- **Scope of evidence:** every number below measures **agreement with served vLLM** (or vLLM self-repeatability). None measures fidelity to Snowball's original TPU pretraining forward or downstream evaluation / RL reward quality.

## Knob inventory

Recommendation labels separate knobs that measurably reduce the train–serve gap from knobs that are diagnostic only or showed a negligible train–serve gain:

- **Recommended** — measurably reduces train–serve mismatch or serving repeat noise; safe to enable when parity is the goal.
- **Required with replay** — producer setting required by router replay.
- **Measured, small effect** — measured on hardware; train–serve reduction is near noise relative to cost; keep opt-in.
- **Diagnostic only** — useful for isolating rounding differences against a reference; not recommended for production training.
- **Optimization (separate)** — changes which parameters update, not step-0 forward agreement.

| Knob | Where it lives | Recommendation | What it changes | Measured effect on serve agreement | Cost / side effects | Default | Status |
|---|---|---|---|---|---|---|---|
| `SnowballConfig.moe_capacity_factor` | Levanter (`SnowballConfig`) | **Recommended** | Sets per-shard MoE dispatch capacity; set to the `expert` mesh axis size (for example `8.0`) for dropless routing. | Removing silent MoE drops raised Snowball v0.1 top-1 token agreement with vLLM from 88.6% to 99.4% ([issue 10695](https://github.com/marin-community/marin/issues/10695), [issue 10698](https://github.com/marin-community/marin/issues/10698)); fixing drops and combine cut p99 log-prob gap from 3.96 to 0.60 nats ([issue 11561](https://github.com/marin-community/marin/issues/11561)). | Larger per-shard dispatch buffers and compute on imbalanced batches. | `1.0` | main ([PR 10685](https://github.com/marin-community/marin/pull/10685)) |
| `SnowballConfig.router_dot_fp32` + `field_dtype_overrides={"router_bias": "float32"}` | Levanter (`SnowballConfig` + `TrainerConfig` / `SFTSpec`) | **Recommended** | Upcasts router operands to FP32 and keeps `router_bias` in FP32 across state, compute view, HF load, and HF export. Matches vLLM, where `GrugMoeMoE` runs `self.router` and `router_logits_dtype` in FP32 unconditionally (no vLLM knob exists). `train_lm` rejects `router_dot_fp32=True` without FP32 `router_bias`. | Part of the all-five Snowball v0.1 `lr0-r8` gate pass (`p99 <= 0.70`, `max <= 3.5` nats; [issue 10698](https://github.com/marin-community/marin/issues/10698)). Not isolated on its own. | Negligible compute/HBM (`[E]` bias per layer). Exports `router_bias` in FP32 in the resulting HF checkpoint. For `hf_to_levanter`, pass `compute_dtype=None` so conversion does not round `router_bias` to BF16. | `False` / `{}` | main ([PR 10684](https://github.com/marin-community/marin/pull/10684)) |
| `SnowballConfig.fp32_activations` | Levanter (`SnowballConfig`) | **Recommended** | Computes SiLU and sigmoid in FP32 and casts to compute dtype once (GatedNorm, attention gate, shared MLP, routed experts). | Single-op BF16 diff vs torch dropped 30–36% to 0%; full-model Levanter-vs-vLLM route agreement rose 0.8984 to 0.9123 (torch-vs-vLLM: 0.9120); per-layer route check rose 10 to 22 of 26 layers ([issue 10698](https://github.com/marin-community/marin/issues/10698)). | Rejects `moe_implementation="sonic_cute"`. Peak HBM unchanged at gate shapes (50.5 GiB on H100x8); training throughput delta unmeasured. | `False` | main ([PR 10687](https://github.com/marin-community/marin/pull/10687)) |
| `SnowballConfig.router_combine_fp32` | Levanter (`SnowballConfig`) | **Recommended** (once merged) | Keeps MoE combine weights, weighted expert outputs, per-token sum, and expert-axis reduction in FP32, casting once at the end. | Contrastive max `\|Δ log-prob\|` vs vLLM fell 4.28 to 2.18 nats (limit 3.5); top-1 disagreements fell 264 to 163 of 27,228 ([issue 10698](https://github.com/marin-community/marin/issues/10698)). | Rejects `moe_implementation` `"sonic"` and `"sonic_cute"`. FP32 combine/reduction; step-time delta unmeasured. | `False` | Open approved ([PR 10686](https://github.com/marin-community/marin/pull/10686)) |
| `SnowballConfig.attention_implementation="torch_order"` (+ HF reload preservation) | Levanter (`SnowballConfig` + `train_lm.py`) | **Diagnostic only** | Transcribes torch eager softmax/normalization order and preserves `attention_implementation` across `use_hf_model_config=True` reloads. On `main`, HF reload drops it so GPU runs use FA4. | Attention core vs torch eager drops 30.4% to 2 of 814,080 elements; `lr0-r8` passed the 8K log-prob gate (`22/26` route layers). Served vLLM uses FA3, which rounds closer to FA4 than to torch; FA4 + no-excess-precision reached 0.9572 L0 route agreement vs 0.9568 for `torch_order` ([issue 10698](https://github.com/marin-community/marin/issues/10698), [flag-arms reading](https://gist.github.com/karansag/2ac0c5798b18bfdbec9a81b7090361d2#file-diag-flag-arms-r1-reading-chukar-md)). | Materializes `S × S` FP32 attention scores (quadratic memory/time at long context; untested above 8,192 tokens). | `None` (FA4 on GPU) | Open approved ([PR 10688](https://github.com/marin-community/marin/pull/10688)) |
| `XLA_FLAGS="--xla_allow_excess_precision=false"` | Levanter worker environment | **Diagnostic only** | Prevents XLA GPU fusions from keeping excess intermediate precision. | With FA4 reached 0.9572 L0 route agreement vs vLLM, but failed candidate max (2.5058 > 2.5) and passed fewer layers (21/26 vs 22/26); combining with `torch_order` dropped route passes to 20/26 ([flag-arms reading](https://gist.github.com/karansag/2ac0c5798b18bfdbec9a81b7090361d2#file-diag-flag-arms-r1-reading-chukar-md)). | Disables some XLA fusion optimizations; training throughput delta unmeasured. | unset | External XLA flag |
| `trainer.{policy,ref}.megatron_config.moe_router_replay` | SkyRL / Megatron | **Recommended** (RL) | Replays expert assignments captured during vLLM rollout during Megatron scoring and updates. | On Hero 535B (PP48/EP4/CP4), reduced batch-order gradient sensitivity from 71.96% to 1.37% and passed the 4K RL gate (`<0.1%` probability-ratio outliers outside `[0.8, 1.2]`); serve–trainer gap still nonzero ([issue 10388](https://github.com/marin-community/marin/issues/10388)). | `+5.3%` median scoring-plus-update time at Hero PP48/EP4/CP4, plus route-tensor transfer from vLLM. | `false` | main |
| `generator.engine_init_kwargs.enable_return_routed_experts` (`--enable-return-routed-experts`) | vLLM serve / SkyRL generator | **Required with replay** | Captures and returns per-token top-k expert IDs alongside responses. | Producer required for `moe_router_replay` and route-agreement diagnostics ([issue 10388](https://github.com/marin-community/marin/issues/10388), [issue 10699](https://github.com/marin-community/marin/issues/10699)). | Route payload serialization/transport overhead; counted in the replay timing above. | `false` | main |
| `VLLM_BATCH_INVARIANT=1` (`ServeConfig.vllm_batch_invariant=True`) + optional `TORCHINDUCTOR_DETERMINISTIC=1` (`benchmark_combo_kernel: false`) | vLLM serve | **Recommended** (repeatable scoring / RL) | Enables fixed-root expert-output reduction and batch-invariant kernels in the Marin vLLM fork (`01911be3+`). | Eliminated vLLM repeat and cross-rank drift on Snowball v0.1 (`206/206` identical repeats; max summed log-prob change `0.619 -> 0` nats) and June step-42150 (`136/136` exact matches across 8 ranks and 2 fresh launches; [issue 10699](https://github.com/marin-community/marin/issues/10699)). | Scoring 256 requests: 38 s default compile, 38–44 s deterministic compile, 56–57 s `--enforce-eager`; disables NCCL symmetric memory. Default and deterministic compiles disagree by up to 0.0034 nat/tok, so compare only matching compile modes. | unset (`None`) | main |
| `trainer.algorithm.batch_invariant` | SkyRL recipe | **Recommended** (with batch-invariant RL) | Coordinates batch-invariant serving and training settings across SkyRL workers. | Used with `VLLM_BATCH_INVARIANT=1` and router replay in Hero 4K RL qualification ([issue 10388](https://github.com/marin-community/marin/issues/10388)). | Inherits vLLM's one-channel NCCL limit unless `nccl_max_channels` is set on TP1/ETP1 Megatron policies ([SkyRL guide](../dev-guide/skyrl.md#grug-training-communication)). | `false` | main |
| `generator.engine_init_kwargs.hf_overrides: {head_dtype: float32}` (`--hf-overrides '{"head_dtype": "float32"}'`) | vLLM serve (pinned Marin fork) | *Measured, small effect* (opt-in) | Runs the vLLM LM head with BF16 operands, FP32 accumulation, and FP32 logit output. | On Snowball 10/05 with router replay ([issue 11561](https://github.com/marin-community/marin/issues/11561), [PR 11772](https://github.com/marin-community/marin/pull/11772)), pairing FP32 heads on both engines cut mean train–serve log-prob gap by only **0.00011 nats (0.6%)** (95% CI `[0.00001, 0.00022]`; BF16 repeat drift was 0.00010); p99 and k3 unchanged. **Halved vLLM's own repeat gap** (`0.01287 -> 0.00692` nats/tok; finite-token binary KL `0.000186 -> 0.000098`). | ~12% lower vLLM scoring throughput (120.9k / 129.8k vs 145.5k / 139.4k tok/s) and `+0.6 GiB/GPU` peak memory. | unset (BF16) | Pinned Marin vLLM fork; SkyRL probe in draft [PR 11772](https://github.com/marin-community/marin/pull/11772) |
| `trainer.policy.megatron_config.fp32_lm_head` (and `ref`) | SkyRL / Megatron | *Measured, small effect* (opt-in) | Casts BF16 final-norm hidden states and LM-head weights to FP32 before the GEMM (TF32 off) and keeps FP32 logits into log-prob/entropy. | On Snowball 10/05 ([issue 11561](https://github.com/marin-community/marin/issues/11561), [PR 11772](https://github.com/marin-community/marin/pull/11772)), Megatron-only FP32 gave no train–serve reduction (`0.01809` vs `0.01803`/`0.01813` BF16); paired with vLLM FP32 cut mean gap only **0.6%**. Megatron repeats were already exact at BF16. | TP1 only; disables Megatron fused gradient accumulation on the LM head. No scoring slowdown observed on Snowball 10/05 (`5,144` vs `4,764`/`4,886` tok/s). | `false` | Draft [PR 11671](https://github.com/marin-community/marin/pull/11671) + stacked [PR 11771](https://github.com/marin-community/marin/pull/11771) |
| `FrozenParameterAdaptorConfig(frozen_field_names=("router_bias",))` | Levanter (`TrainLmConfig.adapter`) | Optimization (separate) | Keeps `router_bias` unchanged across optimizer updates while exporting the full model. | Does not change step-0 forward agreement; used in identity SFT runs ([issue 10695](https://github.com/marin-community/marin/issues/10695)) and the zero-LR cluster parity harness ([PR 11061](https://github.com/marin-community/marin/pull/11061)). | Native checkpoints omit frozen leaves; resume requires the same pinned base checkpoint. | `NoAdaptorConfig()` | main |

## Copyable configurations

### Levanter Snowball SFT, context distillation, and self-distillation

[experiments/sft/configs/snowball_h100.py](https://github.com/marin-community/marin/blob/main/experiments/sft/configs/snowball_h100.py) exports `VLLM_PARITY_SPEC`, a named `SFTSpec` preset that turns on the parity knobs currently on `main` without changing the default `SPEC`. Copy or adapt it with `dataclasses.replace`:

```python
from dataclasses import replace

from levanter.models.snowball import SnowballConfig

from experiments.sft.configs.snowball_h100 import SPEC
from experiments.sft.launcher import HFModel

assert isinstance(SPEC.model, HFModel)
VLLM_PARITY_SPEC = replace(
    SPEC,
    name="checkpoints/snowball-openthoughts-agent-sft-vllm-parity",
    model=replace(
        SPEC.model,
        model=SnowballConfig(
            moe_capacity_factor=8.0,  # match the mesh's expert-axis size
            router_dot_fp32=True,
            fp32_activations=True,
        ),
    ),
    field_dtype_overrides={"router_bias": "float32"},
)
```

Once [PR 10686](https://github.com/marin-community/marin/pull/10686) merges, add `router_combine_fp32=True` inside `SnowballConfig(...)`. Do not add `attention_implementation="torch_order"` unless running a short-context diagnostic against torch eager.

For other Levanter entry points:

- **Converted checkpoints (`ConvertedCheckpointModel`)**, used by [curriculum SFT and self-distillation](https://github.com/marin-community/marin/blob/main/experiments/post_training/curriculum_sft/trial.py): `hf_to_levanter` defaults to `compute_dtype="bfloat16"`, which rounds `router_bias` to BF16 during conversion before training starts. Pass `compute_dtype=None` to `hf_to_levanter` to preserve each tensor's stored HF dtype, replace the Snowball runtime flags on `conversion.model`, and set `field_dtype_overrides={"router_bias": "float32"}` on `SFTSpec`.
- **Direct `TrainLmConfig` launches**: set the same `SnowballConfig` fields on `config.model` and `field_dtype_overrides={"router_bias": jnp.dtype("float32")}` on `config.trainer` (in YAML: `field_dtype_overrides: {router_bias: float32}`).

### SkyRL / Megatron RL and on-policy distillation

For RL and on-policy KL distillation on Megatron + vLLM, enable captured-route replay and batch invariance in the SkyRL recipe overlay:

```yaml
trainer:
  algorithm:
    batch_invariant: true
  policy:
    megatron_config:
      moe_router_replay: true
  ref:
    megatron_config:
      moe_router_replay: true
generator:
  engine_init_kwargs:
    enable_return_routed_experts: true
```

FP32 LM heads (`trainer.policy.megatron_config.fp32_lm_head: true` and `generator.engine_init_kwargs.hf_overrides: {head_dtype: float32}`) are intentionally omitted above: on Snowball 10/05 they reduced the train–serve gap by only 0.6% while reducing vLLM scoring throughput by ~12% ([issue 11561](https://github.com/marin-community/marin/issues/11561)).

### Repeatable vLLM scoring

To remove vLLM's own repeat and cross-rank noise when scoring fixed checkpoints (for example evaluation gates or mismatch probes), set `vllm_batch_invariant=True` on `ServeConfig` (which sets `VLLM_BATCH_INVARIANT=1`). When comparing scores across separate launches, keep the compile mode identical across serves (default compile vs `TORCHINDUCTOR_DETERMINISTIC=1` with `benchmark_combo_kernel: false` disagree by up to 0.0034 nat/tok; [issue 10699](https://github.com/marin-community/marin/issues/10699)).

## When parity matters by workflow

- **Direct log-prob comparison (on-policy KL distillation, KL/log-prob self-distillation, RL importance ratios):** training directly compares learner and vLLM log-probabilities. On-policy KL distillation and RL in Marin currently run through SkyRL (Megatron + vLLM), where router replay and batch invariance are the primary levers.
- **Fixed-text SFT, fixed-text context distillation, and verified-sample self-distillation:** recipes such as [snowball_h100.py](https://github.com/marin-community/marin/blob/main/experiments/sft/configs/snowball_h100.py), the identity SFT runs in [issue 10695](https://github.com/marin-community/marin/issues/10695), and [curriculum self-distillation](https://github.com/marin-community/marin/blob/main/experiments/post_training/curriculum_sft/self_distill.py) train on fixed chat text via `sft_step` / `train_lm`. They do not compare Levanter and vLLM log-probabilities directly, but their loss and gradients come from Levanter while the exported model is served with vLLM. No run has shown whether that gap changes SFT outcomes. Default `moe_capacity_factor=1.0` can still silently drop routed assignments in any expert-parallel run, SFT included.

## Evidence boundaries and limits

- **Measured serving agreement vs training fidelity:** agreement with Snowball's original TPU pretraining forward was not measured. The Levanter parity settings move Levanter toward the served vLLM computation; they do not establish which forward is closer to pretraining.
- **Downstream quality and RL reward:** no run measured the effect of the Levanter parity flags or FP32 LM heads on downstream benchmark scores or RL reward.
- **Levanter gate coverage:** the `lr0-r8` gate ([final run](https://gist.github.com/karansag/2ac0c5798b18bfdbec9a81b7090361d2#file-lr0-r8-reading-chukar-md)) ran once on Snowball v0.1, 32 H100s, 8,192 tokens, against vLLM with FA3, with all five settings enabled together (`22/26` route layers passed; 4 accepted deviations). Other checkpoints, TPUs, other vLLM versions or attention backends, and longer sequences (32,768 or 262,144 tokens) were not measured. Separate diagnostics checked combine, activations, and attention individually, but single-setting effects on training or evals were not isolated.
- **FP32 LM head probe coverage:** the Snowball 10/05 probe ([PR 11772](https://github.com/marin-community/marin/pull/11772), [issue 11561](https://github.com/marin-community/marin/issues/11561)) ran scoring-only (no optimizer steps) on 512 responses (480,150 loss tokens) at Megatron TP1/PP2/CP1/EP8 vs vLLM DP8/EP8 with router replay on. Production Snowball RL uses CP5, and Mochi (Hero) was not tested.
- **Backend incompatibilities:** `fp32_activations=True` rejects `moe_implementation="sonic_cute"`; `router_combine_fp32=True` ([PR 10686](https://github.com/marin-community/marin/pull/10686)) rejects both `"sonic"` and `"sonic_cute"`; Megatron `fp32_lm_head` ([PR 11671](https://github.com/marin-community/marin/pull/11671) / [PR 11771](https://github.com/marin-community/marin/pull/11771)) supports TP1 only.
- **Automated training-forward check and prior scripts:** [PR 11061](https://github.com/marin-community/marin/pull/11061) (draft) adds a position-level Levanter-vs-vLLM cluster test (not yet run on hardware). The identity SFT runs in [issue 10695](https://github.com/marin-community/marin/issues/10695) used custom launch scripts pinned in their [reproduction guide](https://github.com/marin-community/marin-logbooks/blob/58e821a0b55328a48446b789ae9b7702b64b95a0/logbooks/10695-identity-sft/REPRODUCE.md).
