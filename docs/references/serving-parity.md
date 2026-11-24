# Serving Numerical Parity

Levanter, SkyRL/Megatron, and vLLM expose opt-in knobs that reduce numerical disagreement between training and served vLLM inference, and remove vLLM repeat drift. Every knob below is off by default.

All measurements in this page compare training against served vLLM (or vLLM against itself across repeats). They do not measure fidelity to Snowball's original TPU pretraining forward, downstream eval accuracy, or RL reward.

## Recommended presets

### Levanter Snowball SFT and distillation

Use `VLLM_PARITY_SPEC` in [experiments/sft/configs/snowball_h100.py](https://github.com/marin-community/marin/blob/main/experiments/sft/configs/snowball_h100.py), or adapt it with `dataclasses.replace`:

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

For recipes that convert checkpoints through `hf_to_levanter` (such as [curriculum SFT](https://github.com/marin-community/marin/blob/main/experiments/post_training/curriculum_sft/trial.py)), pass `compute_dtype=None` so conversion does not round `router_bias` to BF16 before training starts. For direct `TrainLmConfig` launches, set the same fields on `config.model` and `config.trainer.field_dtype_overrides`.

### SkyRL / Megatron RL and on-policy distillation

Enable captured-route router replay and batch invariance in the SkyRL recipe overlay:

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

FP32 LM heads are omitted on purpose: on Snowball 10/05 with router replay, pairing FP32 LM heads on vLLM and Megatron cut the mean train–serve log-prob gap by 0.6% (`0.00011` nats) while lowering vLLM scoring throughput by ~12% ([issue 11561](https://github.com/marin-community/marin/issues/11561)).

### Repeatable vLLM scoring

Set `vllm_batch_invariant=True` on `ServeConfig` (exports `VLLM_BATCH_INVARIANT=1`) to eliminate repeat and cross-rank drift when scoring fixed checkpoints. Keep the compile mode identical across serves you compare: default compile and `TORCHINDUCTOR_DETERMINISTIC=1` (`benchmark_combo_kernel: false`) disagree by up to `0.0034` nat/tok ([issue 10699](https://github.com/marin-community/marin/issues/10699)).

## Knob reference

| Knob | Use when | Measured effect | Cost / limits |
|---|---|---|---|
| `SnowballConfig.moe_capacity_factor` (default `1.0`) | Any expert-parallel Snowball run; set to the `expert` mesh axis size (e.g. `8.0`) for dropless routing. | Raised Snowball v0.1 top-1 agreement with vLLM from 88.6% to 99.4% ([issue 10695](https://github.com/marin-community/marin/issues/10695), [issue 10698](https://github.com/marin-community/marin/issues/10698)); removing drops and FP32 combine cut p99 log-prob gap from 3.96 to 0.60 nats ([issue 11561](https://github.com/marin-community/marin/issues/11561)). | Larger per-shard dispatch buffers and compute on imbalanced batches. |
| `SnowballConfig.router_dot_fp32` + `field_dtype_overrides={"router_bias": "float32"}` (default `False` / `{}`) | Levanter Snowball parity (matches vLLM, which always runs router dot + logits in FP32). | Part of the Snowball v0.1 8K gate pass (`p99 <= 0.70`, `max <= 3.5` nats; [issue 10698](https://github.com/marin-community/marin/issues/10698)). Not isolated alone. | Negligible compute/HBM; exports `router_bias` as FP32. `train_lm` rejects `router_dot_fp32=True` without FP32 `router_bias`. |
| `SnowballConfig.fp32_activations` (default `False`) | Levanter Snowball parity (FP32 SiLU/sigmoid before casting once). | Single-op BF16 diff vs torch dropped 30–36% to 0%; route agreement vs vLLM rose 0.8984 to 0.9123 (`10` to `22/26` layers passing; [issue 10698](https://github.com/marin-community/marin/issues/10698)). | Rejects `moe_implementation="sonic_cute"`. Peak HBM unchanged at 8K gate shapes (50.5 GiB on H100x8); throughput delta unmeasured. |
| `trainer.{policy,ref}.megatron_config.moe_router_replay` (default `false`) | SkyRL / Megatron RL and on-policy distillation. | On Hero 535B (PP48/EP4/CP4), cut batch-order gradient sensitivity from 71.96% to 1.37% and passed the 4K RL gate (`<0.1%` ratio outliers outside `[0.8, 1.2]`; [issue 10388](https://github.com/marin-community/marin/issues/10388)). | `+5.3%` median scoring + update time; requires `enable_return_routed_experts` on vLLM. |
| `generator.engine_init_kwargs.enable_return_routed_experts` (default `false`) | Required whenever `moe_router_replay` is enabled. | Returns per-token top-k expert IDs from vLLM for replay and route diagnostics. | Route payload transport overhead (included in the `+5.3%` above). |
| `VLLM_BATCH_INVARIANT=1` (`ServeConfig.vllm_batch_invariant=True`) | Repeatable vLLM scoring and batch-invariant RL. | Eliminated vLLM repeat and cross-rank drift on Snowball v0.1 (`206/206` identical repeats; max summed log-prob drift `0.619 -> 0` nats; [issue 10699](https://github.com/marin-community/marin/issues/10699)). | Disables NCCL symmetric memory. 256-request scoring: 38 s default compile, 38–44 s deterministic compile, 56–57 s `--enforce-eager`. |
| `trainer.algorithm.batch_invariant` (default `false`) | SkyRL RL with `VLLM_BATCH_INVARIANT=1`. | Coordinates batch-invariant serving and training in Hero 4K RL qualification ([issue 10388](https://github.com/marin-community/marin/issues/10388)). | Inherits vLLM's 1-channel NCCL limit unless `nccl_max_channels` is set on TP1/ETP1 policies ([SkyRL guide](../dev-guide/skyrl.md#grug-training-communication)). |
| `hf_overrides: {head_dtype: float32}` on vLLM (default BF16) | Opt-in only (small effect). | Halved vLLM self-repeat gap (`0.01287 -> 0.00692` nats/tok), but pairing with FP32 Megatron head cut train–serve gap by only 0.6% ([issue 11561](https://github.com/marin-community/marin/issues/11561)). | ~12% lower vLLM scoring throughput and `+0.6 GiB/GPU` peak HBM. |
| `XLA_FLAGS="--xla_allow_excess_precision=false"` (default unset) | Diagnostic only. | Raised layer-0 route agreement with FA4 to 0.9572, but failed max log-prob bound (`2.5058 > 2.5`) and passed fewer layers (`21/26`). | Disables some XLA GPU fusions; throughput delta unmeasured. |

Unmerged follow-ups tracked in [issue 10698](https://github.com/marin-community/marin/issues/10698) and [issue 11561](https://github.com/marin-community/marin/issues/11561): FP32 MoE combine (`router_combine_fp32`, [PR 10686](https://github.com/marin-community/marin/pull/10686); cut max `|Δ log-prob|` from 4.28 to 2.18 nats, rejects `"sonic"` and `"sonic_cute"`), diagnostic `attention_implementation="torch_order"` ([PR 10688](https://github.com/marin-community/marin/pull/10688); quadratic `S × S` FP32 scores, untested above 8K), and Megatron `fp32_lm_head` ([PR 11671](https://github.com/marin-community/marin/pull/11671); TP1 only, negligible train–serve gain).

## When parity matters

- On-policy RL and on-policy KL distillation directly compare learner and vLLM log-probabilities; enable router replay and batch invariance in SkyRL.
- Fixed-text SFT and offline distillation do not compare learner and vLLM log-probabilities directly, though exported checkpoints are served on vLLM. Default `moe_capacity_factor=1.0` can still silently drop routed tokens in any expert-parallel run, SFT included.

## Coverage limits

- The Levanter Snowball gate ran once on Snowball v0.1 at 8,192 tokens on 32 H100s against vLLM with FA3 (`22/26` route layers passed). Other checkpoints, TPUs, and 32K–262K sequences were not measured.
- The FP32 LM-head probe on Snowball 10/05 ran scoring-only on 512 responses (`480,150` loss tokens) at Megatron TP1/PP2/CP1/EP8 vs vLLM DP8/EP8 with router replay. Production Snowball RL uses CP5, and Mochi (Hero) was not tested.
