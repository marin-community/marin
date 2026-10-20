# Grug Megatron training

`trainer.strategy=megatron` trains Grug through Megatron-Core with pipeline
parallelism as the primary geometry. Snowball's 26 layers split evenly across
PP2 or PP13; TP must stay at one because the model has five KV heads, and
expert parallelism may be layered on top of PP for the 256 experts. Hero
packed training and context parallelism use the schema-specific path below.

## Hero architecture

The same provider also accepts Hero schema-v2 exports. Hero retains the stored
maximum KV-head count in its checkpoint, then selects the leading local or
global head count for each layer. Its local layers use interleaved half-RoPE;
global layers skip RoPE. The provider reads the global-layer period from the
checkpoint instead of assuming every fourth layer.

Hero's routed experts operate on normalized latent projections. The router and
shared experts still receive the full-width input. Separate shared experts are
added in checkpoint order after the routed output returns to full width.
Schema-v2 expert tensors keep their individual names on import and export.

ShortConv runs before the key norm and after the attention and MLP output
projections. Packed documents have independent convolution histories. With
context parallelism, ranks exchange only the history tails needed by the
kernel and preserve Megatron's two-chunk sequence layout. Local attention uses
Transformer Engine's `a2a` context communication because its `p2p` path rejects
sliding windows; global attention uses `p2p`. Choose TP and CP so both local and
global attention head geometry remains valid. Shared-expert overlap and MLP
chunking are currently unsupported when the Hero modules require them disabled.

The eager Grug model supports Snowball only and rejects Hero configuration. It
must not be used as a Hero reference. The tiny Hero worker test exercises all
Hero parameter families, packed training, repeated updates, and checkpoint
continuation; passing it alone does not establish full-Hero capacity or parity
with Levanter.

The port lives in two modules:

- `skyrl_train.models.grug_megatron` holds the Megatron-Core modules that a
  stock GPT spec cannot express: the gated RMS norms, the weightless QK norm,
  attention with the per-layer query scale, XSA, and the per-head output gate,
  the biased top-(k+1) sigmoid router, and a `GPTModel` subclass that applies
  the gated embedding norm on the first pipeline stage.
- `skyrl_train.models.grug_megatron_bridge` registers the Megatron-Bridge
  provider and weight mappings. Importing the Megatron worker registers the
  bridge so `AutoBridge.from_hf_pretrained` resolves Grug checkpoints.

Sliding-window attention on local layers, no RoPE on long layers, half-RoPE,
grouped-GEMM experts with a shared expert, and GQA use Megatron-Core settings
chosen by the provider (`window_size`, `window_attn_skip_freq`, `no_rope_freq`,
`rotary_percent=0.5`, `moe_grouped_gemm`, `moe_shared_expert_intermediate_size`).
Attention runs through Transformer Engine's fused backend; `trainer.flash_attn`
selects the flash backend instead.

## Weights

Snowball HF checkpoints keep stacked `[E, ...]` expert tensors. The bridge maps
each Megatron per-expert grouped-GEMM weight to one slice of the stacked tensor
on import and re-stacks on export, so exported checkpoints and weight sync use
the same names as vLLM serving. The router bias becomes
Megatron's persistent fp32 `expert_bias` buffer and is sent to vLLM in fp32 in
its own weight-sync bucket; every other tensor is sent in the generator dtype.
With BF16 serving, router weights therefore travel as BF16 before vLLM copies
them into its FP32 router parameters. Hero schema-v2 experts retain individual
per-expert names rather than using the stacked Snowball format.

Re-stacking gathers every expert of a layer onto each rank before the tensor
is sent, which needs a few GiB of headroom beyond the resident model, gradient
buffers, and optimizer state. On the 67B-A2B snowball checkpoint at PP2 x EP8 x
DP2 that headroom does not exist once the optimizer state is materialized, and a
colocated reference model needs the same room for its forward, so disaggregated
runs set `trainer.offload_optimizer_during_rollouts=true` to keep the optimizer
state and gradient buffers on CPU from each policy update until the next one.

## Numerics

Two Megatron behaviours can break the on-policy contract that recomputed old
log-probabilities equal the training forward:

- Megatron's unfused unpermute combines the top-k expert outputs with an atomic
  scatter-add. For top-2 routing the two-term sum is order-independent, but Grug
  routes top-4, so the forward was not reproducible run to run. The bridge forces
  `moe_permute_fusion`, whose Transformer Engine kernels reduce in a fixed order.
- cuBLAS selects kernels per GEMM shape, and at Grug's width the per-row rounding
  differs between kernels. The log-probability forward and the training forward
  must therefore use the same micro-batch size
  (`micro_forward_batch_size_per_gpu == micro_train_batch_size_per_gpu`);
  `validate_cfg` rejects Megatron runs where they differ.

`test_grug_megatron_train_forward_matches_eval_forward` guards both on a toy
model and on a Snowball-shaped tiny model (256 experts, top-4, 20/5 heads,
2048-token window) with variable-length rows at long sequences, and
`test_grug_megatron_eval_forward_is_independent_of_peer_rank_batch` checks that
expert-parallel co-batching does not leak between ranks.

## Memory

The Snowball configs select Megatron's `dp_reshardable` optimizer format, which
writes data-parallel-local shards without gathering optimizer state onto
data-parallel rank zero. Resume must keep the tensor, pipeline, context, and
expert geometry fixed.

S3 checkpoints aggregate each rank's torch-dist items into one multipart object.
Each rank overlaps up to four 64MiB `UploadPart` requests while tensor copy-ahead
is bounded at 1GiB. Including the current multipart buffer, the explicit staging
budget is 1.3125GiB per rank, or 10.5GiB on an eight-GPU host, plus transient
serialization and client overhead. Only small control files such as `common.pt`,
`metadata.json`, and the Hugging Face configuration use a local temporary
directory. Local-path checkpoints retain Megatron Core's standard writer.

Megatron Core 0.18's Multi-Storage Client path remains disabled because its
object writer buffers each complete remote file in `BytesIO` before uploading
it. PyTorch 2.11's one-file-per-rank loop retains all resolved tensors until the
file closes, even though that dictionary is only needed for safetensors. SkyRL's
writer omits that retention for torch serialization and streams the bounded
copy-ahead window into one rank object.

On the GPU, the last pipeline stage holds the vocab-sized logits; the loss
computes entropy under no_grad unless an entropy loss is configured, which
avoids saving two vocab-sized copies for backward. The Snowball configs also
set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Throughput

With 1024-token prompts, 8192-token generations, 64 prompts x 8 samples per
step, four policy nodes at PP2 x EP8 x DP16 and four vLLM nodes at DP8 x EP8,
Megatron completed steps in 190-196 seconds. Policy training took 26-27 seconds,
generation 134-139 seconds, forward log-probabilities 6-7 seconds, and weight
synchronization 15-17 seconds.

With equal micro-batch sizes the recomputed old log-probs and the training
forward agree exactly at this scale: `policy/log_ratio_abs_max` is 0 and
`policy/ppo_ratio_exact_unit_fraction` is 1.0 on every step.

The measurements use `cloud/iris/configs/snowball_megatron_full.yaml`, which
overlaps gradient reduction and parameter gathering with compute and reduces
gradients in bf16. Generation takes about 70% of the step.

## Training communication

Workers with `trainer.algorithm.batch_invariant=true` inherit vLLM's one-channel
NCCL limit. Megatron policies with TP1 and expert TP1 can set a separate training limit:

```text
trainer.algorithm.batch_invariant=true
trainer.policy.megatron_config.tensor_model_parallel_size=1
trainer.policy.megatron_config.expert_tensor_parallel_size=1
trainer.policy.megatron_config.nccl_max_channels=8
```

The limit applies before training process groups are created. Weight transfers
using `generator.weight_sync_transport=broadcast` keep the serving channel limit
on both ends. The `expert_block` transport uses separate groups and requires
`nccl_max_channels=null` or `1`. Leave `NCCL_MAX_CTAS` unset when using this
option: it can force training back to one channel or override the transfer
group's limit. The default `nccl_max_channels=null` keeps the existing behavior.

The [Hero 65K channel study](https://marin-us-east-02a.cwobject.com/marin/users/romain/hero-learner-bottlenecks-01a13c0f/reports/HERO65K-LEARNER-BOTTLENECKS.md)
records the tested layout, precision, memory costs and numerical checks. Its
full-Hero measurements carry [#1000](https://github.com/marin-community/MarinSkyRL/pull/1000)
equally in every arm. They measure learner scoring and updates; rollout and
weight-transfer throughput require separate measurements.

Automatic channel selection ran out of GPU memory on its second update in a
paired 192-H100 Hero test at 65K. The eight-channel control completed eight
updates, with warmed medians of 29.29 seconds per update and 11.18 seconds for
scoring. Eight remains the tested starting point for this workload. See the
[automatic-selection comparison](https://marin-us-east-02a.cwobject.com/marin/users/romain/hero-nccl-auto-01a13c0f/reports/HERO-NCCL-AUTOMATIC-SELECTION.md)
for the channel counts, memory measurements and validation.

On full trained Hero, one and eight channels produced identical scores and
expert choices at each tested batch shape. Both settings changed scores when
sequences shared a larger forward pass. This option preserves that observed
baseline behavior; it does not establish exact batch invariance. See the
[trained-model qualification](https://marin-us-east-02a.cwobject.com/marin/users/romain/hero-nccl-invariance-01a13c0f/reports/HERO-NCCL-BATCH-INVARIANCE.md)
for the mismatch tails, scope and repair checks.

## Query bias

Only the frozen query-bias mode is supported. The bias steers expert
selection exactly as in the HF model but is never updated.

## Validation

`skyrl-train/tests/gpu/test_grug_megatron.py` covers HF log-probability parity
at PP1, PP2, and PP2+EP2, a PP2 training step with an export round trip, and a
four-H100 disaggregated cycle with Marin vLLM. Run it on Iris with
`skyrl-train/ci/marin_nightly/run_grug_megatron.sh`, which resolves the frozen
`megatron` runtime profile.

## Restore memory and deadlines

Megatron training restore accepts CoreWeave `s3://` checkpoints in untransformed
`torch_dist` format. The resolved filesystem endpoint must be `cwobject.com`
(off-cluster) or `cwlota.com` (in-cluster). Other sources fail before loading.

Each worker buffers one complete saved record, copies the requested slice into
its destination, and releases the decoded CPU tensor before reading the next
record. Uncached S3 streams avoid retaining read-ahead blocks between records.
Workers read independently. The checkpoint format is unchanged.

Temporary read memory scales with the number of workers per pod and the largest
saved record, even when a rank only needs a small slice. Leave host-memory room
for the serialized buffers and decoded tensors alongside model and optimizer
destinations, restored non-tensor state, metadata, allocator caches, Ray's object
store, and Megatron's subsequent replica exchange. A record that exceeds this
headroom can still cause an OOM.

Large restores can take longer than the default collective deadline. Set
`trainer.distributed.worker_collective_timeout_seconds` before starting workers;
WORLD and Megatron's model-parallel subgroups use the same configured deadline.
