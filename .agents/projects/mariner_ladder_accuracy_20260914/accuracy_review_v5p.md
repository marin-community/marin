# MARINER ladder accuracy: v5p placement correction review

Reviewed 2026-09-14 UTC after the first parent was rejected before TPU admission because the configured v6e pool is in us-east5-b, while the approved placement is us-east5-a. This addendum preserves `accuracy_review.md` unchanged; its SHA-256 is `98266b14cb464b2e49c1c45ef098be90b338d2fcd00287c3345a3ce17bd05071`.

## Verdict

Proceed with the corrected v5p-8 plan. The resource request now matches a configured TPU pool in the required zone, and the runtime batch calculation preserves the planned total batch of eight on either four or eight exposed JAX devices. No new source-level blocking issue was found. Actual admission, cache loading and inference still require first-child verification.

## Placement and resources

- `lib/iris/config/marin.yaml` defines `v5p-preemptible` in us-east5-a and includes size 8. Its per-VM advertised resources are 208 CPUs, 448GB RAM and 100GB disk.
- The worker now requests `v5p-8`, eight CPUs, 64g RAM and 32g disk, with both `regions=(us-east5,)` and `zone=us-east5-a`. `ResourceConfig.with_tpu` constructs that exact request successfully.
- Parent and child region, GCS checkpoint, cache and output paths remain in us-east5; changing TPU generation does not authorize moving data or placement to another region or zone.
- The smaller disk request fits the configured 100GB pool. The regional cache inventory totals 880,355,765 bytes (0.820 GiB). Each archived checkpoint has 12.603 GiB of weight shards, but the current HF converter streams remote safetensors through fsspec/JAX rather than saving those shards to local disk. The 32g request therefore has no identified disk-capacity blocker from staging these inputs; runtime logs and temporary result serialization still consume space.

## Total evaluation batch

The Iris topology table explicitly maps v5p-8 to four chips on one VM, while v6e-8 has eight. After `DistributedConfig.initialize()`, the worker reads `jax.device_count()`, rejects an incompatible total batch, and sets `per_device_eval_parallelism = batch_size // device_count`. Levanter's default mesh sets the data axis to all devices, model and replica axes to one, and computes the total eval batch as per-device parallelism times the data-axis size. Thus the requested batch eight produces two packed sequences per device with four devices, or one with eight devices. I checked `MeshConfig.axis_shapes` for both cases without initializing accelerators. This avoids accidentally halving the total evaluation batch on v5p.

## Unchanged protocol

The original review's scientific and output-retention checks still apply: only completed Proportional and UniMax-8 step-22056 HF exports; all eleven task families, 67 leaves and 44,248 documents; five-shot except zero-shot LAMBADA; no chat template; context length 4096; fixed seed zero; BF16 parameters and compute; complete metrics and samples; strict durable result verification. The checkpoint-local Qwen3 tokenizer/converter correction remains present. No benchmark, checkpoint, prompt, random-seed or reporting change was introduced by this resource correction.

Changing the frozen TPU type changes the plan hash, so the corrected submission must use its regenerated plan/source hashes rather than the rejected v6e plan. No completed inference result exists to mix across the two hardware plans.

Reviewed worker SHA-256: `7a453e2e16515aba84eb98ab3ab9eea8dcf015dd86a996fd31e174ab64cb4cab`.
