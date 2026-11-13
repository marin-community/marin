# Mochi one-rack RL smoke

This experiment starts from retained Mochi SFT step 395161 and its private BF16 schema-3 export. Hero MuonH starts with fresh optimizer state and a fresh RL schedule. The artifact graph has three separate GPU jobs: `4k-initial` applies updates 1–4 and saves native checkpoint 4; `4k` restores that checkpoint, applies update 5 at the same 4,096-token combined cap and saves checkpoint 5; `32k` restores checkpoint 5 and applies updates 6–10 at a 32,768-token cap. Each continuation loads an exact checkpoint and writes to its own output root. Terminal HF export is disabled.

The numerical reference is [Hero EP at Marin 4af966070e5d0252e87d404196400c35bd04883b](https://github.com/marin-community/marin/tree/4af966070e5d0252e87d404196400c35bd04883b/experiments/grug/moe_hero_ep). MuonH owns attention, expert, shared-expert, latent and GatedNorm matrices. AdamH owns the readout; Adam owns embeddings, routers, attention gates, scalar norm gains and ShortConv kernels. Q/K/V and fused gate/up weights remain separate logical matrices. The recipe selects Hero EP's `all_experts` normalization: each layer's routed gate, up and down bank has one Frobenius sphere across expert owners. Only norm scalars cross the expert-parallel group. Shared experts retain separate spheres. The runtime's existing per-expert mode remains available to other recipes.

The smoke uses constant MuonH/AdamH LR 1e-5 and Adam LR 2.307692307692308e-6, preserving the Hero heuristic's 13/3 LR ratio. Betas are (0.9, 0.95), Nesterov momentum is 0.95, and five quintic Newton–Schulz steps run in BF16. Both epsilons are 1e-8. Warmup, weight decay and gradient clipping are zero. These fixed smoke rates and schedule replace the pretraining token-budget heuristic; no pretraining moments are imported. BF16 model weights use full FP32 masters, momentum and Adam moments. Muon momentum stays on pinned CPU memory with bounded GPU transfer scratch. Megatron's distributed optimizer and AdamW CPU-offload route are disabled. Checkpoints identify whole-bank Hero EP MuonH separately from per-expert MuonH and reject an AdamW checkpoint.

The allocation is 15 GB200 trays in one NVLink domain on `cw-us-east-08a`, at `production` priority: 48 training GPUs (TP1/PP12/EP4/CP4) and 12 rollout GPUs (TP1/PP1/DP12/EP12). vLLM compilation stays in mode 3, with CUDA graphs disabled for this recipe. This is a configuration fallback for NCCL failures in the variable-size expert exchange after weight publication; [upstream #59646](https://github.com/vllm-project/vllm/pull/59646) proposes the same graph setting for this backend. Attention/cache updates and the stateful `vllm::grug_moe_short_conv` remain compilation boundaries. Retest piecewise graphs after a fix for this backend is qualified.

Each tray requests 120 CPUs, 900 GiB RAM and 2 TiB disk (`900GB`/`2TB` in Iris's resource parser). The task's memory limit is also 900 GiB; sidecars add their own requests. This leaves room for CPU coordinators on the shared rack. Check cgroup usage and Ray's effective memory limit during load, restore, training and checkpoint saving. Ray's existing per-job override caps its object store at 32 GiB. RunAI uses its CPU-buffered loading path with an 8 GiB buffer and four requests per worker, keeping temporary loading allocations bounded. The model stays BF16; distributed expert-parallel inference and weight publication use their ordinary NCCL paths.

Agent trials have four model turns and a five-minute execution limit; Harbor still runs each original verifier afterward. Individual non-agentic model requests allow five minutes at 4K and ten minutes at 32K, including first-use kernel compilation and long-prompt prefill. Each GPU stage has a five-hour attempt timeout. The CPU coordinator allows 72 hours, including placement waits; its earlier eight-hour limit expired while the 32K job was queued and cancelled that job. The follow-up has no fixed stop time; corrected retries use fresh artifact versions.

Weight publication checks the reported version and byte count from every trainer and receiver, plus each receiver's expert count. Later trajectory records retain the consumed model version. The optional full-byte verification replay is disabled after an observed NCCL broadcast timeout during replay; preceding runs byte-checked the same initial weights successfully. This leaves the replay failure unresolved.

The recipe comes from Ben's long-loop recipe at `bf163020a7115d85fd3fcb02099d3abfda3b6cf2`. It preserves RLOO-N and eight responses per prompt group, using two groups per update. Two rollout workers share the 16-trial Harbor allowance, giving each worker enough slots to execute an eight-response group together. Sample packing enables the existing Megatron context-parallel path; training still uses full trajectories. It keeps zero-reward groups for this runtime smoke, including both the controlled length probe and all-zero agentic groups. Uniform outcome rewards give zero base RLOO-N advantages. The existing reasoning-marker penalty can still produce nonzero token advantages and gradients; inspect the actual masked advantages for each group. Retained trajectories still pass through forward and backward computation. The real tasks are two original Harbor CalibForge tasks and two non-agentic arithmetic/sequence tasks from Ben's pinned mixed-task artifact. The 32K stage has five fixed pairs, each with one agentic and one non-agentic task. Its first pair replaces the arithmetic prompt with a clearly marked visible ledger plus the same question and verifier. The remaining pairs use the original tasks. This exercises the controlled length probe once across the five updates. Its tokenizer preflight has 31,611 valid prompt tokens before the runtime system and tool messages are added. The runtime batch receipts measure the actual valid length; masked padding does not provide the probe length.

`data.py` checks the original Parquet SHA-256 values before selection and emits a manifest of task identities and stage checksums. The tokenizer and chat template are copied without changes from the SFT run's `2026.09.18` tokenizer. The source tokenizer manifest pins their hashes. Generation stops on both token IDs 128001 and 128009. Mini-SWE sends an OpenAI Bash tool definition; automatic tool selection uses the existing Hermes parser, matching the SFT template's `<tool_call>` JSON format. A typed context rejection ends the installed agent normally so Harbor can verify the sampled prefix; other request errors keep their existing handling.

After the CPU export completes and the original tokenizer files and generation config are present, run `handoff.py` in the vendored SkyRL CPU environment with `--hf-root` set to the private export URI. It validates shard sizes against the completed export receipts and writes SkyRL's immutable portable-model manifest. It preserves the original tokenizer bytes and uses the exporter's shard SHA-256 values. Verify tokenizer and policy metadata staging before allocating GPUs.

Build and inspect the graph before submitting it:

```bash
uv run python -m experiments.post_training.mochi_one_rack.launch \
  --stage 4k --version 2026.11.11.9300
```

Load the dedicated `DAYTONA_RL_API_KEY` and `DAYTONA_EVAL_API_KEY` through the documented Secret Manager path. The recipe explicitly selects Daytona's `us` target, which full Harbor trials require in their host connection. Set `MARIN_PREFIX` to the private experiment prefix, then add `--run` to the same command. The `4k` terminal runs the initial job and its short-context continuation. Inspect its applied update 5 before launching `--stage 32k`. `--stage 4k-initial` stops after checkpoint 4. Use a fresh calendar version after runtime changes; the input artifact identities remain explicit in `launch.py`.

Pass `--initial-four-k-version` to reuse a completed checkpoint-4 artifact when retrying short continuation. After short continuation succeeds, use a fresh version with `--stage 32k --four-k-version <completed-4k-version>` to reuse its checkpoint 5. If the initial and resumed 4K jobs used different versions, pass both versions so the graph reuses the same dependencies:

```bash
uv run python -m experiments.post_training.mochi_one_rack.launch \
  --stage 32k --version 2026.11.11.9302 \
  --initial-four-k-version 2026.11.11.9300 \
  --four-k-version 2026.11.11.9301
```

Artifact names under `checkpoints/mochi-muonh/` keep these fresh-optimizer runs separate from the earlier AdamW experiment. Native checkpoints retain the Muon momentum, Adam moments, FP32 masters, optimizer counters, schedule and RNG state.

After a stage completes, use `--stage context` for a measured context trial. It adopts the completed MuonH artifact without rebuilding the earlier ladder and checks its native checkpoint marker before submission. The trial creates a small data artifact from the retained four-task subset and original tokenizer. Every update pairs an original Harbor task with the marked arithmetic length probe and preserves eight responses per group. The requested cap must fit the model's declared maximum. For example, continue a completed 32K checkpoint 10 with three 48K updates:

```bash
uv run python -m experiments.post_training.mochi_one_rack.launch \
  --stage context --version 2026.11.12.9305 \
  --context-cap 49152 --resume-stage 32k \
  --resume-version 2026.11.12.9304 --resume-step 10 --updates 3
```

Inspect the graph and effective configuration before adding `--run`. Use a fresh version for each trial. A trial writes its own checkpoint root and stops at the resumed step plus `--updates`. Three updates provide later rollouts that can consume the first two updated versions. Confirm their archived model versions, actual attended lengths, positive loss masks and applied updates. A configured cap or padded tensor size alone does not establish a passing length.

Choose the next cap from measured CUDA allocation/reservation, cgroup memory peaks and host availability, including checkpoint saving. The 900 GiB main-container limit is lower than Ray's host-based 95% threshold on these hosts, so use the cgroup limit when calculating headroom. Increase in measured steps and retain a passing checkpoint. Stop the search at the model maximum or a practical bracket between passing and resource-limited lengths; diagnose runtime failures before treating them as resource limits.

This qualification uses the vendored SkyRL lock: vLLM source `0ac9dcb157ee86ccb447bc717259159c41ad456a`, Torch 2.14.1/cu132, stock Transformer Engine 2.21, Core 0.18 and Bridge 0.6, and unmodified upstream FA2 source `f87af617fab71c29c11e7f9bc74746ae7534a59d`. Both GPU architectures select the published wheels. Speculators and hs-connectors use the normal `0.8.0+marin.1` release. The separate experimental descriptor `config/external/vllm/mochi-gb200.toml` does not select SkyRL's environment.

This full-model check covers Mochi on GB200. Focused H100 and GB200 checks qualify the same native wheel bytes separately; they do not establish full-model H100 or learning-quality results. Grug BF16 selects FlashAttention because the tested cuDNN training and evaluation paths gave different log-probabilities at fixed weights. Native TE handles ordinary expert matmul; the Torch adapter applies only when `trainer.algorithm.batch_invariant=true` requests exact peer-batch invariance.
