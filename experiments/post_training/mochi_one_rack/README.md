# Mochi one-rack RL smoke

This experiment starts from a privately retained Mochi SFT step 395161 and its BF16 schema-3 export. It runs five optimizer updates with a 4,096-token combined cap, then starts a fresh job against the saved model/optimizer checkpoint and advances from step 5 to step 10 with a 32,768-token cap. The second stage restarts task ordering so its controlled length probe is consumed. Native checkpoints are retained; terminal HF export is disabled.

The allocation is 15 GB200 trays in one NVLink domain on `cw-us-east-08a`, at `production` priority: 48 training GPUs (TP1/PP12/EP4/CP4) and 12 rollout GPUs (TP1/PP1/DP12/EP12). The cluster's ordinary Kueue topology policy keeps this gang in one domain. This fits the 15 schedulable trays observed in the remaining domain; the initial 64-GPU candidate needed one more tray. AdamW optimizer state uses the existing CPU offload path to fit the trainer footprint. vLLM compilation and CUDA graphs remain enabled.

The recipe comes from Ben's long-loop recipe at `bf163020a7115d85fd3fcb02099d3abfda3b6cf2`. It preserves RLOO-N and eight responses per prompt group, using two groups per update. Sample packing enables the existing Megatron context-parallel path; training still uses full trajectories. It keeps zero-reward groups for this runtime smoke, so the controlled length probe is not filtered out. The real tasks are two original Harbor CalibForge tasks and two non-agentic arithmetic/sequence tasks from Ben's pinned mixed-task artifact. The second stage replaces one short arithmetic prompt with a clearly marked visible ledger plus the same arithmetic question and verifier. Its rendered prompt has 31,611 valid tokens; it does not use masked padding to reach that length.

`data.py` checks the original Parquet SHA-256 values before selection and emits a manifest of task identities and stage checksums. The tokenizer and chat template are copied without changes from the SFT run's `2026.09.18` tokenizer. The source tokenizer manifest pins their hashes. Generation stops on both token IDs 128001 and 128009.

After the CPU export completes and the original tokenizer files and generation config are present, run `handoff.py` in the vendored SkyRL CPU environment with `--hf-root` set to the private export URI. It validates shard sizes against the completed export receipts and writes SkyRL's immutable portable-model manifest. It preserves the original tokenizer bytes and uses the exporter's shard SHA-256 values. Verify tokenizer and policy metadata staging before allocating GPUs.

Build and inspect the graph before submitting it:

```bash
uv run python -m experiments.post_training.mochi_one_rack.launch \
  --stage 32k --version 2026.11.11.9264
```

Load the dedicated `DAYTONA_RL_API_KEY` and `DAYTONA_EVAL_API_KEY` through the documented Secret Manager path. Set `MARIN_PREFIX` to the private experiment prefix, then add `--run` to the same command. Stop at `--stage 4k` for a diagnostic of only the first stage. Use a fresh calendar version after runtime changes; the input artifact identities remain explicit in `launch.py`.

The wheel in `config/external/vllm/mochi-gb200.toml` is experimental. `runtime.vllm_backend()` selects it through Marin's ordinary CUDA serving launcher without changing the promoted `gpu.toml`. The vendored SkyRL lock selects the same vLLM bytes. Its Python implementation is the merged schema-3 source `3004cd8f163299116b96d1b324cfff418c387388`; its SM100 native payloads are byte-identical to release `5fa2e9bad66dcc2b9836e657323214aefc179a7f`. The matching Torch 2.14.1/cu132 FlashAttention package retains FA2 kernels and excludes an incompatible optional FA4 namespace. Artifact manifests are attached to the task-scoped prerelease.

Validation is restricted to Mochi on GB200. This experiment does not qualify a general vLLM or SkyRL release. H100, TPU, Snowball parity, other model families and online EAGLE remain outside its validation scope. The promoted Marin GPU descriptor and protected branches stay unchanged. Vendored dependency changes select the experimental runtime on aarch64; x86 GPU pins retain their existing versions and have no new accelerator validation.
