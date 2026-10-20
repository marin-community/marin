# Training-batch capture and replay

This opt-in GPU diagnostic captures the input to the production policy-forward boundary, then replays that
boundary from the named checkpoint without starting rollout generation or inference engines. It does not alter
the normal trainer or its configuration schema, and it cannot recover a historical batch that was never captured.

## Contract

- Run from `skyrl-train/` in the policy image. Do not use `uv --isolated`; replay must use the production Torch,
  CUDA, and extension stack.
- Start from an explicit `trainer.resume_path` ending in `global_step_<N>`. The captured target must be `N + 1`.
- Use a new artifact directory on trusted shared POSIX storage. The artifact contains pickle data and may contain
  task inputs; do not load an artifact from an untrusted writer.
- Keep every training Hydra value identical between capture and replay. The diagnostic-only `batch_replay` subtree
  is excluded from the config fingerprint; its source revision, checkpoint, and target step are checked separately.
- Use the full 40-character source revision baked into the image. Provenance mismatches fail before Ray starts or
  any model worker is dispatched.

## Entrypoint

Use the original job's config and Hydra overrides, but select:

```yaml
entrypoint: tests.gpu.fault_injection.training_batch_replay.entrypoint
```

Add these Hydra overrides for capture:

```text
trainer.resume_mode=from_path
trainer.resume_path=/shared/checkpoints/<run>/global_step_<N>
+batch_replay.mode=capture
+batch_replay.artifact_path=/shared/diagnostics/<case>-pre-forward
+batch_replay.target_step=<N+1>
+batch_replay.source_revision=<40-character-commit>
```

Capture runs the normal fully-async TerminalBench job. Immediately before the target step's policy forward, it
atomically publishes `manifest.json` and `batch.pkl`, then continues into the production forward. The artifact is
therefore complete even if that forward raises an OOM. An existing destination is never overwritten.

For replay, submit the same config and overrides with only:

```text
+batch_replay.mode=replay
```

Replay validates the artifact, source, resolved config, and checkpoint before initialization. It builds the
policy/ref/critic actors and a training context, loads the checkpoint, performs the production policy-event-loop
drain, and invokes `fwd_logprobs_values_reward`. The context owns a rollout buffer, and enabled trajectory retention
starts a sink. An empty inference client and an unstarted rollout pool satisfy the trainer's worker interface.
Full shutdown releases the model actors, buffer and retention resources. Success is the log marker:

```text
TRAINING_BATCH_REPLAY_OK target_step=<N+1> tensors=...
```

A provenance error is a setup failure, not evidence about the original OOM. An OOM or hang after replay begins is
the diagnostic result; preserve the complete job log and GPU/runtime details with the artifact manifest.
