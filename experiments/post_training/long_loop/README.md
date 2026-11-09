# Long-loop prototype

Snowball SFT → asynchronous RLOO-N (GS8) → Micro210. All stages use the
Marin artifact runner and the `rl_build_options` submission path.

The plumbing graph prepares the 21 benchmark environments, imports the pinned
pre-SFT model, runs 15 SFT updates, exports BF16 parameters with frozen router
biases, checks the served tokenizer/thinking template and both EOS tokens, runs
three RL updates, and evaluates one task on each of the 21 Micro210 paths.
The final report rejects incomplete or unverified benchmark coverage. This is a
runtime test, not a Micro210 score or a completed 36-hour experiment.

```bash
uv run python -m experiments.post_training.long_loop --version 2026.11.08 \
  --source-revision "$(git rev-parse HEAD)"
```

The default prints the graph. Add `--run --max-concurrent 1` to submit after
validation. Each calendar version uses one named coordinator; repeating the
command adopts that job instead of creating another coordinator. Failed stages
are not automatically replayed. Reconcile a failed or ambiguous attempt before
using a new version.

Peak stage requests are 8 H100 for the base import, 32 for SFT, 8 for the
CPU export host, 8 for the endpoint probe, 40 for RL, and 16 for evaluation including
the judge. Dependencies serialize these stages. Every GPU job uses RNO2A at
**interactive** priority; the aggregate limit is **256 H100**. Neither priority
changes nor PR creation are authorized. Keep all changes in this worktree and
merge current main regularly.

Plumbing SFT geometry (32k/32 GPU), not target, uses the archived trainer, 256 sequences per update, and the
initial phase of the 10-05 reproduction composition: 20.4734375% replay.
The original 4,750-step learning-rate schedule and special-token treatment are
preserved; this smoke stops at step 15 (~126M scheduled token positions).
The mesh is replica1/data16/context2 with accumulation8. SFT preemption retries are zero.
Each JAX rank preserves stdout/stderr under `IRIS_OUTPUT_DIR` for the attempt archive.
262k/256-GPU SFT and attempt-scoped JAX bootstrap remain target-geometry qualifications.
`runtime-origin.json` records the reused source bytes. The deleted native base
import is reconstructed from HF revision `058ecaf27b9e4f37219df221a51e7d490d58ec3d`.
The import must verify every one of the 502 tensor mappings.

RL uses the maintained quick-loop data selection `snowball-100k@2026.11.04`,
64 groups per update, GS8, staleness 2, and 24,576/8,192 input/output geometry.
It enables per-turn training, including earlier conversations after an agent
reset. Regression tests cover their action masks, logprobs, behavior versions,
router evidence, and malformed-reasoning penalties. The quick-loop defaults
remain unchanged. Actual learner inputs are dumped for the smoke audit.

Micro210 reuses the Mini runtime from branch commit
`36ce62ecb793ef31def1b87ea7847e643e08691e` associated with issue #10933,
adapted to Snowball and the pinned Micro210 dataset. The policy tokenizer is the
original SFT tokenizer at `3f6879acdaf3dae7874b2d48c7d11b9d8c6faf09`.

`runtime_timing.py` persists coordinator stage-wall JSONL even when a stage fails.
These spans include queue and startup time. Native SFT/SkyRL/Iris telemetry is
still required to separate queue, compilation, learner, rollout, sandbox, judge,
and allocation-release measurements; stage-wall spans must not be reported as
compute time. Full phase collection and the 102,400-token CP4 qualification are
not yet complete. The full 50B round and its task budget remain unmeasured.

## Experiment-owned RL recipe

`rl.py` owns role geometry and launch bindings; `rl-smoke.yaml` owns the local
three-update recipe, copied from the quick-loop composition at main commit
`1cf5c009d358261bd41f368bcd033197ecd731f0`. It no longer imports the quick-loop or
async-RL recipe. Shared Marin/SkyRL execution remains reused. GS8, staleness2,
per-turn training, thinking and dual EOS settings remain explicit.

Training task sessions use `DAYTONA_RL_API_KEY`. Micro210 and its environment
preflight use `DAYTONA_EVAL_API_KEY`. Both are dedicated organization credentials;
never substitute `DAYTONA_API_KEY`. Inline evaluation through the training factory
is disabled; evaluation is the separate Eval-org graph stage. A fresh live run
still requires verified organization identity and an explicit Daytona target.

The current reduced smoke excludes `tau3-pi` under [#11083](https://github.com/marin-community/marin/issues/11083): 20 benchmark paths, including six agentic environment paths. Reports retain the exclusion; this is not complete Micro210 coverage. Standalone preflight reserves nine H100s (one coordinator plus an eight-GPU host).

### Continue after a post-export failure

Use a fresh `--version` and `--reuse-sft-version <completed-version>` to reuse
committed inputs, base weights, SFT, and the verified BF16 export. The launcher
requires each original artifact record, preserves it in the new adoption record,
and builds only the endpoint check, RL, and evaluation under the new version.
This does not resume the SFT optimizer or replay training. The live endpoint check
still compares exact token IDs and tests both EOS tokens before RL starts.

To repair serialization while retaining the native checkpoint, add
`--handoff reexport`. This adopts only the committed inputs, base, and native SFT
artifacts, writes a separately timed export under the new version, and then
continues through endpoint, RL, and evaluation. Both reused and new exports must
contain all schema-2 per-expert tensors before serving starts.
