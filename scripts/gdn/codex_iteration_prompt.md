You are running one unattended hill-climb iteration for Gated DeltaNet TPU kernels.

Iteration metadata:
- Iteration: {{ITERATION}} / {{TOTAL_ITERATIONS}}
- Starting commit: {{HEAD_SHA}}

Primary objective:
- Reach major end-to-end train-step speedups and push MFU toward ~50%.
- The current evidence says the main bottleneck is no longer just kernel math. It is the lowered train-path control structure around the kernels.
- The latest evidence says the residual `while ~31.6 ms` may now be mostly fused CE on XLA rather than GDN-train-shell work.
- Optimize the training chunk path (`chunk_gated_delta_rule` / flash train kernels) first.

Repo context:
- Kernel implementation: `lib/levanter/src/levanter/layers/gated_deltanet.py`
- Correctness tests: `lib/levanter/tests/test_gdn_kernels.py`, `lib/levanter/tests/test_gdn_layer.py`
- Optimization recipe: `docs/recipes/optimize_gdn_pallas_tpu.md`
- Running log: `lib/levanter/.agents/projects/gdn_pallas_tpu_hillclimb.md`
- Infra CLI: `scripts/gdn/gdnctl.py`
- Reference code: `~/Projects/Work/Marin/flash-linear-attention/fla/ops/gated_delta_rule/`
- Pallas docs: <https://docs.jax.dev/en/latest/pallas/tpu/index.html>

Current diagnosis to optimize against:
- Multiple recent variants cut train-path forward/backward `shard_map/pallas_call` closed-call time by ~40-52%.
- End-to-end throughput still regressed because device-side `while` grew to about `31.5-31.7 ms`.
- Treat that as the dominant failure mode unless new evidence disproves it.
- A candidate that halves closed-call time but worsens train-path control flow is not a win.
- After Iterations 64-66, treat CE backend selection as a first-class bottleneck split, not background noise.

Required behavior for this iteration:
1. Read the latest entries in the running log and identify the current train-path control bottleneck.
2. Generate a shortlist of 3 candidates with estimated upside and implementation risk.
3. Select exactly one macro-move category from `docs/recipes/optimize_gdn_pallas_tpu.md`.
4. Classify the candidate as exactly one of:
   - `CE backend`
   - `outer control structure`
   - `inner kernel math`
5. Prefer a CE/backend or outer-control move over a kernel-math move unless you can explain why the current CE/control-shell evidence does not apply.
6. State what you expect to happen to `while_ms`, and whether the candidate should be rejected if `while_ms` remains flat.
7. Implement one concrete high-upside candidate.
8. Validate correctness on TPU.
9. Launch a lightweight profiled training run on TPU.
10. Analyze the profile, update the running log with structured hotspot metrics, and commit exactly one commit.

Session directives:
- If this prompt includes an extra "Session directives for this codex-loop run" block, treat it as mandatory guidance for this run.

Macro-move category (pick ONE per iteration):

Highest priority:
- `P` CE backend forcing / A/B benchmark on the real train run (default CE vs forced `pallas_tpu`).
- `O` Control arm / reduced-Pallas benchmark branch to test whether the current train-path abstraction boundary is fundamentally wrong.
- `M` XLA-first outer train path with Pallas only as leaf chunk kernels.

High priority only when nested inside `M`/`O` or justified by new CE evidence:
- `N` Backward tape-contract redesign: compressed summaries, new remat/checkpoint boundaries, or another backward structure that reduces scanned residual state.
- `L` Associative chunk summaries / chunk-level affine scan to reduce or remove train-path serial scan shells.

Secondary moves (only when nested inside a new outer structure or explicitly justified):
- `E` V-tiling / shared-K precompute.
- `H` Shared-RHS matmul batching.
- `G` Ct^2 exp-diff reformulation.
- `I` Prepare+recurrent fusion.
- `J` Ct/Seg sweep.

Deprioritized unless you explain why the new control-flow diagnosis does not apply:
- standalone kernel-local wins that preserve the same train-path `scan` / `while` shell,
- standalone associative-summary work with no CE/backend change,
- runtime branchy hot-path variants,
- more iterations whose only visible success metric is lower closed-call time.

Major-bet requirement:
- The optimization must materially change algorithmic decomposition, outer train-path orchestration, backward tape structure, or the lowering-visible control structure.
- Equivalent mathematical reformulations are allowed if semantics remain correct and end-to-end training improves.
- At least one of these must be true:
  - fewer device-side loop/control-flow regions in the hot train path,
  - less scanned residual state in backward,
  - chunk summaries that compose more associatively / in parallel,
  - outer train-path orchestration shifted toward XLA instead of Pallas/custom-VJP scaffolding,
  - direct reduction in end-to-end train-step time without a compensating `while`/`conditional` increase.

Disallowed as a standalone iteration:
- only tweaking scalar constants,
- only toggling config flags/checkpointing/remat with no train-path structural change,
- only reducing forward/backward closed-call time while leaving control-flow overhead worse,
- only logging/plumbing changes.

Hot-path control-flow checklist (answer this in your writeup):
- Change class: `CE backend` | `outer control structure` | `inner kernel math`
- Does this candidate add or preserve a hot-path `lax.scan`?
- Does it add a hot-path `lax.cond` / runtime dispatch?
- Why should that not become a TPU `WhileOp` / `Conditional` hotspot?
- If the candidate keeps a scan shell, why is that still the right bet despite recent evidence?
- What do you expect to happen to `while_ms`?
- Should this candidate be rejected if `while_ms` remains flat? Why?

Acceptance gate checklist (must appear in the iteration writeup):
- Correctness:
  - TPU tests command + result.
- Perf:
  - `CE backend selected: ...`
  - `CE-attributed while: ... ms -> ... ms` when the trace can isolate it.
  - Forward closed-call `... ms -> ... ms`.
  - Backward closed-call `... ms -> ... ms`.
  - `while: ... ms -> ... ms`.
  - `conditional: ... ms -> ... ms`.
  - `Kernel budget: ... ms -> ... ms`.
  - `Control budget: ... ms -> ... ms`.
  - `Train-path budget: ... ms -> ... ms`.
  - `throughput/mfu`, `throughput/tokens_per_second`, `throughput/duration` deltas.
- Governance:
  - If `while` or `conditional` grows materially and MFU does not improve strongly, revert.
  - If CE backend is still `xla` and residual `while` stays large, treat the result as incomplete bottleneck attribution, not closure.
  - If train-path budget worsens, do not treat the candidate as promising even if closed-call time improved.

Failed-attempt handling:
- If the profile run fails, correctness fails deterministically, or control-flow overhead dominates and MFU regresses, do not leave speculative code changes in the tree.
- Revert the failed code attempt and log the exact failure mode.
- Never leave `Commit: (pending)` or `Commit: this commit` in a new log entry.

Constraints:
- TPU-only optimization target.
- No backward-compatibility shims/fallback hacks.
- Do not relax test tolerances.
- If blocked on infra, document the blocker with exact commands and stop without speculative code changes.

Preferred commands:
- `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name "$USER-gdn" --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --ce-implementation pallas_tpu --no-sync`
- `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name "$USER-gdn" --tests both`
- `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name "$USER-gdn" --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --no-sync`
- `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-east5-a --tpu auto --tests both`
- `uv run python scripts/gdn/gdnctl.py ray-profile --cluster us-east5-a --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --no-wait`
- `uv run python scripts/gdn/gdnctl.py lint-log`

Artifact guidance:
- Prefer `perfetto_trace.json.gz` / `trace.json.gz` artifacts by default.
- Only pull `.xplane.pb` artifacts when you will actually use XProf.

Definition of done:
- One high-upside structural attempt committed, tests green, one profiled run completed, running log updated with:
  - measured MFU/tokens/sec deltas,
  - explicit control-flow and kernel budgets,
  - why the change did or did not remove the train-path control bottleneck,
  - next bold hypothesis.
