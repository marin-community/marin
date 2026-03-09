You are running one unattended hill-climb iteration for Gated DeltaNet TPU kernels.

Iteration metadata:
- Iteration: {{ITERATION}} / {{TOTAL_ITERATIONS}}
- Starting commit: {{HEAD_SHA}}

Primary objective:
- Reach major end-to-end train-step speedups and push MFU toward ~50%.
- The current evidence says CE backend selection was the last giant false wall, and that has already been partly fixed.
- The next bottlenecks are:
  - the residual CE backward/custom-VJP shell, and
  - the untracked post-train-path remainder of the step.
- Optimize the training chunk path (`chunk_gated_delta_rule` / flash train kernels) first.

Repo context:
- Kernel implementation: `lib/levanter/src/levanter/layers/gated_deltanet.py`
- Correctness tests: `lib/levanter/tests/test_gdn_kernels.py`, `lib/levanter/tests/test_gdn_layer.py`
- Optimization recipe: `docs/recipes/optimize_gdn_pallas_tpu.md`
- Running log: `lib/levanter/.agents/projects/gdn_pallas_tpu_hillclimb.md`
- Infra CLI: `scripts/gdn/gdnctl.py`
- Reference code: `~/Projects/Work/Marin/flash-linear-attention/fla/ops/gated_delta_rule/`
- Additional local references:
  - `~/Projects/Work/Marin/ejkernel/ejkernel/modules/operations/gated_delta_rule.py`
  - `~/Projects/Work/Marin/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py`
  - `~/Projects/Work/Marin/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_bwd.py`
  - `~/Projects/Work/Marin/EasyDeL/easydel/operations/kernels/gated_delta_rule.py`
- Pallas docs: <https://docs.jax.dev/en/latest/pallas/tpu/index.html>

Current diagnosis to optimize against:
- Iteration 67 was the regime change: forcing TPU fused CE to `pallas_tpu` cut CE-attributed `while` from about `31.6 ms` to about `10.1 ms` and improved MFU by about `10.9%`.
- Iteration 68 then reduced GDN forward closed-call and train-path budget materially while MFU still regressed slightly.
- Iteration 69 showed a full-XLA chunk control arm is useful as diagnosis, but clearly worse than the deployable head.
- External review of local `ejkernel` / `EasyDeL` references points to one new high-value idea:
  - smaller backward residual contract,
  - recompute-heavy backward from raw inputs plus chunk-start state,
  - simpler chunk-first experimental control surface.
- Therefore:
  - `GDN budget down` is no longer a reliable proxy for `step faster`,
  - the residual CE backward/custom-VJP shell is a first-class target,
  - `remainder_budget_ms = step_duration_ms - train_path_budget_ms` must be tracked explicitly,
  - the next control arm should validate the ejkernel-style backward/tape tradeoff before returning to kernel-local tuning.

Required behavior for this iteration:
1. Read the latest entries in the running log and identify the current train-path control bottleneck.
2. Generate a shortlist of 3 candidates with estimated upside and implementation risk.
3. Select exactly one macro-move category from `docs/recipes/optimize_gdn_pallas_tpu.md`.
4. Classify the candidate as exactly one of:
   - `CE backend`
   - `outer control structure`
   - `inner kernel math`
5. Unless you are explicitly running the CE backward A/B matrix, keep CE fixed at:
   - `LEVANTER_FUSED_CE_IMPLEMENTATION=pallas_tpu`
   - one explicit CE backward mode (`pallas` or `xla_streaming`)
6. State what you expect to happen to:
   - `while_ms`
   - `step_duration_ms`
   - `remainder_budget_ms`
7. State whether the candidate should be rejected if `while_ms` remains flat or if `remainder_budget_ms` grows.
8. Implement one concrete high-upside candidate.
9. Validate correctness on TPU.
10. Launch a lightweight profiled training run on TPU.
11. Analyze the profile, update the running log with structured hotspot metrics, and commit exactly one commit.

Session directives:
- If this prompt includes an extra "Session directives for this codex-loop run" block, treat it as mandatory guidance for this run.

Macro-move category (pick ONE per iteration):

Highest priority:
- `R` ejkernel-style training control arm:
  - minimize saved residuals/tape,
  - recompute chunk prepare intermediates in backward from raw inputs + saved chunk-start state,
  - prefer a plain chunked experimental surface before reintroducing any `segment_size` hierarchy,
  - if the arm changes geometry, try chunk-size candidates `{32, 64}` first.
- `P` CE backward-mode A/B on the real train run (`pallas_tpu` CE + `pallas` backward vs `pallas_tpu` CE + `xla_streaming` backward).
- `O` Control arm / reduced-Pallas benchmark branch to diagnose whether the current train-path abstraction boundary is fundamentally wrong. This is diagnostic, not a mainline promotion target.
- `M` XLA-first outer train path with Pallas only as leaf chunk kernels. This is lower priority than `P` unless CE backward evidence is exhausted.

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
- more iterations whose only visible success metric is lower closed-call time,
- forward-only GDN wins that do not move `step_duration_ms` or `remainder_budget_ms`.

Major-bet requirement:
- The optimization must materially change algorithmic decomposition, outer train-path orchestration, backward tape structure, or the lowering-visible control structure.
- Equivalent mathematical reformulations are allowed if semantics remain correct and end-to-end training improves.
- If you use the `ejkernel` / `EasyDeL` reference arm:
  - copy the backward/tape idea first, not the entire stack,
  - keep inference conclusions separate from training conclusions,
  - state exactly which residuals are removed and which intermediates are recomputed instead.
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
- only logging/plumbing changes,
- only copying external `ejkernel` / `EasyDeL` code without adapting it to this train path.

Hot-path control-flow checklist (answer this in your writeup):
- Change class: `CE backend` | `outer control structure` | `inner kernel math`
- Does this candidate add or preserve a hot-path `lax.scan`?
- Does it add a hot-path `lax.cond` / runtime dispatch?
- Why should that not become a TPU `WhileOp` / `Conditional` hotspot?
- If the candidate keeps a scan shell, why is that still the right bet despite recent evidence?
- What do you expect to happen to `while_ms`?
- What do you expect to happen to `remainder_budget_ms`?
- Should this candidate be rejected if `while_ms` remains flat or `remainder_budget_ms` grows? Why?

Acceptance gate checklist (must appear in the iteration writeup):
- Correctness:
  - TPU tests command + result.
  - Treat TPU correctness as complete only when it runs through `gdnctl`'s remote TPU test wrapper with both `torch` and `transformers` installed. A result like `49 passed, 40 skipped` is incomplete; the full parity slice is expected to look like `87 passed, 2 skipped`.
- Perf:
  - `CE backend selected: ...`
  - `CE bwd mode: pallas | xla_streaming`
  - `CE-attributed while: ... ms -> ... ms` when the trace can isolate it.
  - Forward closed-call `... ms -> ... ms`.
  - Backward closed-call `... ms -> ... ms`.
  - `while: ... ms -> ... ms`.
  - `conditional: ... ms -> ... ms`.
  - `Kernel budget: ... ms -> ... ms`.
  - `Control budget: ... ms -> ... ms`.
  - `Train-path budget: ... ms -> ... ms`.
  - `Step duration: ... ms -> ... ms`.
  - `Remainder budget: ... ms -> ... ms`.
  - `throughput/mfu`, `throughput/tokens_per_second`, `throughput/duration` deltas.
- Governance:
  - If `while` or `conditional` grows materially and MFU does not improve strongly, revert.
  - If CE backend is still `xla` and residual `while` stays large, treat the result as incomplete bottleneck attribution, not closure.
  - If train-path budget improves but step duration does not, classify it as `off-critical-path` / `overlap-loss`.
  - If remainder budget grows materially, do not treat the candidate as promising even if train-path budget improved.

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
- `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name "$USER-gdn" --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --ce-implementation pallas_tpu --ce-bwd-mode pallas --no-sync`
- `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name "$USER-gdn" --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --ce-implementation pallas_tpu --ce-bwd-mode xla_streaming --no-sync`
- `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name "$USER-gdn" --tests both`
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
