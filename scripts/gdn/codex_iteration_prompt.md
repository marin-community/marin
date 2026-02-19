You are running one unattended hill-climb iteration for Gated DeltaNet TPU kernels.

Iteration metadata:
- Iteration: {{ITERATION}} / {{TOTAL_ITERATIONS}}
- Starting commit: {{HEAD_SHA}}

Primary objective:
- Reach major end-to-end speedups and push MFU toward ~50%.
- This is still the architecture/kernel-design phase, not fine-grained tuning.
- Favor bold, high-upside kernel redesigns over safe micro-optimizations.

Repo context:
- Kernel implementation: `lib/levanter/src/levanter/layers/gated_deltanet.py`
- Correctness tests: `lib/levanter/tests/test_gdn_kernels.py`, `lib/levanter/tests/test_gdn_layer.py`
- Optimization recipe: `docs/recipes/optimize_gdn_pallas_tpu.md`
- Running log: `lib/levanter/.agents/projects/gdn_pallas_tpu_hillclimb.md`
- Infra CLI: `scripts/gdn/gdnctl.py`
- Reference code: `~/Projects/Work/Marin/flash-linear-attention/fla/ops/gated_delta_rule/`
- Pallas docs: <https://docs.jax.dev/en/latest/pallas/tpu/index.html>

Required behavior for this iteration:
1. Read the latest entries in the running log and identify the dominant current bottleneck from traces.
2. Generate a shortlist of 3 candidate optimizations with estimated upside and risk.
3. Select 1 candidate with high expected impact (target >=10% MFU gain, or clear path to many-fold speedup if successful).
4. Implement the selected change in code.
5. Validate correctness on TPU by running GDN tests.
6. Launch a lightweight profiled training run on TPU.
7. Download trace/profiler artifact, analyze before/after hotspots, and update the running log.
8. If validation passes, commit exactly one commit with the optimization and evidence.

Session directives:
- If this prompt includes an extra "Session directives for this codex-loop run" block, treat it as mandatory guidance for this run.

Major-bet requirement:
- The optimization must materially change kernel structure, launch structure, or algorithmic decomposition.
- Equivalent mathematical reformulations are allowed (including reformulations that avoid explicit triangular inversion) if end-to-end semantics remain correct and performance improves.
- At least one of these must be true:
  - fewer/larger Pallas custom calls per training step (reduce launch overhead),
  - more useful work per kernel (higher arithmetic intensity),
  - changed dataflow/layout/tiling that increases MXU utilization,
  - replaced sequential dependencies with more parallel blockwise/associative structure.

Disallowed as a standalone iteration:
- only tweaking scalar constants (unroll/chunk/segment/batch) with no structural kernel changes,
- only toggling config flags/checkpointing/remat,
- only log formatting or measurement plumbing.

Escalation rule:
- If measured MFU gain is <3% and dominant hotspot is unchanged, mark the attempt as low-impact in the log and set the next hypothesis to a more radical design (not another small tuning step).

Failed-attempt handling:
- If the profile run fails (for example OOM/infra errors) or the change regresses meaningfully, do not leave speculative code changes uncommitted in the tree.
- Either:
  - revert the failed code attempt and record `Commit: none (failed attempt)`, or
  - keep only clearly justified scaffolding that is necessary for the next planned attempt and commit it with explicit rationale.
- Never leave `Commit: (pending)` in a newly-added log entry.

Constraints:
- TPU-only optimization target.
- No backward-compatibility shims/fallback hacks.
- Do not relax test tolerances.
- If blocked on infra/transient cluster issues, document blocker + exact failing command in the running log and stop without speculative code changes.

Preferred commands:
- `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-central1 --tpu auto --tests both`
- `uv run python scripts/gdn/gdnctl.py ray-profile --cluster us-central1 --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --no-wait`
- `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-central1 --tpu-name "$USER-gdn" --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --no-sync`
- `uv run python scripts/gdn/gdnctl.py ray-wait --cluster us-central1 <job_id> --show-logs --tail 400`
- `uv run python scripts/gdn/gdnctl.py lint-log`

Artifact guidance:
- Prefer `perfetto_trace.json.gz` / `trace.json.gz` artifacts by default.
- Only pull `.xplane.pb` artifacts when you will actually run XProf analysis.

Definition of done:
- One high-impact optimization committed, tests green, one profiled run completed, running log updated with:
  - measured MFU/tokens/sec deltas,
  - top hotspots before/after,
  - why this change did or did not unlock large speedup,
  - next bold hypothesis.
