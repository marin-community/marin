# EP25 mission brief (shared by all four agents)

## Goal

Raise p50 MFU at the GB200 EP operating point to **>= 25%** while preserving numerics and
fidelity. Hard constraints:

- MFU denominator locked to **2.5 PFLOP/s per GB200 bf16 dense**; report p50 over >= 100
  measured steps.
- Numerics: gradient parity at rtol=atol=1e-5 in kernel-level tests; end-to-end loss
  trajectory sane and descending over 100+ steps.
- Fidelity: dropped-token counts must not significantly increase vs baseline (report the
  counts in every A/B).

## Operating point

MuonH · d5120 · **8-of-256 experts** · 48 layers · i1280 · shared i5120 · seq 4096 ·
batch 1024 · sliding window 2048 · EP64 (one GB200 rack = 16 nodes x 4 GPUs) ·
fixed-capacity all_to_all (`SCALE_A2A_FIXED=1`) + gather dispatch. The hero-run candidates
(#7201) are d6144 4-of-256 / 4-of-128; d5120 8-of-256 is the measured proxy shape — stay on
it unless your direction specifically requires otherwise.

**Baseline: 20.558% p50 MFU** (318.5K tok/s, 13.17s step) — issue 7201, comment
5073017396. The 17.552% scatter control is the committed tip of branch `rav/ep-2`
(fe21ea495 = your worktree base). The +3.0pp gather-dispatch patch is NOT committed —
reconstruct it from the comment (core snippet is in the comment body; it modifies
`lib/levanter/src/levanter/grug/_moe/ep_ragged_all_to_all.py`, selected via
`SCALE_A2A_GATHER_DISPATCH=1`).

Read these first (READ-ONLY gh usage):

```bash
gh api repos/marin-community/marin/issues/comments/5074952738 --jq .body   # ranked directions, cross-thread state
gh api repos/marin-community/marin/issues/comments/5073017396 --jq .body   # baseline run: config, patch, exact submit cmd
```

## Exact baseline submission (adapt job-name/env for your arms)

```bash
uv sync   # once, in your worktree
IRIS_USER=mwittmann .venv/bin/iris --cluster=marin job run --no-wait \
  --target-cluster cw-us-east-08a --priority interactive \
  --job-name <JOB> -e RUN_ID <JOB> \
  -e SCALE_ATTN_IMPL gpu_fa4_cute \
  -e SCALE_WATCH_INTERVAL 0 -e SCALE_CHECKPOINTS local \
  -e SCALE_A2A_FIXED 1 -e SCALE_A2A_CHUNKS 1 -e SCALE_A2A_NO_BARRIER 1 \
  -e SCALE_GPUS_PER_NODE 4 -e SCALE_GPU_TYPE GB200 -e SCALE_GPU_REPLICAS 16 \
  -e SCALE_EXPERT_AXIS 64 -e SCALE_NUM_EXPERTS 256 -e SCALE_TOP_K 8 \
  -e SCALE_HIDDEN_DIM 5120 -e SCALE_NUM_LAYERS 48 \
  -e SCALE_INTERMEDIATE 1280 -e SCALE_SHARED_INTERMEDIATE 5120 \
  -e SCALE_SEQ_LEN 4096 -e SCALE_BATCH 1024 -e SCALE_SLIDING_WINDOW 2048 \
  -e SCALE_STEPS 120 -e SCALE_MOE_IMPL ragged_all_to_all \
  -e SCALE_OPTIMIZER muonh -e SCALE_MUON_SYRK 1 \
  -e SCALE_SCAN_LAYERS 1 -e SCALE_REMAT recompute_all \
  -e SCALE_TRACKER json_logger -e SCALE_JSON_LOGGER <JOB>.metrics \
  -e SCALE_DISABLE_CHECKPOINT 1 \
  -- python -m experiments.grug.moe.launch_cw_scale --version <your-tag> --run
```

Job names: prefix with your direction id, e.g. `ep25d1-<desc>-<date>`. Monitor with
`.venv/bin/iris --cluster=marin job status/logs <job>`; see `lib/iris/OPS.md`.

## Cluster etiquette and gotchas

- Cluster: `cw-us-east-08a` (GB200, arm64 nodes). NEVER stop/restart any Iris cluster.
- **At most ONE rack-scale (16-replica) job in flight per agent.** Debug on 1 replica
  (4 GPUs) first. Jobs may queue behind other agents — poll, do not resubmit.
- Use `SCALE_TRACKER json_logger` (wandb streaming is blocked from this sandbox).
- Placement variance is ±2–4 pp MFU across allocation draws: run matched A/B legs
  back-to-back, and never claim a win < ~0.5pp from single draws.
- Fast-restart hazard: a populated JAX persistent cache at leader startup can deadlock
  NCCL clique init — rotate/disable the cache dir if you hit boot hangs.
- New files under `lib/*` are silently skipped by a global gitignore: `git add -f` them,
  or iris bundling will not ship them and your job will run stale code. Verify a changed
  file actually took effect (e.g. log a sentinel at import).
- No cross-region data movement; keep artifacts in-region (s3://marin-us-east-02a/tmp/ttl=30d/...).
- GB200 4-GPU dev jobs need explicit `--cpu`/`--memory` or jax import OOMs.

## Rules of engagement

- Work ONLY inside your assigned worktree. Commit locally, frequently, with descriptive
  messages. **NEVER `git push`. NEVER comment/create/edit anything on GitHub** (reads via
  `gh api` are fine).
- Keep an append-only `AGENT_LOG.md` at your worktree root. Every entry timestamped.
  **At least every 15 minutes** append a check-in block:

  ```
  ## Check-in <UTC time>
  Findings so far: <2-6 bullets, numbers first>
  Confidence: <n>/10 that this direction contributes a significant step toward 25% MFU
  Next: <what you are doing right now>
  ```

- The coordinator will read your AGENT_LOG.md and may send you follow-up instructions
  (including peer-agent summaries). Rank honestly; a confident negative is a valuable
  result.
- Definition of done for a round: either a measured matched A/B at the operating point
  (p50 MFU + drop counts + numerics evidence), or a confident negative with the evidence
  that killed the direction.
