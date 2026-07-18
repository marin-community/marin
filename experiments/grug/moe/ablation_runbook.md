# MoE Ablation Runbook

Standing procedure for the d1024 / d2048 MoE architecture ablations. Every ablation
follows this recipe exactly; only the single variable under test changes. Everything not
specified comes from the **May Recipe heuristic** (`MoeHeuristic` in `heuristic.py`), which
`build_scale_model` already sources automatically.

**Branch:** run from `b200_mla` (or `grug/embedding-gather-shard-map`) — these carry the
`SCALE_DATA=datakit` toggle and the heuristic-config/LR sourcing (commits `039a1b74a7`,
`f33d239fcf`). No launcher changes are needed.

## Fixed parameters (both sizes)

| | d1024 ablation | d2048 ablation |
|---|---|---|
| Hardware | **4× B200** | **16× B200** |
| Batch size (sequences) | **256** | **1024** |
| Sequence length | **4096** | **4096** |
| Token budget | **5B** | **50B** |
| **Steps** = tokens / (batch × seq) | **4,768** | **11,921** |
| Dataset | datakit 2-phase mix | datakit 2-phase mix |
| Experts / top-k | **128 / 4** | **128 / 4** |
| LR schedule | warmup 0.01, linear decay to 0.05× peak | same |
| Optimizer | MuonH (heuristic) | MuonH (heuristic) |

Step counts:
- d1024: `5e9 / (256 × 4096)` = `5e9 / 1,048,576` = **4,768**
- d2048: `50e9 / (1024 × 4096)` = `50e9 / 4,194,304` = **11,921**

## Heuristic-derived model configs (128 experts)

`build_scale_model` bases the config on `MoeHeuristic().build_model_config(hidden_dim,
seq_len=4096)` (layers/heads/intermediate/`initializer_std = 0.5/√hidden`) and overrides
only the routed-expert count, layer count, and backend knobs. Layer count is the heuristic
formula `round(d / (64 + log2(d)·4 − 9))` — **passed explicitly** because the launcher's
`SCALE_NUM_LAYERS` default is 48, not the heuristic value.

| | d1024 | d2048 |
|---|---|---|
| Layers | **11** | **21** |
| Heads / KV | 8 / **2** | 16 / **2** |
| head_dim | 128 | 128 |
| Routed intermediate | 512 | 1024 |
| Shared-expert intermediate | 1024 | 2048 |
| Experts / top-k | 128 / 4 | 128 / 4 |
| Sliding window | 2048 | 2048 |
| initializer_std | 0.0156 | 0.0110 |
| qk_mult | 1.3 | 1.3 |
| Active params (excl. embed/lm_head) | ~140M | ~1.02B |
| Total params (all 128 experts) | ~2.29B | ~17.4B |
| Training FLOPs (`3·fpt·tokens`, excl. lm_head, seq 4096) | ~6.8e18 | ~4.05e20 |

> Heuristic default is 256 experts; these ablations use **128** (matches the recent e128
> convention). Set via `SCALE_NUM_EXPERTS=128` (launcher default on this branch is 64).
>
> **GQA default is 2 KV heads** for both sizes — set via `SCALE_NUM_KV_HEADS=2`. (The launcher
> default is ~4:1, which would give d1024 = 2 but d2048 = 4; we fix it at 2, so d1024 is 4:1 and
> d2048 is 8:1.)
>
> **MLA defaults** (vs the GQA baseline): attention gate **off** (the `MultiheadLatentAttention`
> path has no headwise gate; GQA's `CausalSelfAttention` keeps it on), **qk_mult off (1.0)** instead
> of the heuristic's 1.3, and the **latent-dim corrections on** (post-RMSNorm `sqrt(hidden/latent)`
> rescale of the Q/KV latents). Override with `SCALE_QK_MULT`, `SCALE_MLA_SCALE_Q_LORA`,
> `SCALE_MLA_SCALE_KV_LORA`.
>
> **All-global attention** (no sliding window): `SCALE_SLIDING_WINDOW=4096` (= seq_len) makes the
> short-layer window span the full sequence, so every layer is global (long layers are already).

## Optimizer / LR

`SCALE_OPTIMIZER=muonh` (default) routes through
`MoeHeuristic(min_lr_ratio=0.05, max_learning_rate=SCALE_MAX_LR).build_optimizer_config(...)`:
peak LR / β / ε set from (tokens, batch, hidden_dim), **warmup 0.01, linear decay to 0.05×
peak, no grad clipping** — exactly the requested schedule, no override needed. `SCALE_LR`
overrides the peak if ever wanted; `SCALE_MAX_LR` caps it (default 0.05).

## Dataset: datakit 2-phase mix

`SCALE_DATA=datakit` selects the two-phase datakit store mixture (`datakit_data_config` in
`launch_datakit_moe_mix.py`, store `datakit/store_8ac06c74`, root path at `marin_prefix` so
it resolves on CoreWeave):

- Per-bucket `phase_0` and `phase_1` weights (cluster×quality buckets), each normalized.
- **Phase 1 begins at 0.8 of total steps** — first 80% draws the phase-0 mixture, last 20%
  the phase-1 mixture.
- Simulated epoching against the store's ~10.37T-token budget.

The launcher builds it from `(steps, batch_size, max_seq_len)` automatically; just pass
`SCALE_DATA=datakit`. Default remains `slimpajama` (the fast MFU/throughput dataset).

## Evals & checkpoints

- **Evals:** the standard **paloma + uncheatable** validation suites (tokenized with
  `marin_tokenizer` to match the datakit store) run **every 1000 steps by default**
  (`eval_batch_size=512`, `max_eval_batches=8`, current model, no EMA). **On by default**
  (`SCALE_EVAL=0` to skip, e.g. a pure throughput run); `SCALE_EVAL_INTERVAL` overrides the
  1000-step cadence (used for short smoke tests). Wired into `launch_cw_scale.py`.
- **Checkpoints:** use `SCALE_CHECKPOINTS=s3` (the default) for ablations — it writes periodic
  (every 10 min) **and** a forced **final checkpoint** to the durable `output_path`. Do **not**
  use `local` here: that writes to node-local `/tmp` and is disposable (fine only for the MFU
  throughput tests). The final checkpoint is always force-saved at the last step regardless of
  interval; `s3` is what makes it persist.

## Run command template

`iris job run` against CoreWeave (`cw-us-east-08a`), launched from the ablation branch.

```bash
# d1024 ablation: 4 B200, bs256, 5B tokens (4768 steps)
iris --cluster=marin job run --no-wait --enable-extra-resources \
  --target-cluster cw-us-east-08a --priority interactive \
  --cpu 2 --memory 8GB --disk 32GB --timeout 21600 --job-name <RID>-coord \
  -e SCALE_GPUS_PER_NODE 4 -e SCALE_GPU_TYPE GB200 -e SCALE_GPU_REPLICAS 1 \
  -e SCALE_EXPERT_AXIS 1 -e SCALE_REPLICA_AXIS 1 \
  -e SCALE_HIDDEN_DIM 1024 -e SCALE_NUM_LAYERS 11 -e SCALE_NUM_EXPERTS 128 -e SCALE_NUM_KV_HEADS 2 -e SCALE_TOP_K 4 \
  -e SCALE_SEQ_LEN 4096 -e SCALE_BATCH 256 -e SCALE_STEPS 4768 \
  -e SCALE_OPTIMIZER muonh -e SCALE_DATA datakit -e SCALE_EVAL 1 \
  -e SCALE_ATTN_IMPL gpu_fa4_cute -e SCALE_MOE_IMPL sonic_cute -e SCALE_SCAN_LAYERS 1 \
  -e SCALE_MUON_DIST_NONEXPERT 1 -e SCALE_MUON_INTRA_RACK 1 -e SCALE_MUON_PAD_NONEXPERT 1 -e SCALE_MUON_SYRK 1 \
  -e CE_IMPL liger -e CE_LIGER_CHUNK 8192 -e SCALE_REMAT recompute_all -e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async \
  -e SCALE_TRACKER wandb -e WANDB_API_KEY "$WANDB_API_KEY" -e WANDB_PROJECT marin_moe -e SCALE_CHECKPOINTS s3 \
  -e RUN_ID <RID> -- python -m experiments.grug.moe.launch_cw_scale

# d2048 ablation: 16 B200, bs1024, 50B tokens (11921 steps)
#   change: SCALE_GPU_REPLICAS 4, SCALE_HIDDEN_DIM 2048, SCALE_NUM_LAYERS 21,
#           SCALE_BATCH 1024, SCALE_STEPS 11921
```

Batch divisibility: d1024 256/(4 shards)=64; d2048 1024/(16 shards)=64. Both integer.

## Sub-branch + issue per ablation

This runbook lives on the harness branch **`grug/moe-ablations`**. Every ablation gets its own
**sub-branch off that harness branch** holding just the single-variable change, plus a tracking
**issue** that rolls up to [#7374](https://github.com/marin-community/marin/issues/7374) and
links the branch + the exact launch commands.

1. **Sub-branch:** one per ablation, off this harness branch —
   `git checkout grug/moe-ablations && git checkout -b grug/moe-abl/<slug>`. Make the single
   change under test (default everything else to this runbook), commit, and push it.
2. **Title:** `Agent MoE Experiment: <description>`
3. **Body:** the **verbatim user prompt** that defined the ablation, then:
   - a link to the sub-branch (`grug/moe-abl/<slug>`),
   - the derived config (d1024 + d2048 params) and the single variable under test,
   - the **exact launch command for each size** — the run-command template above with this
     ablation's `RUN_ID` and any changed knobs filled in (copy-pasteable).
4. **Roll-up:** add as a **sub-issue of #7374** via the GitHub GraphQL API (required — do not
   skip). Fetch the parent + child node IDs, then call `addSubIssue`:

   ```bash
   # 1. Get node IDs for the parent (#7374) and the new issue
   gh api graphql -f query='
   query {
     repository(owner: "marin-community", name: "marin") {
       parent: issue(number: 7374) { id }
       child: issue(number: <NEW_ISSUE_NUMBER>) { id }
     }
   }'

   # 2. Add the sub-issue relationship
   gh api graphql -f query='
   mutation {
     addSubIssue(input: {issueId: "<PARENT_ID>", subIssueId: "<CHILD_ID>"}) {
       issue { number }
       subIssue { number }
     }
   }'
   ```

   Also add the `agent-generated` label to the child issue.
5. Post throughput / loss / eval results as `🤖`-prefixed comments as runs land.

## Per-ablation checklist

1. Sub-branch `grug/moe-abl/<slug>` off `grug/moe-ablations`; apply the single variable under
   test (default everything else to this runbook); commit + push.
2. Open the `Agent MoE Experiment: …` issue — verbatim prompt + config + sub-branch link +
   the exact launch command per size; roll it up to #7374.
3. Launch d1024 (4×B200, 4768 steps) and d2048 (16×B200, 11921 steps) with `SCALE_DATA=datakit`
   from the sub-branch.
4. Record final loss + throughput + eval; post results to the issue.
