# Ising Tokenizer: Research Logbook

Issue: https://github.com/marin-community/marin/issues/4104
Branch: https://github.com/marin-community/marin/tree/research/ising-tokenizer

## Scope

- Goal: test whether a small temperature-conditioned transformer can learn tokenized 2D Ising BKL trajectories from synthetic off-critical data.
- Primary metric(s): train/validation next-token loss, deterministic tokenization, and basic held-out off-critical / near-critical teacher-forced loss.
- Constraints: stay local-first, keep the code explicit, and avoid framework growth before the data path is validated.

## Baseline

- Date: 2026-03-23
- Code refs:
  - `/Users/dlwh/.codex/worktrees/86f8/marin/experiments/ising_tokenizer/base/data.py`
  - `/Users/dlwh/.codex/worktrees/86f8/marin/experiments/ising_tokenizer/base/model.py`
  - `/Users/dlwh/.codex/worktrees/86f8/marin/experiments/ising_tokenizer/base/train.py`
  - `/Users/dlwh/.codex/worktrees/86f8/marin/experiments/ising_tokenizer/base/launch.py`
- Baseline numbers:
  - local smoke train loss: `4.884 -> 4.309`
  - local smoke held-out off-critical validation loss: `4.889 -> 4.289`
  - near-critical teacher-forced probe loss at `Tc=2.26918531421`: `4.531 -> 4.219`
  - default smoke shape: `L=8`, `48` train examples, `48` recorded events, `seq_len=225`, `vocab_size=131`

## Experiment Log

### 2026-03-23 00:00 - Kickoff scaffold

- Hypothesis: a minimal local scaffold is enough to validate the Ising trajectory/tokenization path before adding executor or rollout complexity.
- Command:
  - scaffold only
- Config:
  - local synthetic dataset
  - BKL-style single-spin-flip dynamics with naive full-lattice rate recompute
  - `[pos][spin]` initial state and `[pos][dt_bin]` event tokenization
  - continuous scalar temperature conditioning in a grug-style transformer
- Result:
  - scaffold started
- Interpretation:
  - first decision point is whether a tiny smoke run produces sane losses and dataset statistics
- Next action:
  - run the local smoke and append the numbers here

### 2026-03-23 22:54 - First local synthetic smoke is healthy

- Hypothesis: even a very small local run should learn enough off-critical structure to reduce next-token loss on both train and held-out temperatures, with the near-critical probe at least staying finite.
- Command:
  - `uv run python experiments/ising_tokenizer/base/launch.py --output-dir artifacts/ising_tokenizer/small_bkl_v0 --steps 16 --train-examples 48`
- Config:
  - `8x8` lattice
  - burn-in `96`
  - recorded events `48`
  - train temperatures: `1.5`, `1.8`, `2.8`, `3.1`
  - held-out validation temperatures: `1.6`, `2.9`
  - critical probe temperature: `2.26918531421`
  - tokenizer: `64` log-spaced `dt` bins
  - model: `hidden_dim=96`, `num_layers=3`, `num_heads=4`
  - trainer: `16` steps, batch size `8`, AdamW `2e-3`
- Result:
  - initial train loss: `4.88399`
  - final train loss: `4.30946`
  - initial validation loss: `4.88857`
  - final validation loss: `4.28920`
  - critical probe loss: `4.53057 -> 4.21903`
  - validation by temperature:
    `T=1.6 -> 4.34655`
    `T=2.9 -> 4.23363`
  - artifact:
    `/Users/dlwh/.codex/worktrees/86f8/marin/artifacts/ising_tokenizer/small_bkl_v0/metrics.json`
- Interpretation:
  - the basic path is real: deterministic synthetic data, continuous temperature conditioning, and local grug-style training all work end to end
  - off-critical generalization is not obviously broken on the first smoke
  - the near-critical teacher-forced loss is finite, which is the minimum sanity bar before any Tc rollout claims
- Next action:
  - decide whether to spend the next pass on rollout sampling, better `dt` quantization, or executor/Marin integration

### 2026-03-23 23:07 - First Iris `v4-8` launch exposed TPU sequence-length constraint

- Hypothesis: the local smoke path should scale to a modest Iris TPU run on `us-central2-b` without any TPU-specific code changes.
- Command:
  - `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --cpu 16 --memory 64GB --extra marin:tpu --tpu v4-8 --region us-central2 --zone us-central2-b --job-name ising-tokenizer-bkl-v4-8-r1 -- uv run python experiments/ising_tokenizer/base/launch.py --output-dir gs://marin-tmp-us-central2/ttl=7d/dlwh/ising_tokenizer/ising-tokenizer-bkl-v4-8-r1 --steps 1024 --train-examples 2048`
- Config:
  - Iris config: `lib/iris/examples/marin.yaml`
  - TPU: `v4-8`
  - region/zone: `us-central2` / `us-central2-b`
  - train examples: `2048`
  - train steps: `1024`
  - output path:
    `gs://marin-tmp-us-central2/ttl=7d/dlwh/ising_tokenizer/ising-tokenizer-bkl-v4-8-r1`
- Result:
  - job id: `/dlwh/ising-tokenizer-bkl-v4-8-r1`
  - worker started the user command
  - run failed during model evaluation with:
    `NotImplementedError: Splash attention requires key/value sequence length to be a multiple of 128.`
- Interpretation:
  - this was a TPU-path configuration bug, not an Iris scheduling failure
  - the fixed-length Ising sequence needed explicit pad-tail handling for TPU splash attention
- Next action:
  - add explicit pad token + pad-tail masking, then resubmit under the same monitoring loop

### 2026-03-23 23:18 - TPU padding fix landed and Iris resubmit is pending on capacity

- Hypothesis: after padding trajectories to a multiple of `128` and masking the pad tail, the same `v4-8` launch should get past the earlier splash-attention initialization failure.
- Command:
  - same Iris submit command as above
- Config:
  - added explicit `PAD` token support
  - padded `seq_len` to the configured multiple of `128`
  - masked `EOS` plus the full pad tail in the loss
  - `monitoring_state.json` updated with `restart_count=1`
- Result:
  - scaffold tests still pass after the padding fix
  - resubmitted job id remains `/dlwh/ising-tokenizer-bkl-v4-8-r1`
  - current Iris state:
    `JOB_STATE_PENDING`
  - current pending reason:
    `Autoscaler: Unsatisfied autoscaler demand: no_capacity: tpu_v4_8-us-central2-b=backoff`
- Interpretation:
  - the code-side TPU blocker is fixed locally
  - the active blocker is now slice availability in `us-central2-b`, not user-code failure
- Next action:
  - keep the Iris job pending and retry only if the user wants a different region, TPU type, or a new submission strategy

### 2026-03-24 00:43 - `v5p-8` mesh fix landed and a fresh Iris rerun is pending

- Hypothesis: the `v5p-8` worker failure was not a TPU kernel bug, it was the smoke loop calling splash attention outside a JAX mesh; restoring the normal TPU attention path and entering a real mesh in the local train/eval loop should let the worker start cleanly once capacity arrives.
- Command:
  - `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --cpu 16 --memory 64GB --extra marin:tpu -e WANDB_API_KEY $WANDB_API_KEY --tpu v5p-8 --region us-central1 --zone us-central1-a --job-name ising-tokenizer-bkl-v5p-8-r2 -- uv run python experiments/ising_tokenizer/base/launch.py --output-dir gs://marin-tmp-us-central1/ttl=7d/dlwh/ising_tokenizer/ising-tokenizer-bkl-v5p-8-r2 --steps 1024 --train-examples 2048 --wandb-project tokexplore --wandb-entity marin-community --wandb-group ising-tokenizer --wandb-run-name ising-tokenizer-bkl-v5p-8-r2 --wandb-tags ising-tokenizer bkl ising v5p-8 iris`
- Config:
  - restored `attention(...)` in the Ising model instead of the reference-only fallback
  - added a one-axis JAX `data` mesh around the smoke loop
  - replicated model/optimizer arrays onto `NamedSharding`
  - batch-sharded inputs on axis `0` when divisible, otherwise replicated
  - recorded the debugging thread in:
    `/Users/dlwh/.codex/worktrees/86f8/marin/docs/debug-log-ising-tokenizer-tpu-mesh.md`
- Result:
  - targeted repo checks passed:
    `./infra/pre-commit.py --fix experiments/ising_tokenizer/base/model.py experiments/ising_tokenizer/base/train.py docs/debug-log-ising-tokenizer-tpu-mesh.md`
  - scaffold tests passed again:
    `uv run --with pytest python -m pytest -o addopts='' tests/test_ising_tokenizer_scaffold.py`
  - the original `v5p` job `/dlwh/ising-tokenizer-bkl-v5p-8-r1` remains failed with the pre-fix mesh traceback
  - fresh rerun submitted as:
    `/dlwh/ising-tokenizer-bkl-v5p-8-r2`
  - current Iris state:
    `JOB_STATE_PENDING`
  - current pending reason:
    `Scheduler: Insufficient TPUs (need 4, available 0) ... Autoscaler: Waiting for workers in scale group 'tpu_v5p_8-us-central1-a' to become ready`
- Interpretation:
  - the mesh-side code fix is in and locally verified
  - the active blocker for the new `v5p` job is again capacity, not the earlier startup traceback
- Next action:
  - wait for `v5p-8` capacity and inspect the first worker logs for a new W&B run link or a new startup error

### 2026-03-24 00:49 - First successful `v5p-8` training run looks healthy

- Hypothesis: if the mesh fix is sufficient, the first successful TPU run should show a large drop in teacher-forced loss on train and held-out temperatures without any NaNs or worker instability.
- Command:
  - completed run:
    `/dlwh/ising-tokenizer-bkl-v5p-8-r2`
- Config:
  - TPU: `v5p-8`
  - region/zone: `us-central1` / `us-central1-a`
  - train examples: `2048`
  - train steps: `1024`
  - W&B:
    `https://wandb.ai/marin-community/tokexplore/runs/i23pldvi`
  - artifact:
    `gs://marin-tmp-us-central1/ttl=7d/dlwh/ising_tokenizer/ising-tokenizer-bkl-v5p-8-r2/metrics.json`
- Result:
  - Iris state:
    `JOB_STATE_SUCCEEDED`
  - initial train loss: `4.91060`
  - final train loss: `2.20123`
  - initial validation loss: `4.90710`
  - final validation loss: `2.21021`
  - final critical probe loss at `Tc=2.269`: `2.19839`
  - held-out validation by temperature:
    `T=1.600 -> 2.22154`
    `T=2.900 -> 2.19795`
- Interpretation:
  - the TPU path is now genuinely healthy
  - the model is learning a nontrivial trajectory distribution, not just staying finite
  - this is enough evidence to justify one incrementally larger run without changing the experiment design yet
- Next action:
  - launch a still-small but more serious follow-up with more data and optimization steps on the same hardware path

### 2026-03-24 00:49 - Small-real follow-up launched on `v5p-8`

- Hypothesis: scaling the same off-critical setup from `2048/1024` to `8192/4096` examples/steps should give a cleaner estimate of whether the loss improvements persist without needing any new modeling machinery.
- Command:
  - `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --cpu 16 --memory 64GB --extra marin:tpu -e WANDB_API_KEY $WANDB_API_KEY --tpu v5p-8 --region us-central1 --zone us-central1-a --job-name ising-tokenizer-bkl-v5p-8-small-real-r1 -- uv run python experiments/ising_tokenizer/base/launch.py --output-dir gs://marin-tmp-us-central1/ttl=7d/dlwh/ising_tokenizer/ising-tokenizer-bkl-v5p-8-small-real-r1 --steps 4096 --train-examples 8192 --wandb-project tokexplore --wandb-entity marin-community --wandb-group ising-tokenizer --wandb-run-name ising-tokenizer-bkl-v5p-8-small-real-r1 --wandb-tags ising-tokenizer bkl ising v5p-8 iris small-real`
- Config:
  - TPU: `v5p-8`
  - region/zone: `us-central1` / `us-central1-a`
  - train examples: `8192`
  - train steps: `4096`
  - output path:
    `gs://marin-tmp-us-central1/ttl=7d/dlwh/ising_tokenizer/ising-tokenizer-bkl-v5p-8-small-real-r1`
- Result:
  - job id:
    `/dlwh/ising-tokenizer-bkl-v5p-8-small-real-r1`
  - immediate Iris state:
    `JOB_STATE_RUNNING`
- Interpretation:
  - this is the same validated code path with a modest scale-up, not a new experimental branch
  - the next useful signal is whether the new run starts logging cleanly and maintains the improved off-critical generalization
- Next action:
  - inspect early logs for the new W&B run id and watch for the first validation points

### 2026-03-24 08:41 - Small-real follow-up completed cleanly and improved validation further

- Hypothesis: if the same model/data path scales cleanly, the `8192/4096` run should finish without TPU instability and beat the earlier `2048/1024` validation numbers by a modest but real margin.
- Command:
  - completed run:
    `/dlwh/ising-tokenizer-bkl-v5p-8-small-real-r1`
- Config:
  - TPU: `v5p-8`
  - region/zone: `us-central1` / `us-central1-a`
  - train examples: `8192`
  - train steps: `4096`
  - artifact:
    `gs://marin-tmp-us-central1/ttl=7d/dlwh/ising_tokenizer/ising-tokenizer-bkl-v5p-8-small-real-r1/metrics.json`
- Result:
  - Iris state:
    `JOB_STATE_SUCCEEDED`
  - initial train loss: `4.91046`
  - final train loss: `2.16178`
  - initial validation loss: `4.90710`
  - final validation loss: `2.15782`
  - final critical probe loss at `Tc=2.269`: `2.17671`
  - held-out validation by temperature:
    `T=1.600 -> 2.12187`
    `T=2.900 -> 2.19660`
- Interpretation:
  - the larger off-critical run still behaves cleanly
  - validation improved versus the earlier `v5p-8-r2` run (`2.21021 -> 2.15782`)
  - the near-critical teacher-forced probe also improved (`2.19839 -> 2.17671`), so scaling data/steps is still buying signal
- Next action:
  - either launch one more modest scale-up, or switch effort into sampling / trajectory-level evaluation around `Tc`

### 2026-03-24 08:52 - Rollout evaluation path works locally

- Hypothesis: the minimal useful next step is not another teacher-forced-only run, but a rollout path that prompts on the true initial spin field and samples only the dynamics tokens, then compares sampled vs reference trajectory observables by temperature.
- Command:
  - `uv run python experiments/ising_tokenizer/base/launch.py --output-dir artifacts/ising_tokenizer/rollout_smoke_v0 --steps 16 --train-examples 48 --rollout-examples 2`
- Config:
  - rollout prompt = true initial state tokens from held-out examples
  - sampled suffix = fixed-length `[pos][dt]...` dynamics tokens with a simple grammar mask
  - rollout observables:
    `mean_wait_time`
    `boundary_flip_fraction`
    `final_abs_magnetization`
- Result:
  - local scaffold tests passed with rollout enabled
  - local smoke completed and wrote rollout metrics into:
    `/Users/dlwh/.codex/worktrees/86f8/marin/artifacts/ising_tokenizer/rollout_smoke_v0/metrics.json`
  - teacher-forced losses still improved as expected:
    train `4.90982 -> 4.27244`
    validation `4.90708 -> 4.28874`
  - rollout metrics are noisy and still rough at this tiny scale, which is expected for a barely trained local smoke:
    `T1.600 wait_time_ratio -> 0.343`
    `T2.269 wait_time_ratio -> 2.044`
    `T2.900 wait_time_ratio -> 4.673`
- Interpretation:
  - the rollout-eval machinery is alive end to end
  - the local smoke is only a plumbing check, not an evidence run
  - the next useful experiment is the same stable TPU training path with rollout evaluation enabled at the end
- Next action:
  - submit a rollout-enabled `v5p-8` run at the already-validated `8192 / 4096` scale

### 2026-03-24 08:52 - Rollout-enabled TPU run submitted

- Hypothesis: the stable `8192 / 4096` training setup is the right place to first read rollout quality; if the model is genuinely learning near-critical structure, this should show up in the sampled observables before we spend time on more elaborate analysis.
- Command:
  - `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --cpu 16 --memory 64GB --extra marin:tpu -e WANDB_API_KEY $WANDB_API_KEY --tpu v5p-8 --region us-central1 --zone us-central1-a --job-name ising-tokenizer-bkl-v5p-8-rollout-r1 -- uv run python experiments/ising_tokenizer/base/launch.py --output-dir gs://marin-tmp-us-central1/ttl=7d/dlwh/ising_tokenizer/ising-tokenizer-bkl-v5p-8-rollout-r1 --steps 4096 --train-examples 8192 --rollout-examples 16 --wandb-project tokexplore --wandb-entity marin-community --wandb-group ising-tokenizer --wandb-run-name ising-tokenizer-bkl-v5p-8-rollout-r1 --wandb-tags ising-tokenizer bkl ising v5p-8 iris rollout`
- Config:
  - TPU: `v5p-8`
  - region/zone: `us-central1` / `us-central1-a`
  - train examples: `8192`
  - train steps: `4096`
  - rollout examples per temperature: `16`
  - output path:
    `gs://marin-tmp-us-central1/ttl=7d/dlwh/ising_tokenizer/ising-tokenizer-bkl-v5p-8-rollout-r1`
- Result:
  - job id:
    `/dlwh/ising-tokenizer-bkl-v5p-8-rollout-r1`
  - immediate Iris state:
    `JOB_STATE_PENDING`
  - current pending reason:
    `Autoscaler: Unsatisfied autoscaler demand: no_capacity: tpu_v5p_8-us-central1-a=backoff`
- Interpretation:
  - rollout evaluation is now on the critical path for the experiment series
  - current blocker is TPU availability, not user-code failure
- Next action:
  - wait for allocation, then inspect the first worker logs and eventual rollout metrics
