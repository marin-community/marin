# Plan To Run Tunix On TPU With `infra/launch.py`-Style Orchestration

## Goal

Get Tunix running on TPU VMs with the same operational model that already works
for Levanter:

1. build a Docker image locally,
2. push it to Artifact Registry or GHCR,
3. create or reuse a TPU queued resource,
4. SSH to the TPU VM,
5. `docker run` the training job there,
6. optionally retry and autodelete.

The end result should feel like this:

```bash
python infra/launch_tunix.py \
  --zone us-central1-a \
  --tpu_name tunix-worker \
  --tpu_type v5p-32 \
  --capacity_type on-demand \
  -- \
  python3 -m tunix.cli.grpo_main ...
```

## Recommendation

The best path is:

1. keep the TPU orchestration logic from Levanter almost verbatim,
2. do **not** try to shoehorn Tunix into the Marin workspace,
3. instead, create a **local-only Tunix launcher overlay inside a cloned Tunix repo**,
4. build the Docker image from the Tunix repo root,
5. use a Tunix-specific Dockerfile,
6. validate with a supported Tunix CLI workflow first: `peft_main.py` or
   `grpo_main.py`.

This is better than extending the current Marin/Levanter launcher in-place.

## Why This Is The Best Way Forward

### What Levanter already solves well

`infra/launch.py` plus the helpers under `src/levanter/infra/` already give us
the hard part:

- Docker build and push: `infra/launch.py`, `src/levanter/infra/docker.py`
- TPU queued-resource lifecycle: `src/levanter/infra/tpus.py`
- remote Docker setup on TPU VMs: `src/levanter/infra/tpus.py`
- retry/autodelete/foreground behavior: `infra/launch.py`

Those pieces are reusable and do not care that the payload is Tunix instead of
Levanter.

### What is Levanter-specific and should not be reused blindly

The current Levanter launcher bakes in repo assumptions that are wrong for
Tunix:

- it finds the current git repo root and assumes the Dockerfile lives at
  `lib/levanter/docker/tpu/Dockerfile.incremental`
- its Dockerfile is a Marin workspace build that depends on `uv.lock`,
  workspace `pyproject.toml` files, and `uv sync --package levanter`
- its default image metadata and environment are Levanter-oriented

Those assumptions are in:

- `infra/launch.py`
- `docker/tpu/Dockerfile.incremental`
- `src/levanter/infra/cli_helpers.py`

Trying to run Tunix from inside the Marin workspace would work only after
adding more special cases than we want.

### Why the Tunix clone should own the launcher overlay

Tunix itself expects:

- execution from the Tunix repo root,
- `python3 -m tunix.cli.peft_main ...` or
  `python3 -m tunix.cli.grpo_main ...`,
- `tunix/cli/base_config.yaml` or the compatibility fallback for bare
  `base_config.yaml`,
- repo-local example scripts under `examples/...`

Relevant files:

- `/Users/ahmed/code/.cache/tunix/tunix/cli/config.py`
- `/Users/ahmed/code/.cache/tunix/tunix/cli/peft_main.py`
- `/Users/ahmed/code/.cache/tunix/tunix/cli/grpo_main.py`
- `/Users/ahmed/code/.cache/tunix/examples/rl/grpo/gsm8k/run_llama3.2_1b.sh`
- `/Users/ahmed/code/.cache/tunix/examples/sft/mtnt/run_qwen2.5_0.5b.sh`

If we put the launcher overlay in the Tunix clone, all of those assumptions
stay natural:

- Docker build context is just the Tunix repo,
- the container `WORKDIR` can be the Tunix repo root,
- example scripts keep working,
- Tunix can be updated independently with `git fetch` / `git checkout`.

## Important Constraint: Tunix CLI Does Not Currently Cover DPO

Your old Levanter example was a DPO launch:

```bash
python infra/launch.py ... -- python src/levanter/main/train_dpo.py ...
```

Tunix is different:

- officially exposed CLI entrypoints are `peft_main.py` and `grpo_main.py`
- DPO exists as a library API (`DPOTrainer`)
- there is **not** a first-class Tunix DPO CLI entrypoint in this snapshot

That means the plan should be:

1. first prove the TPU launcher with a supported Tunix CLI target,
2. then, if DPO is the real goal, add a small custom `run_dpo.py` entrypoint in
   the Tunix overlay that imports and calls Tunix's `DPOTrainer`.

Do not make DPO the first milestone. That adds too many moving parts at once.

## Recommended Repository Layout

I would use a Tunix clone as the working repo and add a local-only launcher
overlay there.

Example:

```text
/Users/ahmed/code/tunix/
  .git/
  tunix/
  examples/
  infra/
    launch_tunix.py
    docker.py
    tpus.py
    cli_helpers.py
  docker/
    Dockerfile.tpu
  scripts/
    run_grpo_llama3_1b.sh
    run_sft_qwen_0p5b.sh
    run_dpo.py              # later, only if needed
```

The files under `infra/` should start as copied/adapted versions of:

- `infra/launch.py`
- `src/levanter/infra/docker.py`
- `src/levanter/infra/tpus.py`
- `src/levanter/infra/cli_helpers.py`

This is intentionally pragmatic. It is better to cargo-cult a working launcher
than to redesign TPU orchestration from scratch.

## Proposed Docker Strategy

### Recommendation

Use a Tunix-specific Dockerfile, but start from the same TPU-capable base image
that Levanter already uses:

```dockerfile
ARG IMAGE=ghcr.io/marin-community/levanter-base
ARG TAG=latest
FROM ${IMAGE}:${TAG}
```

Reason:

- the Levanter base image is already known to work inside Docker on TPU VMs,
- Tunix's own docs mostly assume "install on the VM directly", not "run inside
  Docker on TPU VM",
- reusing the known-good TPU container base removes the largest integration
  risk.

### What the Tunix Dockerfile should do

At a high level:

1. start from the TPU-ready base image,
2. set `WORKDIR /opt/tunix`,
3. copy the Tunix repo into the image,
4. create or reuse a Python environment,
5. install Tunix with TPU deps,
6. install a few extra packages that Tunix commonly expects,
7. set `WORKDIR` to the Tunix repo root.

Sketch:

```dockerfile
ARG IMAGE=ghcr.io/marin-community/levanter-base
ARG TAG=latest
FROM ${IMAGE}:${TAG}

ENV PATH=/opt/tunix/.venv/bin:$PATH \
    HOME=/home/tunix

RUN mkdir -p /opt/tunix /home/tunix
WORKDIR /opt/tunix

ADD . /opt/tunix

RUN python3 -m venv /opt/tunix/.venv
RUN . /opt/tunix/.venv/bin/activate && pip install --upgrade pip
RUN . /opt/tunix/.venv/bin/activate && pip install -e ".[prod]"
RUN . /opt/tunix/.venv/bin/activate && pip install gcsfs

WORKDIR /opt/tunix
```

### Why not build Tunix by cloning GitHub inside the Docker build?

Because building from the local clone is cleaner:

- the exact Tunix commit is inspectable before build,
- you can pin to a specific SHA,
- you can carry local patches if needed,
- Docker builds stay deterministic relative to the checkout.

The launcher can still update the clone first:

```bash
git fetch origin
git checkout <ref>
git pull --ff-only
```

## Proposed Launcher Changes

### `infra/launch_tunix.py`

Start from Levanter's `infra/launch.py`, but change these behaviors:

1. make the Dockerfile path Tunix-specific:

```python
docker_file = repo_root / "docker" / "Dockerfile.tpu"
```

2. default image names should be Tunix-specific:

```python
--image_name tunix-<user>
--docker_repository tunix
```

3. default `WANDB_PROJECT` should be `tunix` or a caller-provided override.

4. set `GIT_COMMIT` from the Tunix repo's git SHA, not Marin's.

5. keep the rest unchanged:

- build image
- push image
- create queued TPU resource
- SSH into TPU VM
- run Docker command
- retry/autodelete if requested

### `infra/docker.py`

Mostly copy as-is from Levanter.

Recommended change:

- rename the persistent Docker volume from `levanter` to `tunix`

That affects `make_docker_run_command(...)` and the VM setup logic.

### `infra/tpus.py`

Mostly copy as-is from Levanter.

Recommended changes:

- create Docker volume `tunix` instead of `levanter`
- remove old container named `tunix` instead of `levanter`
- keep the rest identical

The queued-resource logic is already exactly what we want.

## Runtime Command Strategy

### First validation target: SFT or GRPO, not DPO

Use one of these as the first real job:

- SFT: `python3 -m tunix.cli.peft_main ...`
- GRPO: `python3 -m tunix.cli.grpo_main ...`

Recommended first smoke tests:

1. SFT on Qwen 2.5 0.5B:
   `examples/sft/mtnt/run_qwen2.5_0.5b.sh`
2. GRPO on Llama 3.2 1B:
   `examples/rl/grpo/gsm8k/run_llama3.2_1b.sh`

Why these:

- they use Hugging Face rather than Kaggle Gemma paths,
- they are smaller and easier to validate than bigger Gemma or multi-backend
  runs,
- they exercise the official CLI entrypoints.

### Use explicit module invocation once the launcher works

After the first successful run, stop depending on the example shell scripts and
invoke the module directly. It is more explicit and less fragile.

Recommended GRPO form:

```bash
python3 -m tunix.cli.grpo_main \
  tunix/cli/base_config.yaml \
  model_config.model_name="llama3.2-1b" \
  model_config.model_id="meta-llama/Llama-3.2-1B" \
  model_config.model_source="huggingface" \
  model_config.intermediate_ckpt_dir="/tmp/intermediate_ckpt/1" \
  model_config.mesh.shape="(2,4)" \
  model_config.mesh.axis_names="('fsdp','tp')" \
  actor_model_config.lora_config.rank=64 \
  actor_model_config.lora_config.alpha=64.0 \
  tokenizer_config.tokenizer_path="meta-llama/Llama-3.2-1B" \
  tokenizer_config.tokenizer_type="huggingface" \
  batch_size=1 \
  num_batches=100 \
  rl_training_config.max_steps=100 \
  rollout_engine="vanilla" \
  offload_to_cpu=false \
  grpo_config.num_generations=4 \
  reward_functions="['tunix/cli/reward_fn/gsm8k.py']"
```

Note the explicit use of `tunix/cli/base_config.yaml`. This avoids relying on
the CLI's compatibility fallback for bare `base_config.yaml`.

## Environment Variables To Pass Through

At minimum, expect to pass these through `launch_tunix.py -e KEY VALUE`:

- `HF_TOKEN`
- `KAGGLE_USERNAME`
- `KAGGLE_KEY`
- `WANDB_API_KEY`
- `WANDB_PROJECT`

Potentially also:

- `GOOGLE_CLOUD_PROJECT`
- `XLA_FLAGS`
- `JAX_TRACEBACK_FILTERING`

Tunix's config loader also reads `.env`, but for TPU jobs launched through the
container it is safer to pass secrets explicitly or bake a `.env` into the
build context only if you are comfortable with that risk.

## Paths And Persistence Inside The Container

Use host-backed `/tmp` for fast scratch and keep outputs there initially. This
matches both Levanter's Docker launch model and Tunix's example scripts.

Recommended initial paths:

- model download cache: `/tmp/models/...`
- intermediate conversion cache: `/tmp/intermediate_ckpt/...`
- TensorBoard logs: `/tmp/tensorboard/...`
- checkpoints: `/tmp/checkpoints/...`

The current Levanter TPU launcher already bind-mounts `/tmp:/tmp`, so this
fits naturally.

## Two-Phase Delivery Plan

### Phase 1: Prove The Infrastructure

Deliverable:

- local Tunix clone with launcher overlay,
- Tunix Dockerfile,
- ability to build and push image,
- ability to start TPU VM and run a trivial command in the container.

Acceptance check:

```bash
python infra/launch_tunix.py \
  --zone us-central1-a \
  --tpu_name tunix-smoke \
  --tpu_type v5p-32 \
  --capacity_type on-demand \
  --foreground \
  -- \
  python3 -c "import jax; print(jax.devices())"
```

If that fails, stop there and fix Docker/TPU/container integration before
touching Tunix training.

### Phase 2: Prove Tunix Training

Deliverable:

- successful SFT or GRPO Tunix run on TPU.

Recommended first real command:

```bash
python infra/launch_tunix.py \
  --zone us-central1-a \
  --tpu_name tunix-grpo \
  --tpu_type v5p-32 \
  --capacity_type on-demand \
  -e HF_TOKEN "$HF_TOKEN" \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- \
  python3 -m tunix.cli.grpo_main \
    tunix/cli/base_config.yaml \
    model_config.model_name="llama3.2-1b" \
    model_config.model_id="meta-llama/Llama-3.2-1B" \
    model_config.model_source="huggingface" \
    model_config.intermediate_ckpt_dir="/tmp/intermediate_ckpt/1" \
    tokenizer_config.tokenizer_path="meta-llama/Llama-3.2-1B" \
    tokenizer_config.tokenizer_type="huggingface" \
    batch_size=1 \
    num_batches=100 \
    rl_training_config.max_steps=100 \
    reward_functions="['tunix/cli/reward_fn/gsm8k.py']"
```

Only after this works should we decide whether the team wants:

- direct Tunix CLI launches,
- wrapper shell scripts,
- or a custom DPO entrypoint.

## Phase 3: Add DPO Only If We Actually Need It

If the real end goal is DPO-like training, add a small Tunix-side entrypoint,
for example:

```text
scripts/run_dpo.py
```

That script should:

1. construct the model and tokenizer,
2. create mesh and optimizer,
3. instantiate `tunix.sft.dpo.dpo_trainer.DPOTrainer`,
4. load data,
5. call `train(...)`.

This should be done only after Phase 2 succeeds. Otherwise we will not know
whether a failure is caused by:

- TPU Docker runtime,
- Tunix install,
- CLI/config assumptions,
- or new DPO wrapper code.

## Alternative Path: Keep Everything In Marin

This is possible, but I do **not** think it is the best first move.

What it would look like:

- add `infra/launch_tunix.py` in `lib/levanter`,
- add `docker/tpu/Dockerfile.tunix`,
- clone Tunix into a sibling path or pull it during Docker build,
- special-case the build context and repo-root lookup.

Why I do not recommend it first:

- it couples Tunix infrastructure to Marin unnecessarily,
- the current Docker build is workspace-specific,
- every change will have extra path and context handling,
- it is harder to reason about which git SHA is actually in the image.

Use this only if there is a strong desire to keep one launcher codebase for
everything.

## Concrete Implementation Sequence

1. Clone Tunix into a stable working directory, not `.cache`.
2. Create a local branch like `local/tpu-launcher`.
3. Copy `launch.py`, `docker.py`, `tpus.py`, and `cli_helpers.py` from
   Levanter into Tunix under `infra/`.
4. Add `docker/Dockerfile.tpu`.
5. Rename Levanter-specific container and volume names to `tunix`.
6. Point `launch_tunix.py` at `docker/Dockerfile.tpu`.
7. Keep the Docker base image as the known-good TPU base image from Levanter.
8. Add one minimal smoke-test command for `jax.devices()`.
9. Add one minimal Tunix SFT or GRPO command.
10. Only then decide whether to add a DPO wrapper.

## Final Recommendation

The highest-probability, lowest-drama plan is:

- **clone Tunix into its own real working directory**
- **cargo-cult Levanter's TPU launcher into that clone**
- **use a Tunix-specific Dockerfile built from a known-good TPU base image**
- **validate with Tunix SFT or GRPO first**
- **add DPO support later only if we actually need it**

That gets us to a working TPU launch path quickly, keeps Tunix updates easy,
and avoids overfitting the solution to Marin's workspace layout.
