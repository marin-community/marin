# Midtraining redesign logbook

> ## Policy Update 2026-05-16 — CPT Uses Fractional Warmup And Triangular Decay
>
> **Every CPT launch warms up over 10% of its own `num_train_steps` and
> then decays over the full post-warmup remainder.** This is the
> load-bearing default; cell authors override at their own risk.
>
> - Constants live in `lib/marin/src/marin/midtraining/modes.py`:
>   `CPT_DEFAULT_WARMUP_FRACTION = 0.10`,
>   `CPT_DEFAULT_DECAY = None`.
> - Re-exported from `marin.midtraining` as the public surface.
> - The new 3e18 K=0.20 sweep
>   (`experiments/midtrain_specs/delphi_small_cpt_k020.py`) passes these
>   constants directly into `AdamHConfig` instead of reading
>   `base.warmup_fraction` / `base.decay_fraction`. Pretrain-derived
>   warmup/decay are explicitly NOT used for CPT.
>
> **Why this changed.** The original draft said "warmup uses a token budget
> policy, not a magic step count," referring to the legacy
> `experiments/exp_delphi_math_10b_midtrain.py` rule of a fixed 1.05B-token
> warmup budget. That warmup rule does not generalize:
>
> | Base | Batch | Total midtrain steps (K=0.20) | Legacy warmup steps (1.05B tokens) | Warmup as % of run |
> |---|---:|---:|---:|---:|
> | 1e21 | 512 | 4,411 | 500 | **~11%** |
> | 1e22 | 1024 | 7,635 | 250 | **~3%** |
> | 1e23 | 2048 | ~30,000 | 125 | **~0.4%** |
> | 3e18 (new K=0.20 sweep) | 8 | 7,400 | would be 32,000 (capped to 7,399) | **degenerate — entire run becomes warmup** |
>
> At small scales the fixed-token rule degenerates entirely (warmup
> exceeds the whole run). At large scales it leaves the LR ramp far
> shorter than the Delphi pretrain convention. Neither tail is what we
> want.
>
> Switching warmup to a fraction-of-run rule fixes both ends and matches
> what the original 1e21 anchor happened to use (11% ≈ 10%), so the
> reference scale is unchanged in practice. The decay rule intentionally
> follows the legacy CPT launcher, not the Delphi pretrain WSD shape:
> `exp_delphi_math_10b_midtrain.py` used
> `decay_steps = num_train_steps - warmup_steps`, so there was no stable
> middle. In Levanter terms, setting `warmup=0.10` and `decay=None`
> expresses the same triangular warmup -> linear decay profile because
> `None` is the scheduler sentinel for "decay over the full post-warmup
> remainder." Setting `decay=0.20` would create WSD: 10% warmup,
> 70% stable, 20% decay.
>
> **Operational consequence.** The 3e18 sweep currently running
> (launched 2026-05-16 16:25Z) was already using 10% warmup
> *coincidentally* — `DelphiModel.warmup_fraction` defaulted to 0.1 and
> the cell author forwarded `base.warmup_fraction` straight into
> `AdamHConfig`. But those cells also inherited `base.decay_fraction = 0.2`,
> which Levanter renders as WSD. That is a bug in the launched 3e18 sweep.
> Future bases/cells must use the explicit constants so warmup is fractional
> and decay covers the whole post-warmup remainder.
>
> See `.agents/logbooks/midtraining_delphi.md` for the operational record
> of the 3e18 WSD incident.

# codex 5.5 2026-05-16T00:34:42Z

This logbook is a redesign brief for Delphi midtraining after the April-May
2026 incidents in `.agents/logbooks/midtraining_delphi.md` and
`.agents/logbooks/true_midtraining.md`.

The target design is one small, explicit midtraining API that supports two
separate semantics:

- `cpt`: continued pretraining from base weights. This loads model weights only
  and starts a fresh optimizer state, fresh scheduler count, and fresh data
  iterator.
- `cooldown`: true midtraining by resuming a pretrain checkpoint. This restores
  model, optimizer state, scheduler count, and absolute step through Levanter's
  natural checkpoint recovery path, while swapping only the data config.

The current experiment files grew by accretion. They now contain good guards,
but too much state is still spread across env vars, executor hashes, GCS path
conventions, W&B behavior, and logbook memory. The redesign should make the
launch contract representable in one typed spec and one generated manifest.

## Non-negotiable design goals

1. One logical cell per launch. No file should call `executor_main` with a list
   of training cells.
2. One run identity per cell. `output_path`, Levanter `RUN_ID`, W&B `run_id`,
   permanent checkpoint path, temporary checkpoint path, and manifest path must
   derive from the same explicit `RunIdentity`.
3. No semantic identity from physical placement. Region, bucket, TPU slice, and
   cache mirror location must not affect the run id.
4. Base model selection only comes from `experiments/delphi_models.py` or a
   similarly typed registry. No grepping GCS for model names.
5. Resume must prove itself before training. A resumed run must find the prior
   permanent or temporary checkpoint and must log the exact old run id before
   any step runs.
6. CPT and cooldown must be separate mode implementations, not boolean flags on
   one ambiguous path.
7. W&B must never silently create a second row for what the operator thinks is
   a resume.
8. Dataset caches must be preflighted as data artifacts, not rebuilt or
   region-rehashed during a training launch.
9. Defaults are allowed only if they are named fields in the spec, rendered into
   the manifest, and logged before launch.
10. Escape hatches require explicit names, explanations, and run-name suffixes.

## Incident ledger from the old system

This section lists the mistakes and footguns we actually hit. Some are already
fixed in code. Some remain as design debt.

| Failure | What happened | Fix already landed | Redesign requirement |
|---|---|---|---|
| Wrong "1e20 Delphi" base | The historical 1e20 rows used `isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5`, a deprecated v5 isoflop ablation, not a Delphi base. | `experiments/delphi_models.py` bans `adamh_scaling_v5`, has no `DELPHI_1E20`, and raises on `get_delphi_model("1e20")`. Launchable 1e20 rows were removed from math and true-midtrain files. | `MidtrainSpec.base` must be a `DelphiModel` registry object, not a string path. Unknown scales fail closed. |
| `v5` name trap | `adamh_scaling_v5` means deprecated heuristic, while `adamh-scaling-ladder-...-v5-...` is an experiment iteration tag for a v6-heuristic headline run. | Registry docstring and postmortem explain the distinction. | Base registry must carry explicit `recipe_label`, `run_iteration`, and `source_kind`; UI/rendered manifest must show them separately. |
| Multi-cell `executor_main` collision | `exp_delphi_math_10b_midtrain.py` generated many `ExecutorStep`s. All training children used the hardcoded Iris child name `train_lm`, so concurrent calls adopted the same child job. Five cells were marked terminal without training artifacts. | Env selectors were added so operators launch one cell per coordinator. True-midtrain added a real no-selector launch guard. | Unified launcher must not have a code path that submits more than one training cell. Multi-cell sweeps are a shell/driver loop over one-cell launches. |
| Current math no-selector footgun | The math file warns not to run the full list, but its `__main__` still calls `executor_main(steps=runs)` with all generated runs when selectors are absent. | Not fixed as of this redesign note. | Real launch requires an explicit cell id. Dry-run enumeration is a separate command, e.g. `midtrain plan --all`, that never calls training. |
| Invalid selector builds no runs | A typo like `MIDTRAIN_SELECT_BASE=1e21` can filter every run out and produce an empty run list. | Not fixed as of this redesign note. | Selector parsing must validate against enum values before build. Empty launch plans are errors. |
| MirrorFS checkpoint gap | `mirror://` paths bypassed region checks but TensorStore could not open `mirror://` checkpoints. | Levanter staging helper materializes `mirror://` checkpoints to a concrete local-region path before TensorStore opens them. | The new API should separate `logical_checkpoint` from `resolved_checkpoint`. Resolution happens in preflight and is recorded in the manifest. |
| Flat LR from warm-start | `initialize_from_checkpoint_path` restored full opt_state but reset only outer `state.step` to zero. The scheduler count stayed at the pretrain step, so short midtrain runs trained at the LR floor. | `CheckpointInitMode.MODEL_ONLY` was added. CPT sets it explicitly and tests verify propagation through `default_train`. | CPT mode must use a dedicated `ModelOnlyInit` branch. Cooldown mode must forbid `initialize_from_checkpoint_path`. |
| Full-state init is not true resume | Using `initialize_from_checkpoint_path` with `FULL_STATE` restores optimizer count but resets outer step, causing either too many new steps or a clamped schedule. | True-midtrain uses natural resume: pre-stage checkpoint under `output_path/checkpoints/step-N` and leave `initialize_from_checkpoint_path=None`. | Cooldown mode must only support natural resume. If the checkpoint is not under the output namespace, launch refuses. |
| Executor hash ignored important fields | Adding `checkpoint_init_mode=MODEL_ONLY` did not change output paths because the executor hash only saw `versioned(...)` values and dependencies, not plain dataclass fields. Broken and fixed runs reused the same output hash. | Manual run-name suffix bumps were used. Tests cover init-mode propagation, not executor hashing. | Output path must be explicit, not inferred from executor hashing. A separate full config digest is recorded for audit, but it does not secretly choose the run id. |
| W&B monotonic-step rejection | Because fixed reruns reused old W&B run ids, W&B rejected step 1 metrics when the old run had already logged step 4768. | Fresh suffixes or explicit resume identity avoided this later. | W&B preflight must query the target run id. Fresh CPT refuses if a W&B run already exists unless `attempt` is incremented. Resume requires existing run id plus checkpoint proof. |
| W&B project hard-coded | `default_train` always wrote to project `marin`; `WANDB_PROJECT` env alone did not work because config passed an explicit project. | `default_train(..., wandb_project=...)` was added; Delphi passes `delphi-midtraining`. | `wandb_project`, `wandb_entity`, `wandb_run_id`, and `wandb_resume_policy` are required manifest fields. |
| Shared temporary checkpoint namespace | Executor-backed jobs had unique permanent paths but shared a temp checkpoint root. Later jobs picked up incompatible temp checkpoints and failed before model init. | Temp checkpoint root now includes the output basename. A regression test covers executor-scoped temp checkpoints. | Temp path must be a pure function of `RunIdentity.output_path`. Preflight prints permanent and temp search paths. |
| Resume namespace drift | A failed run was relaunched with a similar human-readable step name but a different executor hash. It started a new checkpoint namespace and W&B run instead of resuming. | `MIDTRAIN_RESUME_OUTPUT_PATH` derives output path, `RUN_ID`, and `WANDB_RUN_ID`; legacy `MIDTRAIN_OUTPUT_PATH_OVERRIDE` is rejected. | Resume CLI takes exactly one identity input: `--resume-output-path`. Manual `RUN_ID` and `WANDB_RUN_ID` are not accepted. |
| Parent retry changed region and hash | Parent recovery in a different region caused dataset dependency paths to change from `gs://marin-us-central1/...` to `gs://marin-us-east5/...`, changing the training hash and losing progress. | Mainline #5223 canonicalized region prefixes in executor identity. The logbook top warning requires forcing old output paths on relaunch. | Avoid executor-derived output identity. Dataset cache manifests use logical ids and content/cache fingerprints, never physical bucket names. |
| StepSpec dataset identity leaked physical paths | Nemotron datakit normalizer stored `download.output_path` as a hash attr after it had resolved to a regional GCS path. | Region-stable executor hashing reduced this risk. | The midtrain launcher should consume prebuilt data cache manifests, not live StepSpec graphs, during training launch. |
| Stale BOS-missing cache | A us-central1 `4plus` tokenized cache predated the BOS fix and could be silently reused. | Bad cache deleted; cache rebuilt and sampled to verify Llama-3 BOS `128000`. | Data preflight records tokenizer id, BOS/EOS sample, cache length, and cache path. A launch refuses unknown or unverified cache manifests. |
| `auto_build_caches=False` was partial protection | Levanter cache auto-build was off, but Marin executor dependencies could still rebuild normalize/tokenize upstream. | Logbook documents this. | Training launch must not include data build steps. Cache build is a separate command with its own manifest and verification. |
| Preflight val check hit unresolved `VersionedValue` | Calling `validation_sets()` on full mixes before executor materialization raised `TypeError`. | Preflight checks the math-only config and catches only the known unresolved-wrapper `TypeError`; Layer 1/2 validations run at import. | Data manifests and mix specs should be resolved before launch. Runtime preflight should not depend on executor materialization. |
| Legacy single math path bypasses val safety | Leaving `MIDTRAIN_MIX_NAME` unset preserves old single-step data path and skips val/train disjointness checks. | The file logs a warning. | Real launches require a `MidtrainMixSpec`; legacy raw dataset mode requires `allow_unsafe_no_val_split=True` plus a run-name suffix. |
| Hard-coded eval cadence | Math used fixed `steps_per_eval=200`, giving shorter cells fewer validation points. | Math now uses `_steps_per_eval(num_train_steps)` targeting about 40 evals per run. | Eval cadence is a policy object: fraction, min, max, and basis (`total_steps` for CPT, `remaining_steps` for cooldown) are explicit fields. |
| True-midtrain eval cadence still fixed | `exp_delphi_true_midtrain.py` still has `STEPS_PER_EVAL = 200`. | Not fixed as of this note. | Cooldown eval cadence should use remaining steps, not full pretrain length. |
| Permanent checkpoint cadence was fixed | Shorter runs had fewer rollback points. | Math uses about 10% of run length; true-midtrain uses a similar helper. | Checkpoint cadence is a policy object and is included in W&B tags and manifest. |
| `MIDTRAIN_INIT_CKPT_PATH` override | Math can silently override the registry checkpoint while keeping official run names. | Not fixed as of this note. | Alternate init checkpoint is a typed `CheckpointOverride` requiring selector, run suffix, banned-path check, shape check, and reason. |
| `MIDTRAIN_TOKEN_BUDGET` hard override | A global absolute budget can apply to every selected base despite being a legacy single-base path. | Not fixed as of this note. | Budgets are `ByFraction`, `ByTokens`, or `BySteps`. `ByTokens` requires one selected base and a budget label. |
| Compute override validation is thin | Batch size, per-device parallelism, and tensor parallelism are env vars with limited validation. | Some TPU allowlists exist per base. | Compute config is a typed object with positive/divisibility checks and per-base allowlists. Unknown overrides fail before launch. |
| JAX coordinator port 8476 collisions | v5p-64/v5p-256 jobs repeatedly died from stale coordinator/placement collisions, sometimes with 707 preemption fingerprints. | `MIDTRAIN_MAX_TASK_FAILURES=100` was plumbed through Iris/Fray/Marin and validated on v5p-64 resumes. | Launch profile includes `max_task_failures`, default 100 for midtraining TPU jobs, and optional serialized submission delay. |
| Preemption recovery sometimes failed as framework failure | SIGTERM on one worker could cascade as JAX RPC errors and parent failure, not a clean preemption. | Logbook playbook now treats these as relaunchable with checkpoint proof. | Babysitter classifies failures by checkpoint state, not by Iris label alone. |
| GCP v5p capacity eviction | v5p-256/v5p-512 pool disappeared and left 1e22 cells unfinished. | Recovery paths and v5p-64 allowlist were added. | Compute profile contains fallback TPU shapes and throughput estimates; resuming on fallback requires manifest update, not ad hoc env. |
| HF export config caveats | HF dirs may say `LlamaForCausalLM` for Qwen3 weights and include rope fields that trip transformers validation. | Handoff docs record local repair/override pattern. | Export validation must check config architecture, tokenizer, max position fields, and a small load smoke test before marking a cell inference-ready. |
| Stale logbook state | The true-midtraining logbook claimed nothing had launched after runs had actually run. | Current status blocks were corrected with W&B/GCS checks. | New launcher writes a machine-readable manifest and status file per cell; logbook status is generated or copied from those manifests. |

## What the unified API should look like

Prefer one library file plus one thin CLI:

- `experiments/midtrain.py`: typed specs, validators, config rendering, manifest
  generation, and direct Levanter launch helpers.
- `scripts/launch_midtrain.py`: CLI wrapper for `plan`, `stage`, `launch`,
  `resume`, `watch`, and `mark-finished`.
- Keep `experiments/delphi_models.py` and `experiments/midtraining_mixes.py` as
  source registries. Do not duplicate their data.

Avoid the executor framework for the training cell itself. It caused too many
identity and cache-status footguns. The new launcher should call Iris directly
with a rendered Levanter/Marin training config and an explicit output path. Data
cache construction can still use existing Marin/Zephyr tooling, but it must be a
separate pre-stage step that produces a cache manifest consumed by training.

### Core types

```python
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class MidtrainMode(StrEnum):
    CPT = "cpt"
    COOLDOWN = "cooldown"


class CheckpointSourceKind(StrEnum):
    NATIVE_LEVANTER = "native_levanter"
    HF_WEIGHTS = "hf_weights"


@dataclass(frozen=True)
class RunIdentity:
    name: str
    output_path: str
    wandb_project: str
    wandb_entity: str = "marin-community"
    attempt: str | None = None

    @property
    def run_id(self) -> str:
        return self.output_path.rstrip("/").split("/")[-1]


@dataclass(frozen=True)
class EvalPolicy:
    target_points: int = 40
    min_steps: int = 25
    max_steps: int = 200
    basis: str = "mode_default"  # CPT total steps, cooldown remaining steps


@dataclass(frozen=True)
class CheckpointPolicy:
    permanent_fraction: float = 0.10
    min_permanent_steps: int = 50
    temp_save_interval: str = "10m"
    hf_export_matches_permanent: bool = True


@dataclass(frozen=True)
class ComputeProfile:
    tpu_type: str
    batch_size: int
    per_device_parallelism: int
    tensor_parallel_size: int = 1
    regions: tuple[str, ...] = ("us-east5", "us-central1")
    priority: str = "batch"
    max_task_failures: int = 100
    launch_spacing_seconds: int = 30


@dataclass(frozen=True)
class DataCacheManifest:
    mix_name: str
    cache_paths: tuple[str, ...]
    tokenizer_id: str
    seq_len: int
    total_sequences: int | None
    total_tokens: int | None
    bos_token_id: int
    validation_fingerprint: str | None


@dataclass(frozen=True)
class CptInit:
    source_kind: CheckpointSourceKind
    checkpoint_path: str | None = None
    hf_repo: str | None = None
    hf_revision: str | None = None
    reset_optimizer: bool = True
    reset_data_loader: bool = True


@dataclass(frozen=True)
class CooldownResume:
    pretrain_checkpoint_path: str
    resume_step: int
    staged_output_path: str
    preserve_optimizer: bool = True
    preserve_scheduler_count: bool = True
    preserve_state_step: bool = True


@dataclass(frozen=True)
class MidtrainSpec:
    mode: MidtrainMode
    base_key: str
    run: RunIdentity
    data: DataCacheManifest
    compute: ComputeProfile
    eval: EvalPolicy
    checkpoints: CheckpointPolicy
    cpt_init: CptInit | None = None
    cooldown_resume: CooldownResume | None = None
    token_budget: int | None = None
    token_budget_fraction: float | None = None
    lr_factor: float | None = None
    lr_multiplier: float = 1.0
    min_lr_ratio: float = 0.0
```

Codex correction 2026-05-16: optimizer knobs must not live on `CptMode`
unless the renderer consumes them. The implemented code keeps optimizer
policy in `MidtrainSpec.optimizer_config`; `CptMode` owns only CPT init and
budget sizing.

The API should accept exactly one `MidtrainSpec` and return a `LaunchPlan`.
`LaunchPlan` contains the rendered training config, manifest, preflight checks,
Iris command, and expected startup log predicates.

```python
def midtrain(spec: MidtrainSpec) -> LaunchPlan:
    resolved = resolve_midtrain_spec(spec)
    validate_midtrain_spec(resolved)
    train_config = build_train_config(resolved)
    manifest = build_manifest(resolved, train_config)
    return LaunchPlan(spec=resolved, train_config=train_config, manifest=manifest)
```

## CPT mode contract

CPT means "continue pretraining from model weights only." It is the semantics of
`experiments/exp_delphi_math_10b_midtrain.py` after the optimizer-state fix.

Required fields:

- `mode = MidtrainMode.CPT`
- `base_key`: key in `experiments.delphi_models`
- `cpt_init`: source and checkpoint/repo information
- one of `token_budget` or `token_budget_fraction`
- `lr_factor` and optimizer policy
- `data`, `compute`, `eval`, `checkpoints`, `run`

Defaults, explicit in the spec:

- `reset_optimizer=True`
- `checkpoint_init_mode=MODEL_ONLY`
- `reset_data_loader=True`
- `num_train_steps = token_budget / (batch_size * seq_len)`
- ~~warmup uses a token budget policy, not a magic step count~~
  **SUPERSEDED 2026-05-16** — warmup is a fixed fraction of
  `num_train_steps` (10% warmup) and decay covers the whole remaining run
  (`decay=None` in Levanter), via
  `CPT_DEFAULT_WARMUP_FRACTION` / `CPT_DEFAULT_DECAY`. See the
  top-of-file POLICY DECISION callout for the rationale and the
  per-base step-count math. The legacy fixed-token rule degenerated for
  short CPT runs and undershot warmup at large batch; the legacy triangular
  post-warmup decay is preserved.
- eval cadence targets 40 points over `num_train_steps`
- permanent checkpoints every 10% of `num_train_steps`

Hard guards:

1. `cooldown_resume is None`.
2. `cpt_init.reset_optimizer is True`.
3. `checkpoint_init_mode` renders as `MODEL_ONLY`.
4. `initialize_from_checkpoint_path` is set only for native Levanter checkpoint
   sources. If HF weights are used, the source path must go through a validated
   HF-to-Levanter import or a direct Levanter HF load path with shape checks.
5. If `CheckpointSourceKind.HF_WEIGHTS` is used, `hf_revision` must be pinned,
   tokenizer id must match the base registry, and a model-shape smoke test must
   run before launch.
6. `RunIdentity.output_path` must not already contain checkpoints unless this is
   an explicit same-run resume of a prior CPT attempt.
7. W&B must not already have `run_id` unless same-run resume is intended.

HF is acceptable for CPT because optimizer state is intentionally discarded.
Native Levanter checkpoints remain the safer default because they avoid HF
conversion ambiguity and preserve exact architecture metadata. HF should be a
configured source, not an implicit fallback.

## Cooldown mode contract

Cooldown means "true midtraining": pick up a pretrain checkpoint at a chosen
absolute pretrain step, keep optimizer state and scheduler count, and change the
data distribution.

Required fields:

- `mode = MidtrainMode.COOLDOWN`
- `base_key`
- `cooldown_resume`
- `run.output_path == cooldown_resume.staged_output_path`
- original pretrain `num_train_steps`
- original pretrain optimizer config
- `data`, `compute`, `eval`, `checkpoints`, `run`

Defaults, explicit in the spec:

- `initialize_from_checkpoint_path=None`
- `checkpoint_init_mode=FULL_STATE` only as documentation; it is moot because
  the init path is forbidden
- `num_train_steps = base.num_train_steps`
- eval cadence basis is `remaining_steps = num_train_steps - resume_step`
- permanent checkpoint cadence can use either remaining steps or full run
  length; choose one policy field and render it
- W&B `resume="allow"` with `run_id=basename(output_path)`

Hard guards:

1. `cpt_init is None`.
2. `initialize_from_checkpoint_path is None`.
3. `cooldown_resume.resume_step` must equal the staged checkpoint step exactly.
4. The staged checkpoint must exist at
   `<output_path>/checkpoints/step-<resume_step>`.
5. The staged checkpoint must include `manifest.ocdbt`, `metadata.json`, and
   `d/`.
6. The latest checkpoint under permanent or temp paths must be at least
   `resume_step`. For first launch, it should equal `resume_step`; for recovery,
   it can be greater.
7. Startup logs must contain `Resuming training from step <N>`, where
   `N >= resume_step`. If startup says `Starting from scratch`, the babysitter
   kills the job.
8. `num_train_steps` must equal the original pretrain target. Setting it to the
   remaining tail recreates the flat-LR bug.
9. The optimizer config must match the pretrain heuristic fields recorded in
   the base registry: warmup, decay, min LR ratio, beta2, epsilon, peak LR, and
   peak Adam LR.

Cooldown should not load from HF. HF weights do not contain optimizer state,
scheduler count, or trainer step.

## Launch state machine

The CLI should expose five concrete stages. Each stage reads and writes a
manifest so humans and agents do not infer state from memory.

### `plan`

Inputs: a TOML/YAML/Python spec for one cell.

Outputs:

- rendered `MidtrainSpec`
- resolved base registry entry
- rendered training config
- deterministic `RunIdentity`
- permanent checkpoint path
- temporary checkpoint path
- W&B run id
- expected startup predicates
- full config digest

No GCS writes and no Iris jobs happen in `plan`.

### `stage`

CPT:

- Verify base checkpoint or HF source exists.
- If HF is selected, download/convert once into a named staging path and record
  the pinned revision.
- Verify model shapes and tokenizer id.

Cooldown:

- Copy the pretrain checkpoint into
  `<output_path>/checkpoints/step-<resume_step>`.
- Verify TensorStore files and metadata.
- Write `midtrain_manifest.json` next to the output path before launch.

Data:

- Verify every cache path in `DataCacheManifest`.
- Sample BOS/EOS.
- Verify validation partition if present.
- Refuse to launch if data build would be required.

### `launch`

Launch exactly one cell.

The command should be direct Iris, not executor:

```bash
uv run iris --cluster=marin job run \
  --cpu 1 --memory 3GB --disk 9GB \
  --region us-east5 \
  --priority interactive \
  --no-preemptible \
  --job-name <run-name>-<timestamp> \
  --no-wait \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python scripts/run_midtrain_cell.py --manifest <manifest-path>
```

The outer coordinator is a long-lived CPU parent and must not run on
preemptible capacity. If it dies, Iris cascade-kills the TPU child. The
training child submitted by the launcher remains the preemptible TPU job.

`scripts/run_midtrain_cell.py` reads the manifest, constructs the training
config, sets `RUN_ID` and W&B identity from `RunIdentity`, and launches the
training job. It does not compute output paths from a dependency graph.

### `watch`

The babysitter checks:

- Iris parent state
- training child state
- W&B run id and current step
- permanent latest checkpoint
- temporary latest checkpoint
- expected startup log lines
- learning-rate sanity at first eval

For CPT, first W&B metrics should start at step 0/1 and LR should warm up from
zero. For cooldown, first metric step should be at or above the resume step and
LR should match the pretrain WSD schedule at that absolute step.

### `resume`

Resume accepts only:

```bash
scripts/launch_midtrain.py resume \
  --output-path gs://marin-<region>/checkpoints/<old-run-id> \
  --expected-min-step <N>
```

It discovers the manifest from the old output path, checks permanent and temp
checkpoints, checks W&B, and relaunches with the exact old `RunIdentity`.

It does not accept manual `RUN_ID`, manual `WANDB_RUN_ID`, or a new output path.

## Identity and W&B rules

Use one identity source:

```text
run_id = basename(output_path)
WANDB_RUN_ID = run_id
RUN_ID = run_id
permanent_checkpoints = output_path/checkpoints
temporary_checkpoints = tmp_root/<region>/checkpoints-temp/<run_id>
manifest = output_path/midtrain_manifest.json
```

Fresh launch rules:

- If `output_path` exists with a manifest, refuse unless `--resume` is used.
- If W&B has `run_id`, refuse unless `--resume` is used.
- If permanent or temp checkpoints exist, refuse unless `--resume` is used.

Resume rules:

- `output_path` must exist.
- Manifest must exist and match the requested mode.
- W&B run id must be the output basename.
- Latest checkpoint must be at or above the expected floor.
- Startup logs must confirm the same run id and output path.

Attempt rules:

- A new attempt that intentionally starts over must use a different
  `RunIdentity.name` or `attempt` suffix.
- Attempts are recorded in the manifest, not hidden in W&B display names.

## Data rules

Training launch should consume data manifests, not StepSpec graphs.

Each `DataCacheManifest` should include:

- logical dataset names
- cache paths per region
- tokenizer id
- seq length
- cache hash or content fingerprint
- total sequences/tokens when known
- BOS/EOS sample result
- validation carve-out policy
- validation partition fingerprint when available
- mix weights

Real launch refuses:

- unknown cache paths
- cache path in the wrong tokenizer
- missing BOS for Llama-3/Qwen3 tokenizers
- weights that do not sum to 1
- midtrain/pretrain component name collisions
- `shuffle_before_trainval_split=False`
- unsafe legacy raw dataset mode without explicit override
- cache auto-build during training

This deliberately moves cache building out of training. A failed training launch
should never spend 35 minutes tokenizing because a cache hash changed.

## Configuration policy

Use a config file, not a pile of env vars.

Allowed env vars:

- `WANDB_API_KEY`
- cluster connection settings
- explicit debug flags for dry-run output verbosity

Everything else is in `MidtrainSpec`:

- base
- mode
- mix
- budget
- LR policy
- optimizer state policy
- checkpoint source
- output path
- W&B identity
- compute profile
- eval cadence
- checkpoint cadence
- resume step
- max task failures

Unknown env vars with prefix `MIDTRAIN_`, `TRUE_MIDTRAIN_`, `RUN_ID`, or
`WANDB_RUN_ID` should cause launch failure unless they are explicitly allowed by
the wrapper. This prevents a stale shell from changing a cell.

## Example CPT spec

```python
spec = MidtrainSpec(
    mode=MidtrainMode.CPT,
    base_key="1e21",
    run=RunIdentity(
        name="delphi-1e21-p33m67-9p25b-lr0.5-cpt",
        output_path="gs://marin-us-east5/checkpoints/delphi-1e21-p33m67-9p25b-lr0.5-cpt",
        wandb_project="delphi-midtraining",
    ),
    data=DataCacheManifest.from_registry("pretrain_33p_math_67p_highquality_nemo_math"),
    compute=ComputeProfile(
        tpu_type="v5p-64",
        batch_size=512,
        per_device_parallelism=-1,
    ),
    eval=EvalPolicy(target_points=40, min_steps=25, max_steps=200),
    checkpoints=CheckpointPolicy(permanent_fraction=0.10),
    cpt_init=CptInit(
        source_kind=CheckpointSourceKind.NATIVE_LEVANTER,
        checkpoint_path=DELPHI_1E21.verified_checkpoint_path,
    ),
    token_budget_fraction=0.20,
    lr_factor=0.5,
    min_lr_ratio=0.0,
)
```

Rendered training config must include:

- `initialize_from_checkpoint_path = DELPHI_1E21.verified_checkpoint_path`
- `checkpoint_init_mode = MODEL_ONLY`
- `reset_data_loader_on_init = True`
- fresh AdamH schedule derived from the CPT budget
- `num_train_steps` equal to CPT steps, not pretrain steps

## Example cooldown spec

```python
spec = MidtrainSpec(
    mode=MidtrainMode.COOLDOWN,
    base_key="1e22",
    run=RunIdentity(
        name="true-midtrain-1e22-p33m67-step30000-v5p64",
        output_path="gs://marin-us-east5/checkpoints/true-midtrain-1e22-p33m67-step30000-v5p64",
        wandb_project="delphi-midtraining",
    ),
    data=DataCacheManifest.from_registry("pretrain_33p_math_67p_highquality_nemo_math"),
    compute=ComputeProfile(
        tpu_type="v5p-64",
        batch_size=1024,
        per_device_parallelism=4,
    ),
    eval=EvalPolicy(target_points=40, min_steps=25, max_steps=200, basis="remaining_steps"),
    checkpoints=CheckpointPolicy(permanent_fraction=0.10),
    cooldown_resume=CooldownResume(
        pretrain_checkpoint_path=DELPHI_1E22.levanter_checkpoint_path(step=30_000),
        resume_step=30_000,
        staged_output_path="gs://marin-us-east5/checkpoints/true-midtrain-1e22-p33m67-step30000-v5p64",
    ),
)
```

Rendered training config must include:

- `initialize_from_checkpoint_path = None`
- no model-only init branch
- original pretrain `num_train_steps`
- original pretrain AdamH schedule
- output path already containing `checkpoints/step-30000`
- W&B and Levanter run id equal to `true-midtrain-1e22-p33m67-step30000-v5p64`

## Preflight checklist

The launcher should print and save this checklist before it submits Iris:

| Check | CPT | Cooldown |
|---|---|---|
| Base comes from registry | required | required |
| Banned substrings absent | required | required |
| Tokenizer matches model | required | required |
| Data cache manifest exists | required | required |
| BOS sample passes | required | required |
| Val/train disjointness passes or is explicitly disabled | required | required |
| Output path empty for fresh launch | required | not allowed; must contain staged checkpoint |
| W&B run absent for fresh launch | required | not required |
| W&B run id equals basename(output_path) | required | required |
| Permanent checkpoint search path printed | required | required |
| Temporary checkpoint search path printed | required | required |
| Init checkpoint exists | required | forbidden |
| Staged resume checkpoint exists | forbidden | required |
| Optimizer state reset | required | forbidden |
| Original optimizer state restored | forbidden | required |
| `num_train_steps` basis validated | CPT budget | pretrain target |
| Eval/checkpoint cadence rendered | required | required |
| TPU shape allowed for base | required | required |
| `max_task_failures` rendered | required | required |

## Startup proof

Do not mark a launch healthy until logs show the expected mode.

CPT fresh expected lines, native Levanter init:

```text
Using output path <output_path>
Using run ID <basename(output_path)>
No checkpoints found in [<output_path>/checkpoints, <temp_path>]
Loading checkpoint from <base checkpoint>
checkpoint_init_mode=MODEL_ONLY
```

CPT fresh expected lines, HF init:

```text
Using output path <output_path>
Using run ID <basename(output_path)>
No checkpoints found in [<output_path>/checkpoints, <temp_path>]
No training checkpoint found. Initializing model from HF checkpoint <repo>@<revision>
```

Cooldown / same-run resume expected lines:

```text
Using output path <output_path>
Using run ID <basename(output_path)>
Discovered latest checkpoint at <output_path>/checkpoints/step-<N> or <temp_path>/step-<N>
Loading checkpoint from <discovered training checkpoint>
Resuming training from step <N>
```

Forbidden lines:

```text
Starting from scratch
Step <new> is less than current W&B step <old>
No checkpoints found ...  # cooldown / same-run resume only
```

If any forbidden line appears, the babysitter should kill the job and mark the
manifest `failed_preflight_runtime`.

## Analysis and handoff rules

Every finished cell should write:

- final manifest with status
- W&B URL
- latest permanent checkpoint
- latest temp checkpoint if unfinished
- final HF export path
- eval cadence and checkpoint cadence
- model config validation result
- known inference overrides, if any

Analysis scripts should consume these manifests instead of scraping free-form
logbook tables. The logbook can summarize status, but it should not be the only
source of truth.

Historical 1e20 rows must be labeled as v5-isoflop/non-Delphi in every analysis
artifact. A run should be called "Delphi" only if its base registry entry says
it is a verified Delphi model.

## Minimal migration plan

1. Implement `experiments/midtrain.py` with specs, validators, manifest writer,
   and dry-run rendering only. No launch code yet.
2. Add tests for every incident in the ledger: wrong base, mode mismatch,
   stale W&B id, checkpoint namespace mismatch, invalid selector, unsafe data
   path, and cooldown `num_train_steps`.
3. Port the current CPT math launcher to emit `MidtrainSpec` objects and render
   equivalent train configs.
4. Port true-midtrain cooldown to `MidtrainSpec`.
5. Add `scripts/launch_midtrain.py plan/stage/launch/watch/resume`.
6. Run one CPT dry-run and one cooldown dry-run; compare rendered configs
   against the current experiment files.
7. Freeze old launch files by adding a top-level `raise RuntimeError` unless
   `MIDTRAIN_LEGACY_LAUNCH_OK=1` is set.
8. After one successful CPT and one successful cooldown launch under the new
   wrapper, delete the legacy env-var launch paths.

## Open questions

1. CPT default checkpoint source: native Levanter checkpoint is safer; HF is
   cheaper and enough for model-only if direct load/shape validation is solid.
   Decide whether HF should be allowed in production CPT by default or only via
   `CheckpointSourceKind.HF_WEIGHTS`.
2. Data cache manifest format: JSON is easiest for CLI interop; Python
   dataclasses are easier to import. Use JSON for saved manifests and typed
   Python for builders.
3. Direct Levanter launch boundary: decide whether `scripts/run_midtrain_cell.py`
   calls `run_levanter_train_lm` or invokes Levanter's CLI with a rendered YAML.
   The YAML route gives better reproducibility.
4. Whether to keep `default_train` in the new path. Avoiding executor is the
   priority; reusing `SimpleTrainConfig` is fine only if it does not drag in
   executor identity.
5. Cooldown checkpoint staging cost: native full-state checkpoints are large.
   The API should support staging once per base/step and fanning into output
   namespaces server-side, but launch must still verify per-cell staged copies.

# codex 5.5 2026-05-16T00:40:29Z

This addendum answers the design-coverage question directly. The first draft
specified the shape of a safer launcher, but it did not prove that the launcher
can express every experiment family we actually ran. This section adds that
coverage proof and expands the CPT budget model so fixed-ratio, fixed-token, and
per-base variable budgets are first-class.

## Coverage matrix: old experiments expressible in the new API

The unified `midtrain(spec)` API must cover all of these without falling back to
legacy env-var launch code.

| Historical experiment shape | Old mechanism | New representation | Required fields |
|---|---|---|---|
| April 10 B math LR sweep | `exp_delphi_math_10b_midtrain.py`, `MIDTRAIN_SELECT_BASE`, `MIDTRAIN_SELECT_LR`, fixed 10 B token budget | `mode=CPT`, `BudgetPolicy.FixedTokens(10_000_000_000)`, `DataMixRef(full_highquality_nemo_math)` or legacy math-only data manifest, LR grid expanded into one spec per LR | base, fixed tokens, LR factor, model-only init, explicit run identity |
| April zero-end-LR/BOS rerun | Same 10 B sweep, but rebuilt BOS-correct cache and `min_lr_ratio=0.0` | Same CPT spec with `min_lr_ratio=0.0` and `DataCacheManifest` pointing at the verified BOS-correct cache | data cache fingerprint, BOS sample proof, min LR policy |
| W&B project relaunch | `wandb_project="delphi-midtraining"` added to `default_train` | `RunIdentity(wandb_project="delphi-midtraining")` | W&B entity/project/run id in manifest |
| 20 B 1e20 mix-LR sweep | Hard token override plus run-name label | `BudgetPolicy.FixedTokens(20_000_000_000, label="20b")` | fixed tokens, explicit label, one spec per mix/LR/base |
| May K=0.20 per-scale sweep | `midtrain_token_budget(pretrain_tokens, fraction=0.20)` | `BudgetPolicy.PretrainFraction(0.20)` | pretrain tokens from base registry, computed token budget, computed steps |
| 4 LR x 3 mix x 3 scale design matrix | Nested loops over `BASES`, `LR_FACTORS`, and `MIDTRAIN_MIX_NAME` | `SweepSpec` expands to one `MidtrainSpec` per `(base, mix, lr, budget)` cell; launch still executes one cell at a time | grid expansion is dry-run only; launch takes one cell id |
| p33m67, p50m50, p67m33 replay mixes | `experiments.midtraining_mixes.midtraining_mix_by_name` | `DataMixRef` or `DataCacheManifest` resolved from the mix registry | weights, component names, val carve-out, cache paths |
| Full math/no pretrain replay | `FULL_HIGHQUALITY_NEMO_MATH_NAME` or legacy unset mix | `DataMixRef(full_highquality_nemo_math)` for safe mode; `UnsafeRawDatasetRef` only with explicit opt-in | safe mode must have val split; unsafe mode requires suffix and reason |
| CPT from native Levanter checkpoint | `initialize_from_checkpoint_path=<gcs checkpoint>`, `MODEL_ONLY` | `CptInit(source_kind=NATIVE_LEVANTER, checkpoint_path=...)` | checkpoint exists, banned-path check, shape check |
| CPT from HF weights | Previously discussed but not the main launch path | `CptInit(source_kind=HF_WEIGHTS, hf_repo=..., hf_revision=...)` | pinned revision, tokenizer match, conversion/direct-load proof; optimizer state intentionally absent |
| True midtraining/cooldown | `exp_delphi_true_midtrain.py`, pre-staged checkpoint under output path, natural resume | `mode=COOLDOWN`, `CooldownResume(pretrain_checkpoint_path, resume_step, staged_output_path)` | staged checkpoint integrity, original pretrain steps, original optimizer config |
| Resume after failure/preemption | `MIDTRAIN_RESUME_OUTPUT_PATH` plus expected floor | `launch_midtrain.py resume --output-path ... --expected-min-step ...` | old manifest, checkpoint search, W&B run id equality |
| Fallback from v5p-256 to v5p-64 | Manually adding `V5PComputeConfig("v5p-64", per_device_parallelism=4)` | Alternate `ComputeProfile` on the same `RunIdentity` for resume, or new `RunIdentity` for fresh attempt | same output path for resume; changed compute profile recorded in manifest attempt history |
| Placement-collision resiliency | `MIDTRAIN_MAX_TASK_FAILURES=100` env var | `ComputeProfile(max_task_failures=100)` | rendered into Iris launch request, logged in manifest |
| Single-cell dry-run/introspection | Import file, inspect `runs` | `launch_midtrain.py plan --cell <id>` | no Iris submission, no executor status writes |
| Finished-run inference handoff | Manual handoff doc and analysis scripts | `mark-finished` writes final manifest and optional handoff table row | HF path, W&B URL, evals, config repair notes |

This matrix implies one important implementation rule: sweep expansion and cell
launch are different operations. A `SweepSpec` may generate 36 `MidtrainSpec`
objects for review, but `launch` accepts exactly one selected `cell_id`.

## Budget model for CPT

CPT needs more than one budget mode. The old code used at least three:

- fixed absolute budget: 10 B and 20 B token sweeps;
- fixed fraction of pretraining: K=0.20, per-scale dynamic budgets;
- ad hoc smoke/debug budgets: short step counts or small token counts.

The redesign should make these explicit instead of overloading
`MIDTRAIN_TOKEN_BUDGET`.

```python
from dataclasses import dataclass
from enum import StrEnum


class BudgetKind(StrEnum):
    PRETRAIN_FRACTION = "pretrain_fraction"
    FIXED_TOKENS = "fixed_tokens"
    FIXED_STEPS = "fixed_steps"
    PER_BASE_TOKENS = "per_base_tokens"
    PER_BASE_STEPS = "per_base_steps"


@dataclass(frozen=True)
class BudgetPolicy:
    kind: BudgetKind
    label: str | None = None
    fraction: float | None = None
    tokens: int | None = None
    steps: int | None = None
    tokens_by_base: dict[str, int] | None = None
    steps_by_base: dict[str, int] | None = None
    rounding: str = "round"
```

Recommended constructors:

```python
BudgetPolicy.pretrain_fraction(0.20)
BudgetPolicy.fixed_tokens(10_000_000_000, label="10b")
BudgetPolicy.fixed_tokens(20_000_000_000, label="20b")
BudgetPolicy.fixed_steps(200, label="smoke200")
BudgetPolicy.per_base_tokens({"1e21": 9_250_000_000, "1e22": 32_070_000_000})
BudgetPolicy.per_base_steps({"1e21": 4_411, "1e22": 7_647})
```

Resolution:

```python
def resolve_cpt_budget(policy: BudgetPolicy, *, base: DelphiModel, batch_size: int, seq_len: int) -> ResolvedBudget:
    if policy.kind == BudgetKind.PRETRAIN_FRACTION:
        token_budget = round(base.tokens * policy.fraction)
    elif policy.kind == BudgetKind.FIXED_TOKENS:
        token_budget = policy.tokens
    elif policy.kind == BudgetKind.FIXED_STEPS:
        token_budget = policy.steps * batch_size * seq_len
    elif policy.kind == BudgetKind.PER_BASE_TOKENS:
        token_budget = policy.tokens_by_base[base.flops_key]
    elif policy.kind == BudgetKind.PER_BASE_STEPS:
        token_budget = policy.steps_by_base[base.flops_key] * batch_size * seq_len
    else:
        raise ValueError(policy.kind)

    num_train_steps = max(1, round(token_budget / (batch_size * seq_len)))
    actual_tokens = num_train_steps * batch_size * seq_len
    return ResolvedBudget(
        policy=policy,
        requested_tokens=token_budget,
        actual_tokens=actual_tokens,
        num_train_steps=num_train_steps,
        pretrain_fraction=actual_tokens / base.tokens,
        label=policy.label or default_budget_label(token_budget),
    )
```

Validation:

1. Exactly one budget value family must be set.
2. `PRETRAIN_FRACTION` requires `0 < fraction <= 1`.
3. `FIXED_TOKENS` and `FIXED_STEPS` require positive values.
4. `PER_BASE_TOKENS` and `PER_BASE_STEPS` must include every selected base.
5. The resolved `num_train_steps` must be positive.
6. The rendered run name includes the resolved label.
7. W&B tags include `budget_kind`, `requested_tokens`, `actual_tokens`,
   `pretrain_fraction_actual`, `num_train_steps`, `batch_size`, and `seq_len`.
8. If `FIXED_TOKENS` is used across multiple bases, that is allowed, but the
   manifest must say `budget_kind=fixed_tokens` so nobody mistakes it for K.
9. If `PRETRAIN_FRACTION` is used, the label should either be derived per base
   (`9p25b`, `32p07b`) or include the fraction (`k0p20`) depending on the naming
   policy. The manifest stores both, so analysis does not rely on the name.

This explicitly supports:

- old 10 B sweeps: `BudgetPolicy.fixed_tokens(10_000_000_000, label="10b")`;
- old 20 B sweeps: `BudgetPolicy.fixed_tokens(20_000_000_000, label="20b")`;
- May per-scale K sweep: `BudgetPolicy.pretrain_fraction(0.20)`;
- future variable budgets by scale:
  `BudgetPolicy.per_base_tokens({"1e21": 5_000_000_000, "1e22": 20_000_000_000})`;
- short smoke runs: `BudgetPolicy.fixed_steps(20, label="smoke20")`.

## SweepSpec: generating old-style design matrices safely

The API should include a sweep object only for planning. It expands to one-cell
specs; it never launches them as a batch inside Python.

```python
@dataclass(frozen=True)
class SweepSpec:
    mode: MidtrainMode
    bases: tuple[str, ...]
    mixes: tuple[str, ...]
    lr_factors: tuple[float, ...]
    budgets: tuple[BudgetPolicy, ...]
    compute_profiles: dict[str, ComputeProfile]
    naming: NamingPolicy
```

Examples:

```python
# Original fixed-budget LR sweep.
SweepSpec(
    mode=MidtrainMode.CPT,
    bases=("1e21", "1e22"),
    mixes=("full_highquality_nemo_math",),
    lr_factors=(0.5, 0.67, 0.83),
    budgets=(BudgetPolicy.fixed_tokens(10_000_000_000, label="10b"),),
    compute_profiles={"1e21": v5p64, "1e22": v5p256},
)

# K=0.20 36-cell sweep shape.
SweepSpec(
    mode=MidtrainMode.CPT,
    bases=("1e21", "1e22"),
    mixes=("p33m67", "p50m50", "p67m33"),
    lr_factors=(0.33, 0.5, 0.67, 0.83),
    budgets=(BudgetPolicy.pretrain_fraction(0.20),),
    compute_profiles={"1e21": v5p64, "1e22": v5p256},
)
```

Expansion writes a table:

| cell_id | base | mix | lr | budget_kind | tokens | steps | tpu | output_path |
|---|---|---:|---:|---|---:|---:|---|---|

The operator launches one row:

```bash
scripts/launch_midtrain.py launch --sweep sweep.toml --cell delphi-1e22-p50m50-k0p20-lr0p5
```

If no `--cell` is provided, the command prints the table and exits non-zero.
That avoids recreating the `executor_main` multi-cell collision.

## Cooldown budget is not a token-budget policy

Cooldown has a different meaning. It should not accept CPT budget policies by
default because it is an absolute-step continuation of a pretrain run.

Cooldown controls:

- `resume_step`: absolute step of the pretrain checkpoint staged into output;
- `num_train_steps`: original pretrain target step;
- `remaining_steps = num_train_steps - resume_step`;
- optional `stop_step_override`, only for debug or partial cooldown studies.

If `stop_step_override` exists, it must satisfy:

```text
resume_step < stop_step_override <= pretrain_num_train_steps
```

and the manifest must mark the run as a partial cooldown. The default must be
the full original pretrain target. Setting `num_train_steps` to a CPT-style
budget is forbidden because it recreates the flat-LR/count mismatch.

## More complete configuration surface

The first draft listed the major types, but a production spec needs these
additional policy objects:

```python
@dataclass(frozen=True)
class OptimizerPolicy:
    family: str = "AdamH"
    lr_factor: float | None = None          # CPT only
    lr_multiplier: float = 1.0
    min_lr_ratio: float = 0.0
    # Updated 2026-05-16: see top-of-file POLICY DECISION callout. CPT
    # warmup is now a fixed fraction of num_train_steps (default 0.10);
    # decay is Levanter's full-remainder sentinel (default None) so CPT is
    # triangular without fractional-stage truncation drift.
    # the warmup_tokens / warmup_fraction either-or below is retained only
    # as a sketch of the original design, NOT current behavior.
    warmup_tokens: int | None = None        # CPT — DEPRECATED, no longer used
    warmup_fraction: float | None = None    # CPT — now the only path, default CPT_DEFAULT_WARMUP_FRACTION
    schedule_source: str = "base_registry"  # cooldown uses pretrain schedule


@dataclass(frozen=True)
class NamingPolicy:
    prefix: str
    include_budget_label: bool = True
    include_lr: bool = True
    include_mode: bool = True
    attempt: str | None = None
    max_wandb_name_len: int = 64


@dataclass(frozen=True)
class ResumePolicy:
    expected_min_step: int
    require_wandb_run: bool = True
    require_manifest: bool = True
    allow_empty_resume: bool = False


@dataclass(frozen=True)
class SafetyPolicy:
    require_val_train_disjoint: bool = True
    allow_unsafe_no_val_split: bool = False
    require_bos_sample: bool = True
    reject_unknown_midtrain_env: bool = True
    reject_banned_base_paths: bool = True
```

These objects are mostly boring, but making them named objects prevents the old
pattern where behavior was hidden in scattered env vars.

## Experiments the design should refuse

A thorough design also defines refusals.

The unified launcher should refuse:

1. `mode=CPT` with `checkpoint_init_mode=FULL_STATE`.
2. `mode=COOLDOWN` with `initialize_from_checkpoint_path` set.
3. `mode=COOLDOWN` from HF weights.
4. `mode=COOLDOWN` with `num_train_steps != base.num_train_steps`, unless an
   explicit partial-cooldown stop step is set and labeled.
5. Any base key not registered in `experiments.delphi_models.py`.
6. Any path containing banned substrings such as `adamh_scaling_v5`.
7. Any fresh launch whose output path already contains checkpoints.
8. Any fresh launch whose W&B run id already exists.
9. Any resume launch whose latest checkpoint is below the expected floor.
10. Any launch with manual `RUN_ID` or `WANDB_RUN_ID` in the shell.
11. Any launch with unresolved data cache manifests.
12. Any multi-cell launch command.
13. Any selector typo that resolves zero cells.
14. Any TPU profile not allowlisted for the selected base.
15. Any run name longer than the W&B safe limit.

## Test plan for the redesigned API

Tests should encode the incident ledger, not just happy paths.

Unit tests:

- `test_wrong_1e20_base_rejected`
- `test_banned_v5_isoflop_path_rejected`
- `test_cpt_requires_model_only`
- `test_cooldown_forbids_initialize_from_checkpoint_path`
- `test_cooldown_requires_original_pretrain_num_train_steps`
- `test_fixed_tokens_budget_resolves_old_10b_steps`
- `test_pretrain_fraction_budget_resolves_k020_steps`
- `test_per_base_tokens_requires_all_selected_bases`
- `test_invalid_selector_fails_before_plan`
- `test_sweep_launch_without_cell_fails`
- `test_run_identity_derives_all_namespaces`
- `test_fresh_launch_refuses_existing_wandb_run`
- `test_resume_requires_checkpoint_floor`
- `test_unknown_midtrain_env_rejected`
- `test_data_manifest_requires_bos_sample`
- `test_eval_policy_uses_remaining_steps_for_cooldown`

Integration dry-run tests:

- render old 10 B CPT cell and compare `TrainLmConfig` fields to the legacy
  launcher: model shape, batch, seq len, LR, Adam LR, warmup, decay,
  `MODEL_ONLY`, eval cadence, checkpoint cadence;
- render K=0.20 1e21 and 1e22 CPT cells and verify token/step math;
- render true-midtrain 1e21 cooldown and verify no init checkpoint, original
  pretrain steps, staged output path, W&B id, and startup predicates;
- simulate a temp checkpoint under `<temp_root>/<run_id>/step-N` and verify
  resume chooses it over starting fresh.

Live smoke tests before production:

1. CPT smoke with `BudgetPolicy.fixed_steps(20, label="smoke20")` on the
   smallest verified safe base.
2. CPT dry-run with `BudgetPolicy.pretrain_fraction(0.20)` for the real cells.
3. Cooldown dry-run against an already staged checkpoint.
4. Cooldown resume dry-run against a temp checkpoint namespace.

## Answer to the coverage question

With the addendum above, the design can express the experiment families we have
been running:

- fixed-token CPT, including 10 B and 20 B;
- fixed-ratio CPT, including K=0.20 per base;
- arbitrary per-base variable CPT budgets;
- short smoke CPT;
- LR-factor sweeps;
- data-mixture sweeps;
- base-scale sweeps;
- TPU fallback/resume;
- true cooldown with full optimizer state;
- recovery from permanent or temporary checkpoints under the same run id.

The design should not preserve the old "one Python file launches a whole list"
behavior. That behavior is exactly what caused the `train_lm` collision. The
replacement is a planner that renders all cells and a launcher that executes one
selected cell with a manifest-backed identity.

# codex 5.5 2026-05-16T00:48:00Z

This section resolves implementation ambiguities from the first two drafts. If
this section conflicts with an earlier open question, this section wins.

## Concrete implementation decisions

| Question | Decision |
|---|---|
| Stable code location | Put the stable implementation under `lib/marin/src/marin/midtrain/`. Keep `experiments/` for Delphi-specific spec files only. |
| Training launch boundary | Do not call `default_train` and do not create training `ExecutorStep`s. Render a complete Levanter `TrainLmConfig` YAML plus a midtrain manifest, then launch one Iris TPU job that runs Levanter from that YAML. |
| Reuse boundary | Reuse pure config/data/model helpers. Do not reuse helpers whose behavior derives output identity from executor graphs. Extract pure helpers if needed. |
| Manifest location | The authoritative manifest is inside the run directory: `<output_path>/midtrain_manifest.json`. Planning can write a local copy, but launch/resume trust the copy under `output_path`. |
| Output region | `output_path` region pins training I/O. A run with `gs://marin-us-east5/...` trains and checkpoints in `us-east5`. Cross-region compute failover is not automatic for the same run id. |
| Compute fallback | Fallback to another TPU region is a new attempt with a new output path in that region, unless the operator explicitly accepts cross-region I/O for a recovery. |
| `mirror://` inputs | Allowed for source checkpoints in CPT and for source-to-stage resolution in cooldown. Not allowed as `output_path` or staged checkpoint path. Resolution is recorded in the manifest per attempt. |
| Fresh attempts | A fresh restart gets a new output basename and therefore a new W&B run id. Attempts are joined by `logical_cell_id` in manifests, not by reusing the same W&B run. |
| Same-run resume | Resume reuses the exact same `output_path` basename and W&B run id. It must find a checkpoint at or above the expected step before launch. |
| Checkpoint override | Only CPT can use a non-registry checkpoint override. Cooldown cannot. Override is a typed object with reason, expected shape, run suffix, banned-path check, and source proof. |
| Data manifests | Data launch consumes pinned JSON cache manifests. The command that produces them may use StepSpec/Zephyr, but training launch never carries live StepSpec data graphs. |
| Batch ergonomics | Provide `launch-batch`, but implement it as a loop over one-cell launches with manifest-recorded spacing/concurrency. Operators should not write ad hoc Bash loops. |
| Iris layer | Use Iris/Fray lower-level submission primitives through a new midtrain launcher helper. Do not use the Marin executor or its dependency hashing for training identity. |
| Eval basis type | Replace `EvalPolicy.basis: str` with a `StrEnum`. |
| Tokenizer identity | Use a typed `TokenizerRef` from shared constants. Equality is by canonical key and pinned revision/fingerprint, not by informal display string. |

## Concrete module layout

Implement the stable machinery here:

```text
lib/marin/src/marin/midtrain/
  __init__.py
  spec.py              # dataclasses, StrEnums, validation-only methods
  budget.py            # BudgetPolicy resolution
  identity.py          # RunIdentity, attempt grouping, W&B id rules
  data_manifest.py     # DataCacheManifest schema + validation
  levanter_config.py   # render TrainLmConfig YAML from MidtrainSpec
  staging.py           # checkpoint/data staging preflight helpers
  iris_launch.py       # submit one TPU job from manifest
  watch.py             # startup proof, checkpoint/W&B monitoring
  cli.py               # plan/stage/launch/launch-batch/resume/watch/mark-finished
```

Keep experiment-specific declarations thin:

```text
experiments/midtrain_specs/
  delphi_math_cpt.py       # SweepSpec definitions for CPT math runs
  delphi_true_cooldown.py  # SweepSpec definitions for cooldown runs
```

The old files `experiments/exp_delphi_math_10b_midtrain.py` and
`experiments/exp_delphi_true_midtrain.py` should become compatibility wrappers
or be frozen with a clear error once the new path has one successful CPT and one
successful cooldown launch.

## Levanter config rendering: no `default_train`

The runner should produce a concrete Levanter YAML file before Iris launch:

```text
<output_path>/midtrain_manifest.json
<output_path>/train_lm_config.yaml
<output_path>/launch_command.txt
```

The TPU job runs:

```bash
python -m levanter.main.train_lm --config <local-or-gcs-train_lm_config.yaml>
```

If Levanter's CLI entrypoint requires a different exact invocation, use that
entrypoint. The design point is that the training job consumes a complete
`TrainLmConfig` file; it does not reconstruct identity, data, or checkpoints
from executor dependencies at runtime.

Allowed reuse:

- `experiments.delphi_models` registry entries;
- `experiments.midtraining_mixes` for logical mix specs and weights;
- `experiments.scaling_law_sweeps.completed_adamh.completed_adamh_heuristic`
  for model/optimizer construction when the base registry says that is the
  source of truth;
- Levanter config dataclasses and optimizer classes;
- Marin/Iris/Fray resource submission dataclasses.

Forbidden reuse:

- `default_train(...)` for actual launch construction;
- `ExecutorStep` for training cell identity;
- any helper that calls `this_output_path()` or derives a run id from executor
  dependency hashes;
- `RUN_ID` / `WANDB_RUN_ID` shell env as user-provided inputs.

If existing Marin code has useful logic mixed with executor identity, extract a
pure helper first. For example, cadence helpers and HF export cadence can live
in `marin.midtrain.levanter_config`; the code may mirror current behavior, but
it should not import executor-specific launch wrappers.

## Direct Iris invocation

The new launcher submits one TPU training job, not a CPU coordinator that runs
an executor graph.

Target shape:

```bash
uv run python -m marin.midtrain.cli launch --manifest gs://.../midtrain_manifest.json
```

Internally, `marin.midtrain.iris_launch` should use Iris/Fray submission
primitives with:

- explicit TPU resources from `ComputeProfile`;
- explicit region equal to `RunIdentity.output_region`;
- explicit env containing `RUN_ID=<basename(output_path)>`;
- explicit W&B config already present in YAML;
- `max_task_failures` from `ComputeProfile`;
- no executor dependency graph;
- no generated output hash.

The implementation may use a lightweight CPU submitter only if the Iris API
cannot submit the TPU job directly from the user's machine. If so, that CPU job
is a transport detail and must not compute or mutate run identity. The TPU child
still receives the fixed manifest and output path.

## Region and staging policy

`RunIdentity.output_path` has a concrete bucket and therefore a concrete output
region:

```python
output_region("gs://marin-us-east5/checkpoints/foo") == "us-east5"
```

Rules:

1. Training writes permanent checkpoints only under `output_path`.
2. Temporary checkpoints go under the temp bucket for the same region and the
   same run id:

   ```text
   gs://marin-<region>/tmp/ttl=14d/checkpoints-temp/<run_id>/
   ```

   or whatever current Marin temp helper resolves for that region. The run id
   suffix is mandatory.
3. `ComputeProfile.regions` for a real launch must be either exactly
   `(output_region,)` or empty/`None`, which resolves to `(output_region,)`.
4. A multi-region `ComputeProfile.regions=("us-east5", "us-central1")` is
   allowed only at planning time to choose a new attempt region. It is not
   allowed for a single same-run launch.
5. Same-run resume cannot move regions. If the old output path is east5, resume
   launches in east5.
6. Fresh fallback to a different region creates a new attempt:

   ```text
   logical_cell_id = delphi-1e22-p50m50-k0p20-lr0p5
   attempt 1       = gs://marin-us-east5/checkpoints/...-a001
   attempt 2       = gs://marin-us-central1/checkpoints/...-a002
   ```

7. Cross-region checkpoint or cache copying is never implicit. The staging
   command refuses by default and prints a cost estimate.

Cooldown staging policy:

- The launch path requires the staged checkpoint to already exist under
  `<output_path>/checkpoints/step-<resume_step>`.
- `stage-cooldown` can create that staged copy only when source and destination
  are in the same region.
- If source and destination differ by region, `stage-cooldown` refuses unless
  the operator passes:

  ```bash
  --allow-cross-region-copy --copy-budget-gb <N> --reason <text>
  ```

- The manifest records `cross_region_copy=true`, source, destination, byte
  count, budget, and reason.
- There is no hidden canonical multi-region checkpoint store in this redesign.
  If we want replicated base checkpoints, create explicit staged-base manifests
  per region.

This is stricter than the historical workflow. It is intentional: the old
workflow let placement, mirroring, and identity interact in unsafe ways.

## `mirror://` policy

`mirror://` is a source-resolution mechanism, not a run identity.

CPT:

- `CptInit.checkpoint_ref` may point to a registry checkpoint that renders as
  `mirror://...`.
- During `stage` or `launch` preflight, the launcher resolves it in the output
  region and records:

  ```json
  {
    "logical_source": "mirror://...",
    "resolved_source": "gs://marin-us-east5/...",
    "resolved_region": "us-east5"
  }
  ```

- If a same-run resume happens later, the source checkpoint is not used unless
  no run checkpoint exists. Since resume requires an existing run checkpoint,
  source re-resolution should not affect same-run recovery.

Cooldown:

- `cooldown_resume.pretrain_checkpoint_path` may be `mirror://...` only as the
  source for `stage-cooldown`.
- The staged checkpoint path under `output_path` must be concrete `gs://...`.
- Once staged, launch and resume read only the staged run checkpoint or later
  checkpoints from the same run namespace.

If a fresh attempt in a different region resolves the same `mirror://` source to
a different physical bucket, that is allowed because it has a different
attempt/output path and W&B run id. The logical source remains the same in the
attempt group manifest.

## Attempts and W&B identity

Distinguish `logical_cell_id` from `run_id`.

```python
@dataclass(frozen=True)
class RunIdentity:
    logical_cell_id: str      # stable across fresh attempts
    attempt: int              # 1, 2, 3...
    output_path: str          # includes attempt suffix
    wandb_project: str
    wandb_entity: str = "marin-community"

    @property
    def run_id(self) -> str:
        return basename(self.output_path)
```

Fresh attempts:

- Attempt is part of the output basename.
- W&B `run_id` equals the output basename.
- Attempt 2 does not reuse attempt 1 W&B run id, so W&B monotonic-step
  rejection cannot happen.
- Attempt manifests share `logical_cell_id`.

Example:

```text
logical_cell_id: delphi-1e22-p50m50-k0p20-lr0p5
attempt 1 run_id: delphi-1e22-p50m50-k0p20-lr0p5-a001
attempt 2 run_id: delphi-1e22-p50m50-k0p20-lr0p5-a002
```

Same-run resume:

- Same `output_path`.
- Same `run_id`.
- Same W&B run id.
- Requires checkpoint proof.
- Does not increment `attempt`.

Attempt group manifest:

```text
gs://marin-<region>/midtrain-manifests/runs/<logical_cell_id>.json
```

This group manifest lists all attempt manifests and their statuses. Analysis
joins attempts through this file, not through W&B display names.

## Checkpoint override type

The first draft referenced `CheckpointOverride` but did not define it. Define it
as CPT-only:

```python
@dataclass(frozen=True)
class CheckpointOverride:
    checkpoint_path: str
    reason: str
    run_name_suffix: str
    expected_hidden_dim: int
    expected_num_layers: int
    expected_seq_len: int
    expected_tokenizer: TokenizerRef
    allow_hf_weights: bool = False
```

`CptInit` becomes:

```python
@dataclass(frozen=True)
class CptInit:
    source_kind: CheckpointSourceKind
    registry_model: DelphiModel | None = None
    checkpoint_override: CheckpointOverride | None = None
    hf_repo: str | None = None
    hf_revision: str | None = None
    reset_optimizer: bool = True
    reset_data_loader: bool = True
```

Rules:

1. Exactly one of `registry_model`, `checkpoint_override`, or
   `(hf_repo, hf_revision)` is set.
2. `checkpoint_override` requires a non-empty `reason`.
3. `checkpoint_override.run_name_suffix` is appended to `logical_cell_id` and
   output basename.
4. Banned substrings are checked even for overrides.
5. Shape metadata must match before launch.
6. Overrides are not allowed in cooldown. Cooldown source checkpoints come from
   the base registry or from an explicit staged-base manifest produced from the
   base registry.

This makes accidental `MIDTRAIN_INIT_CKPT_PATH` behavior impossible. A custom
checkpoint changes the run name and manifest by construction.

## Data manifest production

Do not make `DataCacheManifest.from_registry(...)` a live StepSpec lookup. It
loads a pinned JSON manifest.

Physical layout:

```text
gs://marin-<region>/midtrain-manifests/data/<mix_name>/<fingerprint>.json
experiments/midtrain_data_manifests/<mix_name>.json  # optional pointer file
```

The in-repo pointer file is small:

```json
{
  "mix_name": "pretrain_33p_math_67p_highquality_nemo_math",
  "approved_manifest_uri": "gs://marin-us-east5/midtrain-manifests/data/.../abc123.json",
  "approved_at": "2026-05-16T00:48:00Z"
}
```

Build command:

```bash
uv run python -m marin.midtrain.cli data-manifest build \
  --mix pretrain_33p_math_67p_highquality_nemo_math \
  --region us-east5 \
  --output-uri gs://marin-us-east5/midtrain-manifests/data/pretrain_33p_math_67p_highquality_nemo_math/
```

This command may call existing StepSpec/Zephyr/cache code. It is allowed to
materialize or verify caches because it is a data-prep command, not a training
launch.

The produced JSON includes:

```json
{
  "schema_version": 1,
  "mix_name": "...",
  "mix_spec_digest": "sha256:...",
  "region": "us-east5",
  "components": [
    {
      "logical_name": "nemotron_cc_math_v1/4plus",
      "cache_path": "gs://marin-us-east5/tokenized/...",
      "cache_digest": "sha256:...",
      "total_sequences": 45096087,
      "total_tokens": 51482572371,
      "tokenizer": {"key": "llama3", "revision": "..."},
      "bos_sample": [128000, 128000, 128000],
      "validation_fingerprint": "sha256:..."
    }
  ],
  "weights": {"pretrain": 0.33, "nemotron_cc_math_v1/4plus": 0.67}
}
```

Fingerprint rules:

- `mix_spec_digest` hashes logical component names, weights, val carve-out,
  seq len, tokenizer ref, and split policy.
- `cache_digest` hashes cache path, cache metadata, tokenizer ref, length, BOS
  sample, and optionally a sampled content hash.
- If a cache silently rebuilds with a new path, length, tokenizer, or sampled
  content, the manifest fingerprint changes and the old approved pointer no
  longer matches.

Training launch requires an explicit `--data-manifest` URI or a registry alias
that resolves to one approved URI. It never walks live StepSpec graphs.

## Batch launch ergonomics

Provide a safe batch loop in the CLI:

```bash
uv run python -m marin.midtrain.cli launch-batch \
  --sweep experiments/midtrain_specs/delphi_math_cpt.py:k020_sweep \
  --filter 'base in ["1e21","1e22"] and mix == "p33m67"' \
  --max-concurrent 1 \
  --launch-spacing-seconds 30
```

Behavior:

1. Expands the sweep to one-cell manifests.
2. Validates every cell.
3. Prints the full table.
4. Submits cells by calling the same one-cell `launch` path.
5. Sleeps `launch_spacing_seconds` between submissions.
6. Writes a batch manifest with cell ids, job ids, attempts, and statuses.
7. Refuses `--max-concurrent > 1` unless every selected cell has a distinct
   output path and the operator passes `--allow-concurrent`.

This preserves ergonomics without reintroducing Python-level multi-cell
`executor_main`.

## Eval and tokenizer concrete types

Use enums and typed tokenizer refs:

```python
class EvalBasis(StrEnum):
    MODE_DEFAULT = "mode_default"
    TOTAL_STEPS = "total_steps"
    REMAINING_STEPS = "remaining_steps"


@dataclass(frozen=True)
class EvalPolicy:
    target_points: int = 40
    min_steps: int = 25
    max_steps: int = 200
    basis: EvalBasis = EvalBasis.MODE_DEFAULT


@dataclass(frozen=True)
class TokenizerRef:
    key: str                 # e.g. "llama3"
    hf_repo: str
    revision: str
    bos_token_id: int
    eos_token_id: int
    vocab_size: int
    fingerprint: str | None = None
```

Canonical tokenizer refs live in one module:

```text
lib/marin/src/marin/midtrain/tokenizers.py
```

Base registry entries and data manifests must both point to the same
`TokenizerRef.key` and revision/fingerprint. Preflight also checks sampled BOS
tokens because equal names alone did not protect us from the stale BOS cache.

## Updated migration plan

Replace the earlier minimal plan with this order:

1. Add `lib/marin/src/marin/midtrain/spec.py`, `budget.py`, `identity.py`, and
   validators. No launch code.
2. Add data manifest schema and a `data-manifest build/verify` CLI. Use it to
   produce one manifest for the existing p33m67 mix in us-east5.
3. Add Levanter YAML rendering for CPT only. Compare rendered CPT config against
   the current math launcher for one 1e21 fixed-token cell and one K=0.20 cell.
4. Add one-cell direct Iris launch for CPT.
5. Add `launch-batch` as a safe loop over one-cell launch.
6. Add cooldown staging and cooldown YAML rendering.
7. Add cooldown launch/resume.
8. Freeze legacy launch files.

This order lets the next agent start with specs and validators without guessing
about Levanter invocation, manifest paths, data manifests, or attempts.

# claude 2026-05-16 — Implementation landed under `lib/marin/src/marin/midtraining/`

This entry summarizes what was actually built against the redesign brief, what
was added beyond the spec, what was cut after a second pass, and what the
public surface now looks like. Folder name is `midtraining/` (not `midtrain/`)
per the operator's preference; the design is otherwise unchanged.

## Final layout

```text
lib/marin/src/marin/midtraining/
  __init__.py          public API, ~50 re-exports
  schema.py            stable RunManifestRow TypedDict + validating reader/writer
  modes.py             CptMode, CooldownMode, CptInit, CooldownResume, CheckpointOverride
  spec.py              MidtrainSpec (flat) + ComputeProfile + cross-cutting validators
  preflight.py         GCS-aware preflight + cooldown checkpoint staging + fake_gcs() helper
  launch.py            build_manifest_row, write_manifest, build_launch_request, submit_launch
  levanter_config.py   render_train_lm_config / render_train_lm_yaml
  data_manifest.py     DataCacheManifest + DataCacheComponent + load/dump
  identity.py          RunIdentity + build_run_identity + output_region
  budget.py            BudgetPolicy (3 kinds) + resolve_cpt_budget + ResolvedBudget
  tokenizers.py        TokenizerRef + canonical LLAMA3_TOKENIZER / QWEN3_TOKENIZER
  watch.py             evaluate_startup + StartupProof
```

12 files, ~2,509 library LOC. Plus 9 test files / ~1,147 LOC, 63 tests passing.

## Things that match the redesign brief literally

- Folder under `lib/marin/src/marin/midtraining/` (relabeled from `midtrain/`).
- `MidtrainSpec` is the single explicit input. `resolve_midtrain_spec` →
  `ResolvedMidtrainSpec` → `validate_midtrain_spec` → preflight → launch.
- Two modes: `CptMode` and `CooldownMode` are concrete classes. They are a
  closed union (`TrainingMode = CptMode | CooldownMode`), not an enum + parallel
  optional fields.
- `RunIdentity` carries `logical_cell_id` + `attempt`. `run_id =
  basename(output_path)` includes a zero-padded `-aNNN` attempt suffix, so a
  fresh attempt is always a distinct W&B row. Attempt-group manifest at
  `gs://marin-<region>/midtrain-manifests/runs/<logical_cell_id>.json` joins
  attempts.
- Data manifests are pinned content-addressed JSON under
  `gs://marin-<region>/midtrain-manifests/data/<mix>/<fingerprint>.json`. The
  loader validates region match, BOS sample presence/value, weight sum, and
  tokenizer uniformity across components.
- CPT renders `initialize_from_checkpoint_path` + `checkpoint_init_mode=model_only`.
  Cooldown renders `initialize_from_checkpoint_path=None` with
  `checkpoint_init_mode=full_state` (moot — checkpoint discovery does the work).
- Temp checkpoint path includes the run id so two attempts never collide.
- `RunManifestRow` is a `TypedDict` in `schema.py`. `read_run_manifest(uri)`
  validates structurally; downstream analysis can consume manifests without
  importing the launcher.
- Iris/Fray submission uses `Entrypoint.from_binary("python", ["-m",
  "levanter.main.train_lm", "--config", <yaml_uri>])`. No `default_train`. No
  training `ExecutorStep`.
- Cross-region cooldown copy refuses unless an explicit `CrossRegionCopyPolicy`
  is passed; the manifest records `cross_region_copy=true` with budget + reason.
- CPT-only `CheckpointOverride` with `reason`, `run_name_suffix`, shape
  metadata, banned-substring check, allow-HF flag. Cooldown cannot override.
- Banned substrings (e.g. `adamh_scaling_v5`) are checked across base
  `gcs_run_root` / `verified_checkpoint_path` / `hf_repo` and mode-specific
  paths. Cell-author code passes the canonical Delphi banned set into the spec.

## Things added beyond the brief

- `BaseModelRef` Protocol in `spec.py`. The `midtraining` package lives in
  `lib/marin/`; the dependency direction in `AGENTS.md` forbids importing from
  `experiments/`. Cell-author code passes `experiments.delphi_models.DelphiModel`
  instances structurally; pyrefly stays honest.
- `fake_gcs(*paths)` helper in `preflight.py` returns `(exists, list_)` callable
  pair over a frozenset for unit tests. No `FakeGcsClient` class.
- `model_config` and `optimizer_config` are required dict fields on
  `MidtrainSpec`. The launcher does not import
  `experiments.scaling_law_sweeps.completed_adamh.completed_adamh_heuristic` —
  the cell author renders both blocks and passes them in. The renderer just
  inlines them under `model:` and `optimizer:` keys in the YAML. The validator
  checks `model_config["hidden_dim"]` and `["num_layers"]` against the base.
- Eval cadence basis is auto-picked by mode (CPT → CPT step count;
  cooldown → remaining tail). No user-facing `EvalBasis` enum.

## What was cut after a second pass against `downstream_scaling/evals`

The first cut of the redesign produced ~3,400 LOC. After comparing to
`downstream_scaling/evals` (~200 LOC for the same kind of protocol +
stable-schema decomposition), a second consolidation pass landed:

1. **CLI dropped** (`cli.py`, ~479 LOC). Operators write Python launcher
   scripts that import library functions directly — matches the
   `downstream_scaling/run_delphi_masked_gsm8k_iid.py` pattern. Discoverability
   moves from `--help` to reading one example launcher.
2. **`SweepSpec` / `SweepCell` / `expand_sweep` / `NamingPolicy` dropped**.
   The cell author writes a `for` loop in their launcher script. The redesign
   already said sweeps are shell loops over one-cell launches; this honors that
   literally.
3. **Policy containers dropped**: `TrainingPolicy`, `RunPolicy`, `EvalPolicy`,
   `CheckpointPolicy`, `SafetyPolicy`, `ResumePolicy` are gone. Their fields
   sit flat on `MidtrainSpec` with sensible defaults (~20 fields total, most
   unset by the cell author).
4. **`EvalBasis` enum dropped**. Auto-picked from the mode.
5. **`PER_BASE_TOKENS` / `PER_BASE_STEPS` budget kinds dropped** (YAGNI; never
   used historically). Three budget kinds remain: `pretrain_fraction`,
   `fixed_tokens`, `fixed_steps`. The cell author can construct one
   `BudgetPolicy` per base in their loop if they want per-base values.
6. **`FakeGcsClient` class dropped**. Callable injection (`exists`, `list_`).
7. **`manifest.py` + `iris_launch.py` merged into `launch.py`**.
8. **`staging.py` → `preflight.py`** (its actual primary job).

Net: ~1,060 LOC out, ~22%. Test count went 76 → 63 because mode-specific
behavior is now tested via consumer tests rather than dedicated `test_modes.py`
methods.

## What was kept despite pressure to cut

- **All validators**. Every one defends against a named incident-ledger entry.
  Banned substrings, region alignment, tokenizer compatibility, BOS sample,
  attempt-suffix encoding, output-region equality, mode/init mismatch —
  load-bearing.
- **`schema.py` as a standalone module**. Mirrors the `evals/framework/schema.py`
  artifact-as-contract pattern. Downstream analysis can `from
  marin.midtraining.schema import read_run_manifest` without pulling in the
  launcher.
- **Mode methods**. I tested replacing `render_init_section()` etc. with `match
  spec.mode: case CptMode(): ...` at consumer sites; consumers grew faster than
  modes shrank. The methods earn their weight as a place to keep mode-specific
  formatting close to mode data.
- **`RunIdentity` validation**. The fat `__post_init__` is the defense against
  the W&B step-rejection and cross-region executor-hash classes of bugs.
- **`BaseModelRef` Protocol**. Cheap; pyrefly stays honest.

## Cell-author surface (the actual API for `experiments/midtrain_specs/`)

```python
from marin.midtraining import (
    BudgetPolicy, CheckpointSourceKind, ComputeProfile, CptInit, CptMode,
    LLAMA3_TOKENIZER, MidtrainSpec, build_launch_request, build_manifest_row,
    build_run_identity, preflight, resolve_midtrain_spec, submit_launch,
    validate_midtrain_spec, write_manifest, write_train_config,
)
from experiments.delphi_models import DELPHI_1E21, DELPHI_BANNED_SUBSTRINGS

def make_spec(lr_factor: float) -> MidtrainSpec:
    run = build_run_identity(
        logical_cell_id=f"delphi-1e21-p33m67-k0p20-lr{int(lr_factor*100)}",
        attempt=1,
        output_region_name="us-east5",
        wandb_project="delphi-midtraining",
    )
    return MidtrainSpec(
        base=DELPHI_1E21,
        run=run,
        compute=ComputeProfile(tpu_type="v5p-64", batch_size=512, regions=("us-east5",)),
        mode=CptMode(
            init=CptInit(source_kind=CheckpointSourceKind.NATIVE_LEVANTER,
                         registry_model=DELPHI_1E21),
            budget=BudgetPolicy.pretrain_fraction(0.20),
        ),
        data_manifest_uri="gs://marin-us-east5/midtrain-manifests/data/p33m67/<fingerprint>.json",
        tokenizer=LLAMA3_TOKENIZER,
        model_config=build_llama_config(DELPHI_1E21),     # cell author
        optimizer_config=build_adamh_config(DELPHI_1E21, lr_factor=lr_factor),
        banned_substrings=DELPHI_BANNED_SUBSTRINGS,
    )

def main():
    for lr in (0.5, 0.67, 0.83):
        spec = make_spec(lr)
        resolved = resolve_midtrain_spec(spec)
        validate_midtrain_spec(resolved)
        report = preflight(resolved)
        if not report.ok:
            print(f"FAIL {spec.run.logical_cell_id}: {report.failures}")
            continue
        row = build_manifest_row(resolved, report, status="launched")
        write_manifest(row, output_path=spec.run.output_path)
        write_train_config(resolved)
        submit_launch(build_launch_request(resolved))
        time.sleep(spec.compute.launch_spacing_seconds)
```

`python experiments/midtrain_specs/delphi_math_cpt_1e21.py` runs the sweep.

## Migration plan — status

Original 8-step migration plan (`midtraining_redesign.md` § "Updated migration
plan"):

| # | Step                                                            | Status |
|---|------------------------------------------------------------------|--------|
| 1 | Add spec/budget/identity/validators                              | done |
| 2 | Data manifest schema + verify CLI                                | schema + load/dump done; build CLI not implemented (cell-author or a separate `marin.midtraining.data` helper can produce one) |
| 3 | Levanter YAML rendering for CPT; compare against legacy launcher | rendering done; no apples-to-apples diff against `exp_delphi_math_10b_midtrain.py` yet |
| 4 | One-cell direct Iris launch for CPT                              | submit path done; not exercised against a live cluster |
| 5 | `launch-batch` as safe loop                                      | dropped (operator writes Python loop) |
| 6 | Cooldown staging + cooldown YAML rendering                       | done |
| 7 | Cooldown launch/resume                                           | submit + resume preflight done; not exercised against a live cluster |
| 8 | Freeze legacy launch files                                       | not done — `experiments/exp_delphi_math_10b_midtrain.py` and `experiments/exp_delphi_true_midtrain.py` still operational |

## Things deliberately not implemented yet

- **Data manifest build command**. The schema + reader exist; the producer that
  walks the existing `experiments/midtraining_mixes.py` + Zephyr cache code and
  emits a fingerprinted JSON is not written. Until it lands, every cell needs
  its data manifest produced by hand or by ad-hoc script. This is the most
  important next gap.
- **`experiments/midtrain_specs/`**. No Delphi-specific spec files exist yet.
  The first one (e.g. `delphi_math_cpt.py`) should mirror the example above and
  include the `model_config` / `optimizer_config` builders that import
  `completed_adamh_heuristic`. Until those land, the new path cannot replace
  the legacy launchers.
- **Legacy file freeze**. The redesign calls for a top-level
  `raise RuntimeError` on `exp_delphi_math_10b_midtrain.py` and
  `exp_delphi_true_midtrain.py` after one successful new-path launch in each
  mode. Premature to add until step 4/7 are exercised live.
- **Dataset-cache build provenance**. The manifest carries a `cache_digest`
  field but the producer is responsible for filling it correctly. Without the
  build command (above), this is on the honor system.
- **`mirror://` resolution path**. Specs can carry `mirror://` source
  checkpoints (and the cooldown source allows them); the actual resolution to
  a regional `gs://` URI at stage time is not implemented — the staging code
  currently expects `gs://` already. The decision is in the redesign doc but
  the code path is a TODO.

## Tests by area

- `tests/midtraining/test_budget.py` — 7 tests (kinds, label, validation).
- `tests/midtraining/test_identity.py` — 9 tests (attempt suffix, region
  resolution, W&B name length, attempt-group URI).
- `tests/midtraining/test_data_manifest.py` — 7 tests (weight sum, region
  match, tokenizer uniformity, fingerprint stability, pointer file).
- `tests/midtraining/test_schema.py` — 6 tests (TypedDict guard, roundtrip,
  malformed rejection).
- `tests/midtraining/test_spec_validators.py` — 16 tests covering each
  incident-ledger entry from the redesign brief.
- `tests/midtraining/test_levanter_config.py` — 5 tests (CPT/cooldown init
  rendering, W&B tags, temp path includes run id).
- `tests/midtraining/test_preflight.py` — 9 tests (existing-namespace refuse,
  resume floor, cooldown staged checkpoint, cross-region copy guard, dry-run).
- `tests/midtraining/test_watch.py` — 4 tests (CPT/cooldown startup proof,
  W&B step regression).

The `test_modes.py` file from the first cut was folded into
`test_spec_validators.py` and `test_levanter_config.py` after the
consolidation.

## Comparison to `experiments/downstream_scaling/evals`

The redesign brief said "the launcher should be small and composable, like the
eval framework." After the consolidation pass, the comparison is:

| Aspect | downstream_scaling/evals | midtraining/ |
|---|---|---|
| Framework LOC | ~200 (core.py + schema.py) | ~480 (spec + modes + launch) |
| Composers | one (`make_eval_step`) | two (`resolve_midtrain_spec`, `build_launch_request`) |
| Stable artifact schema | `PromptRow` / `CompletionRow` / `GradeRow` | `RunManifestRow` |
| User-facing classes | 2 protocols + 3 TypedDicts | 1 dataclass + 1 union + 1 TypedDict |
| Discoverable verbs | `make_eval_step` | `resolve_*`, `validate_*`, `preflight`, `build_launch_request`, `submit_launch`, `write_manifest`, `write_train_config` |
| CLI? | no | no |

Midtraining stays larger because:

1. Identity validation (~190 LOC) defends against W&B/output-namespace bugs;
   evals don't have those.
2. Cooldown staging + cross-region guards (~250 LOC) — evals don't move
   tens-of-GB checkpoints.
3. Mode invariants for CPT vs cooldown (~390 LOC) — evals don't have a state
   machine for optimizer/scheduler.
4. Preflight against asymmetric blast radius — a wrong train is hours of TPU
   plus a W&B step-rejection; a wrong eval is "rerun in 10 minutes."

The remaining gap is not over-engineering — it is the irreducible "training is
costlier than eval" delta. The structural shape (data-as-contract via TypedDict
schemas, composer functions, no inheritance hierarchies, no CLI) is now
parallel.

# claude 2026-05-16 — Small-base K=0.20 sweep + math-val-set bit-equivalence guarantee

This entry adds the artifacts required to launch midtraining on Delphi
3e18 -> 2e20 at K=0.20 while keeping the math validation partition
byte-identical to the 1e21/1e22 K=0.20 sweep.

## Why val identity matters

The 1e21 and 1e22 K=0.20 runs hold out 12,500 sequences from
`nemotron_cc_math_v1/4plus` as the math val partition. Any cross-scale
val-loss plot (3e18 ... 2e20 vs 1e21/1e22) is only interpretable if all
scales see the same val set. Levanter's val carve-out is determined by
(cache content, num_validation_sequences, shuffle config, tokenizer);
there is no random seed. So byte-equivalence is achievable but requires
emitting the same data config.

## Verification of val identity across the reference runs

Fetched `.executor_info` from four canonical 1e21/1e22 K=0.20 runs:
- `delphi-1e21-p33m67-9p25b-lr0.5-efbc63`
- `delphi-1e21-p50m50-9p25b-lr0.5-973c46`
- `delphi-1e21-p67m33-9p25b-lr0.5-114e49`
- `delphi-1e22-p33m67-...`

Every val-determining field is **bit-identical** across all four:

| field | value |
|---|---|
| math cache_dir | `gs://marin-us-east5/tokenized/nemotron_cc_math_v1/4plus-2c5519` |
| `num_validation_sequences` | `{"nemotron_cc_math_v1/4plus": 12500}` |
| `shuffle_before_trainval_split` | `True` |
| `shuffle` | `{io_block_size: 256, window_blocks: 512, perm_type: feistel}` |
| `permutation_type` | `feistel` |
| `block_cross_document_attention` | `True` |
| `mixture_block_size` | `2048` |
| `enforce_eos` | `True` |
| `cache_options` | `{batch_size: 128}` |
| `tokenizer` | `meta-llama/Meta-Llama-3.1-8B` |
| `vocab_size` | `null` (resolved lazily) |
| `stop_strategy` | `restart` |

Only `train_weights` differs across mixes (the math share is 0.67/0.5/0.33
for p33m67/p50m50/p67m33). The val carve-out itself is mix-invariant.

## Gap in the prior `marin.midtraining` rendering

`levanter_config._build_data_section` was emitting a `type: mixture`
wrapper with `type: lm_cached` per component — a schema the legacy runs
don't use. It also lacked `num_validation_sequences`, the deterministic
Feistel `shuffle` block, `block_cross_document_attention`,
`mixture_block_size`, `enforce_eos`, and `cache_options`. Shipping
without the fix would silently change the val carve-out (or skip it).

## Fixes landed

### 1. Capture the three references verbatim

Saved to `experiments/midtrain_specs/data_sections/{p33m67,p50m50,p67m33}.json`
(~22 KB each) via `gcloud storage cat .../.executor_info` and extracting
`.config.train_config.data`. These are now in-repo; git history is the
audit trail.

### 2. `data_section_override` passthrough on `MidtrainSpec`

`MidtrainSpec` now requires *exactly one* of:
- `data_manifest_uri`: gs:// URI of a content-addressed manifest (the
  forward-looking path).
- `data_section_override`: raw Levanter `data:` dict + a
  `data_section_provenance` string (e.g.
  `"legacy:delphi-1e21-p33m67-9p25b-lr0.5-efbc63"`).

When `data_section_override` is set:
- `levanter_config._render_data_section` deep-copies it as the rendered
  `data:` block. No transformation.
- `_assert_run_region_alignment` checks each component's `cache_dir`
  against the run region instead of the manifest URI.
- `_assert_legacy_data_section_consistent` requires the override's
  `tokenizer` matches `spec.tokenizer.hf_repo`, that
  `num_validation_sequences` is non-empty, and that
  `shuffle_before_trainval_split=True`.
- preflight requires `data_section_provenance` to be set (otherwise the
  launch is not auditable).
- The run manifest records `data_manifest_uri = "legacy:<provenance>"`
  and `data_manifest_fingerprint = "legacy:<provenance>"` so analysis
  scripts can detect legacy-passthrough cells.

### 3. Tokenizer correction

The canonical `LLAMA3_TOKENIZER` was pointing at `meta-llama/Meta-Llama-3-8B`;
every Delphi `.executor_info` actually carries `meta-llama/Meta-Llama-3.1-8B`.
The two share BOS/EOS/vocab but the string identity matters for the
passthrough equality check. Updated to `Meta-Llama-3.1-8B` with a pinned
revision (`0e9e39f249a16976918f6564b8830bc894c89659`).

### 4. Optimizer hparams promoted to `DelphiModel`

`experiments/delphi_models.py:DelphiModel` now carries `peak_lr`,
`peak_adam_lr`, `beta2`, `epsilon` (per-base required) plus shared
defaults for `beta1`, `max_grad_norm`, `weight_decay`, `z_loss_weight`,
`warmup_fraction`, `decay_fraction`, `min_lr_ratio`, `lr_schedule`,
`nesterov`. Values for 7 v6 isoflop bases come straight from
`.executor_info.config.train_config.optimizer`; 1e21/1e22 (v5 ladder)
come from the legacy hardcoded block in
`experiments/exp_delphi_math_10b_midtrain.py`. The doc-string captures
the convention:

> DO NOT recompute via the heuristic; the W&B config is the canonical
> source of truth for what the weights were optimized against (G4).

### 5. Cell-author file

`experiments/midtrain_specs/delphi_small_cpt_k020.py` — sweep launcher
for the 6 small bases x 3 mixes x 4 LRs (72 cells). Each cell:

- Loads the mix's legacy data section verbatim (val carve-out by
  construction).
- Reads `base.peak_lr * lr_factor`, `base.peak_adam_lr * lr_factor`, and
  the rest of the optimizer fields from the registry.
- Builds the Qwen3 model config via
  `completed_adamh_heuristic._build_model_config(hidden_size=base.hidden_dim)`.
- Renders `CptInit(source_kind=HF_WEIGHTS, hf_repo=base.hf_repo,
  hf_revision=base.hf_revision)` which Levanter consumes as
  `initialize_from_hf: <repo>@<revision>` and streams over `hf://` URLs.
  No GCS staging; free ingress.

CLI: select bases / mixes / LRs to launch one or many cells. Defaults to
all 72 if no flags; `--dry-run` plans without submitting. Spacing 30s
between submissions guards against Iris coordinator collision.

### 6. Bit-equivalence guard tests

`tests/midtraining/test_val_set_equivalence.py` (27 tests) asserts:

- Every reference (p33m67 / p50m50 / p67m33) declares
  `num_validation_sequences = {math: 12500}` with the canonical Feistel
  config.
- All three mixes share identical val-determining fields (only
  `train_weights` differs).
- For all 18 (base, mix) combinations of the small sweep, the rendered
  `data:` block is byte-identical to its reference.
- `MidtrainSpec` rejects `data_section_override` without provenance.
- `MidtrainSpec` rejects setting both `data_manifest_uri` and
  `data_section_override`.

If a future Levanter version changes the data-config schema or someone
swaps the math cache, this test trips and the launch refuses.

## Run plan

- Bases: 3e18, 9e18, 2e19, 3e19, 9e19, 2e20 (`DelphiModel` registry).
- Mixes: p33m67, p50m50, p67m33 (legacy references).
- LR factors: 0.33, 0.5, 0.67, 0.83.
- Budget: `BudgetPolicy.pretrain_fraction(0.20)`.
- Region: us-east5 (where the math cache + Delphi HF mirror live).
- Init: HF weights at pinned revision (`<repo>@<sha>`).
- TPU per base: v5p-8 for 3e18..3e19, v5p-16 for 9e19, v5p-32 for 2e20.
- Wall-clock per cell (v5p-8): ~33 min (3e18) -> ~28 h (2e20 on v5p-8;
  ~7 h on v5p-32).

Smoke before full sweep: one 3e18 x p33m67 x lr0.5 cell on v5p-8
(~33 min). Verifies HF init path, byte-identical data section, val loss
non-empty on the math partition. Sign-off ⇒ launch the remaining 71 in
per-base parallel waves with `--launch-spacing-seconds 30`.

## Decisions deferred / known constraints

- **Cache snapshot to TTL=infinity**: the math cache lives at
  `gs://marin-us-east5/tokenized/nemotron_cc_math_v1/4plus-2c5519` and
  has no TTL. If it ever gets reaped/rebuilt, the val partition silently
  shifts. Mitigation deferred to operator (copy to a pinned location or
  trust). The val-equivalence test catches schema drift but not cache
  content drift; a content fingerprint would require a one-time scan.
- **Levanter version pin**: the Feistel shuffle algorithm is part of
  Levanter's data path. Not currently pinned; verify the algorithm
  hasn't changed since the 1e21/1e22 runs before launching at scale.
- **Architecture sanity**: Delphi weights are Qwen3 (q_norm/k_norm)
  while HF configs say `architectures: ["LlamaForCausalLM"]` (upstream
  bug). Levanter's `initialize_from_hf` path uses the Levanter model
  class, so as long as `model_config["type"] = "qwen3"` (which the
  heuristic produces), the load works. Verify in smoke.
- **3e20**: not in this sweep (only "<= 2e20"). The 7th base is one
  added entry away if desired.
- **Throughput / step-time registry**: there is no central per-(model,
  TPU) step-time anchor in the repo. The closest thing today is the
  legacy `v5p_compute=(...)` tuple in
  `experiments/exp_delphi_math_10b_midtrain.py`, which is an opinionated
  allowlist with no benchmarked basis behind individual entries; the new
  `ALLOWED_TPUS_PER_BASE` map in
  `experiments/midtrain_specs/delphi_small_cpt_k020.py` is similar — it
  was hand-picked by HBM intuition, not measurement. Decisions like "is
  3e18 fine on v6e-4 or do we need v6e-8?" or "how long will the 9e19
  sweep take?" are currently answered by manually mining each base's
  pretrain W&B run for step time. A stub registry now exists at
  `experiments/throughput_stats.py` — pure data, never imported by the
  training stack — with a `ThroughputAnchor` schema, one seeded entry
  (3e18 on v5p-8, mid-flight measurement), and a CLI that estimates
  wall-clock from anchors. Future work: (1) walk each Delphi base's
  pretrain W&B run once and populate one v5p anchor per base; (2) record
  v6e anchors as soon as any cell completes on v6e-{4,8}; (3) optionally
  add an eval-overhead field so wall-clock estimates can include
  amortized eval cost instead of being train-only. The schema is
  intentionally measurement-keyed-by-W&B-run so any anchor is
  re-verifiable rather than folkloric.

## Tests + lint status

90/90 midtraining tests pass. `pyrefly` and `pre-commit` clean across
`lib/marin/src/marin/midtraining/`, `experiments/midtrain_specs/`,
`tests/midtraining/`, and `experiments/delphi_models.py`.
