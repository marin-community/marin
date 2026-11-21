# Levanter quick-loop status

November 20, 2026 (PST).

The learner-only integration and CPU proofs are ready. The first 40-H100 smoke,
from `646ccb48641442cf3d9b3223f06c1bca0c5a2e0a`, failed before model loading:
the GPU learner inherited `JAX_PLATFORMS=cpu` from its CPU coordinator.
`fc9a60939b4bc40b34b4dce7ffdf3e3d33f47d70` fixes platform selection and checkpoint
locality. Current-source GPU acceptance remains pending. The replacement smoke
is submitted at BATCH priority from `6936528c55df0212ec8ccd1fa619361863b4c420`.
At 16:03 PST, all five GPU tasks are still waiting for gang admission.
The matched comparison remains gated on smoke acceptance.

## Scope and fixed recipe

[The quick-loop CLI](train_quick_loop.py) retains SkyRL orchestration, its
RolloutEngine/vLLM generation, task graders, evaluation, and coordinator-computed
RLOO-Neutral (RLOO-N) advantages. `--learner levanter` replaces the learner with
one isolated JAX process per learner host. It consumes finalized SkyRL batches,
including failure masks and fractional loss weights. Full native RolloutEngine
orchestration migration remains outside this integration.

Both arms preserve these inputs and settings:

- Selection: `s3://marin-us-east-02a/marin/rl-data/task-selection/snowball-100k/2026.11.04`
  ([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Frl-data%2Ftask-selection%2Fsnowball-100k%2F2026.11.04)).
  Read-only preflight verified its published manifest, 100,000 training rows,
  and fixed 100-task evaluation panel: 20 each from calendar, competitive coding,
  reasoning gym, STEM, and structural tasks.
- Initializer: `open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21`, revision
  `fe666478454e79fe6027a0bab0388d9abd68f51a`, from
  `s3://marin-us-east-02a/models/open-athena--Grug-67B-A2B-Datakit-SFT-262K-2026.09.21`.
  Preflight matched all 39 weight-file LFS hashes and sizes to that Hub revision
  and verified tokenizer/config metadata; it did not download all weights locally.
- Synchronous RLOO-N: 128 prompt groups × four answers per update, zero staleness,
  infrastructure failures excluded, agent failures retained at zero reward,
  zero-reward groups filtered, and no batch advantage normalization.
- Base optimizer: AdamW, learning rate `1e-6`, betas `(0.9, 0.999)`, weight decay
  `0`, gradient clip `0.5`, constant schedule with zero warmup. One optimizer
  pass per batch, token-mean clipped PPO with symmetric `0.2` clipping; KL and
  entropy terms disabled.
- Allocation: four eight-H100 learner hosts and one eight-H100 inference host in
  `cw-us-east-02a`; the native mesh is data4/expert8. Automatic failure retries are zero.
- Rollouts: the same task/grader definitions and backends, temperature `1.0`,
  top-p `1.0`, 24,576-token request window, and 8,192-token response limit.
  Evaluation uses the fixed panel before and after training, seed `20261020`,
  temperature `0.6`, and top-p `0.95`. Reasoning-marker penalty remains `0.1`;
  overlong penalty remains zero.

Rendered recipe comparison against main
`fda7484bbfe957c89d08b7d58dbaac3396d2f4bc` found only `trainer.strategy`,
`trainer.batch_builder`, and `trainer.policy.levanter_config` differ between the
instrumented control and native candidate. Both verify publication. The native
profile keeps FA4 attention, ring-gather/combine MoE, FP32 routing and selective
attention operations, and BF16 projection logits with FP32 softmax. The platform
and storage fixes change no kernels, objective math, precision settings, or
numerical tolerances. See [policy learning](../../../docs/references/policy-learning.md#skyrl-quick-loop).

## Evidence and acceptance

| Gate | Observed result | Remaining proof |
| --- | --- | --- |
| Source and CPU | JAX/JAXLIB `0.11.0` data4/EP1 proof: exact state continuation, 78 publication payloads / 232,384 bytes, all eight child exits zero. Four-device Snowball regression: 35 passed, two skipped. | CPU checks do not qualify CUDA, H100, EP8, or the full online loop. |
| Actual H100 | First 40-H100 smoke started all five GPU tasks, then failed the learner device-count guard before model loading. | Current-source 32-GPU Snowball initialization, scoring, and two real parameter updates. |
| Publication and resume | CPU Gloo replay and fresh-process native checkpoint transactions passed, including exact model, Adam, RNG, step, and model-averaging fingerprints. Four native child/host restart cases passed. | H100 NCCL/vLLM receiver replay with zero byte mismatches, subsequent nonempty generation stamped with the updated policy, terminal checkpoint, fresh learner processes restoring exact state, and a further update. |
| Matched 25-update comparison | Both arms are configured; neither comparison has run. | Same inputs, optimizer, rollout/grading and evaluation exposure; measure training reward and fixed-panel evaluation separately, plus groups/sequences/tokens, wall time, and environment/tool failures, with terminal receipts and archives. |

The earlier November 4 single-host, four-H100 toy-pipeline acceptance does not
qualify this current 32-GPU full-MoE learner source.

The failed smoke was version `2026.11.20.1`, coordinator
`/dlwh/users-dlwh-checkpoints-snowball-inventory-2026.11.20.1-coordinator-96655281`,
with GPU child suffix `users-dlwh-checkpoints-snowball-inventory-2026.11.20.1-a02f6689f4d1`.
[W&B run](https://wandb.ai/marin-community/marin-snowball-curriculum/runs/uzjtcvcc).
It failed at 12:28:12 PST; the actual child configuration confirmed the CPU
selector. No successful model update, publication, generation, or checkpoint
continuation receipt resulted. Previous GPU pods were confirmed gone.

The source fixes explicitly select `JAX_PLATFORMS=cuda,cpu`, preserve Ray's
`CUDA_VISIBLE_DEVICES`, and require the actual GPU backend before loading
weights. Bootstrap checks CUDA in the isolated learner environment before Ray
starts. Both Megatron and Levanter now place recovery checkpoints, Ray logs,
and rendezvous state in lifecycle-managed 02a scratch storage beside compute,
overriding shared 08a or ambient temporary-storage defaults. See
[RL storage and retention](../../../docs/references/rl-launching.md#artifact-and-runtime-boundaries).

Relevant source pins:
[checkpoint validation, `646ccb4864`](https://github.com/marin-community/marin/commit/646ccb48641442cf3d9b3223f06c1bca0c5a2e0a),
[regional coordinator route, `32123524a1`](https://github.com/marin-community/marin/commit/32123524a13df6f03385f58114466c55672929b0),
and [GPU selection / 02a storage, `fc9a60939b`](https://github.com/marin-community/marin/commit/fc9a60939b4bc40b34b4dce7ffdf3e3d33f47d70).
Focused native, bootstrap, SkyRL, required lint/type checks, strict docs, and
independent source review passed for these fixes. The earlier broad suite had
17 failures: six sandbox and ten router failures reproduced on clean main;
one nested-launcher setup failure remained unclassified.

## Current run and reproduction

The active smoke is version `2026.11.20.5`, coordinator
`/dlwh/users-dlwh-checkpoints-snowball-inventory-2026.11.20.5-coordinator-2ea612c2`,
with current GPU child suffix `users-dlwh-checkpoints-snowball-inventory-2026.11.20.5-af5857383fd5`.
Both accepted source bundles match the frozen `6936528c55` contents, and both
jobs have actual BATCH priority. The coordinator was preempted once before GPU
execution; Iris killed its first child and submitted the current replacement.
The current gang remains scheduling-gated. Failure retries are zero; Iris
preemption retries retain their defaults of 100 for the coordinator and 1,000
for the child. These are separate retry budgets.

Run from the repository with the required storage, evaluation, and judge
credentials loaded. The commands below record the submitted smoke and planned
comparison arms; choose a fresh unused artifact version for another attempt.
Each uses 40 H100s at BATCH priority with zero automatic failure retries. The regional route
submits directly to the configured CoreWeave context. The smoke restarts JAX
learners within the same allocation to validate native checkpoint continuation.
It retains the parent actors and rollout engines.

```bash
SELECTION_URI=s3://marin-us-east-02a/marin/rl-data/task-selection/snowball-100k/2026.11.04
uv run python -m experiments.post_training.inventory.train_quick_loop \
  --selection-uri "$SELECTION_URI" --version 2026.11.20.5 \
  --learner levanter --steps 2 --verify-publication \
  --validate-native-checkpoint --coordinator-route regional --run
```

After the smoke passes initialization/update, publication/generation, and
checkpoint continuation gates in the table, submit the matched arms:

```bash
uv run python -m experiments.post_training.inventory.train_quick_loop \
  --selection-uri "$SELECTION_URI" --version 2026.11.20.6 \
  --learner megatron --steps 25 --verify-publication --coordinator-route regional --run
uv run python -m experiments.post_training.inventory.train_quick_loop \
  --selection-uri "$SELECTION_URI" --version 2026.11.20.7 \
  --learner levanter --steps 25 --verify-publication --coordinator-route regional --run
```

Checkpoint restart validation is disabled for both comparison arms. Record the
executed source/bundle pins and run links after submission, then inspect the
acceptance receipts and output archives. Native terminal HF export remains
unimplemented. No throughput or reward comparison is available yet.
