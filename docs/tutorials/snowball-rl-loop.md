# Run a Snowball RL development screen

The Snowball loop binds an immutable task selection, a pinned SFT initializer,
a training recipe, and initial/final evaluation. Its terminal stage publishes
one immutable attempt to the private RL progress dashboard. A failed attempt
remains failed even when it has useful intermediate measurements.

## Runtime admission

The fixed recipe requires the qualified Triton attention helper digest
`fe39ea5a707a1979f9576595f580a2f70506a5868a389ee150202b45963b5f03`.
The currently pinned SkyRL vLLM wheel does not contain those bytes. Publish and
adopt a fixed wheel and frozen lock before submitting this recipe. The worker
checks the installed helper before tokenizer/model/placement setup; this check
occurs after Iris allocation. It never patches installed source. A passing
source check alone does not attest the complete wheel or a successful GPU run.

The historical four-arm screen qualified its frozen runtime with a temporary
source overlay. Those jobs are complete; they are not a qualification of a
newly assembled runtime, and must not be resumed to test this integration.

## Plan an experiment

Use a durable checkout and a `codex/` branch. Follow the repository's launch and
pull-request skills. Keep credentials in the supported secret delivery path;
never put values in a recipe, report, command history, or pull request.

Build without `--run` first:

```bash
uv run python -m experiments.post_training.inventory.train_quick_loop \
  --version YYYY.MM.DD --arm a --steps 25
```

The default selection is `snowball-100k/2026.10.23.2`; the pinned initializer is
Datakit SFT revision `fe666478454e79fe6027a0bab0388d9abd68f51a`.
The recipe uses 40 H100s, 24,576-token context, 8,192-token replies,
128 prompt groups per update and four training answers per group. Evaluation
uses the complete declared 100-task panel with seeded greedy decoding.

| Arm | Change from control |
| --- | --- |
| `a` | Synchronous RLOO-N, learning rate 1e-6, token mean, age zero, inflight 128 |
| `b` | Learning rate 2e-6 |
| `c` | Sequence-mean loss; also changes effective gradient scale |
| `d` | Async scheduling, maximum age two, inflight 192, full batches |

All runs start from SFT, disable resume and HF export, and retain native
checkpoints. Use a fresh immutable version for each experiment and runtime
change. Do not shorten budgets to make a failing configuration pass.
Verify model/data identities, physical geometry, source/runtime pins and the
terminal publication stage in the rendered plan before adding `--run`.

## Execute and retain outcomes

An authorized launch uses the same command with `--run`. Check startup and
optimizer progress through Iris, then use heartbeat monitoring. Do not cancel,
restart, or resubmit a failed run without authorization. The loop does not
retry training or regenerate answers to repair evaluation.

Incomplete declared baseline scoring raises after preserving evaluation dumps,
before optimization. Runtime callbacks retain numeric observations in
`exports/loop-evidence.json`, including optimizer metrics, evaluation coverage,
and the exact W&B URL when available. Terminal processing checks update
sequence, finite applied windows, scoring coverage, native outcome and the
final checkpoint marker before writing `exports/loop-result.json`.

The report explicitly leaves unattested runtime task/protocol identities and
allocated GPU-hours unknown. Such records remain descriptive and unranked.
Elapsed time covers the blocking launcher, not coordinator submission or
measured GPU allocation. Callback metrics are observations, not independent
per-token accounting. Review these limits before comparing results.

## Publish and recover publication

Successful training is cached separately from publication. If dashboard
publication fails after training succeeds, retry the terminal publication stage
with the same unchanged recipe/version; the completed training dependency is
reused. For failed training, use its saved summary directly rather than
rerunning the training command:

```bash
uv run python -m experiments.post_training.inventory.publish_loop_result \
  /absolute/path/to/loop-result.json
```

Download only that small new-attempt summary through maintained `fsutil` when
it lives in object storage. The publisher validates the record, inserts only
if its run/attempt key is absent, and checks exact SQL and API readback. An
identical retry succeeds; conflicting content raises instead of overwriting
history. Publish terminal records only. Resolve invalid evidence by preserving
the failed attempt and proposing a fresh experiment, not editing its scores.

## Change the baseline

Keep the initial checkpoint, task selection, evaluation/scoring/decoding policy,
budget, source and runtime identities explicit. Record the intended treatment
before launch. A changed eval policy starts a separate comparison group; fast
panel results guide search, while baseline promotion requires an agreed Evals
confirmation gate. A single-seed development panel is not broad quality or
async noninferiority evidence.

Publish the branch with focused behavioral tests, the affected safe checks and
required review. A merge does not publish a runtime wheel or release a Marina
package. Dashboard source and its separate release procedure live in
`infra/marina/apps/rl_progress/`. Independent operator acceptance still requires
a fresh authorized run from the resulting pinned runtime, with a verified
terminal report and no manual evidence reconstruction.
