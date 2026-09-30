# Private runtime image migration revalidation

`capability_pipeline.image_migration` handles the narrow case where a completed
task needs a portable private verifier image without another construction
repair. It is not a retry-budget override.

The validator requires the reviewed bundle-manifest digest, migration-receipt
digest, and exact target registry digest. It reconstructs the complete terminal
source from the retained pull manifest and staged pre-migration files. It then
proves that the only semantic TaskSpec change is
`steps[0].verifier.verifier.runtime.image`, that both lowered manifests and the
generator constant agree, and that every other task, grader, control, accepted
proposal, status, session, repair, and repair-history artifact retains its
source hash. Both consumed repair-attempt trees must be present.

The executor requires a new output directory. It restores the complete
controller state, archives the prior status, Harbor package, runtime evidence,
quality evidence, and controller reports under `image-migration-history/`, and
calls the maintained `_synthesize_attempt` gate path exactly once. Completed
construction sessions are reused; no builder or semantic repair is invoked.
All protected task artifacts and both repair trees are checked again after the
gate attempt. A pending or failed gate remains pending or failed. Exit status 0
is reserved for `quality_accepted`; other retained outcomes exit 2.

The archived status remains historical evidence, not the current controller
state. Immediately after preparation, the live item status is
`pending_image_migration_revalidation`; while the one maintained gate attempt
runs it is `image_migration_revalidation_running`. Both states link the archived
status path and hash plus the controller operation receipt. A raised exception
closes both records as `image_migration_revalidation_error`. The Iris job state
remains authoritative for whether the worker is active, while these receipts
identify which operation is active and prevent an old failed status from being
mistaken for its outcome.

The reviewed c05 bundle is `data/c05-portable-runtime-006`. Its prepared launch
shape is:

```sh
uv run --project "$CAPABILITY_PIPELINE_ROOT" --frozen python \
  "$CAPABILITY_PIPELINE_ROOT/scripts/run_image_migration_revalidation.py" \
  --bundle "$CAPABILITY_PIPELINE_ROOT/data/c05-portable-runtime-006" \
  --out "$RESULTS" \
  --expected-image 'docker.io/library/python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea' \
  --expected-bundle-manifest-sha256 6fa887f054f6aae1ec89d26f142d3e9742481e314cc5221d9e3680daa815c175 \
  --expected-migration-receipt-sha256 8dbaae73d28bae73b4e7a1303d63422243c985ae90d5b37dfd16f406fb34d2c8 \
  --taskcompendium-source "$TASKCOMPENDIUM_SOURCE" \
  --daytona-tools "$CAPABILITY_DAYTONA_TOOLS" \
  --research-overlay "$CAPABILITY_OMP_CONFIG" \
  --model glm-orion/glm-5.3 \
  --session-time 28800 \
  --max-continuations 8 \
  --validation-timeout 14400
```

The historical failed status is not an acceptance result for the migrated
image. Revalidation 006 stopped at an Iris tree-transport integrity check before
runtime. Revalidation 007 used the reviewed bundle through the opaque,
hash-bound archive transport and completed the maintained gate attempt as
`pending_solver_adjudication`. Its immutable terminal pull is
`runs/synthesis-c05-image-migration-revalidation-007/terminal-pull-0829`
(snapshot `17d1a38f0f534c108a6a7c44ab4defb7`, 1,143 members, pull-manifest
SHA-256 `2ccb140afedbc7dc24e662e1d38d3ce4f7d6f5ba5615a5a28edcc1c3b843a042`).
Both independent solver attempts for each of the two positive controls received
graded reward 0. All five authored malformed/adversarial controls and all three
fresh independent attacks also received 0; the bounded attack suite passed.
One authored oracle replay earned 1, but the runtime planner collapsed the two
same-step authored positives when constructing its oracle suite, so the
controller correctly also reported incomplete exact positive coverage. The
planner regression was fixed after this frozen run; the terminal artifacts and
outcome remain unchanged. Revalidation 007 did not enter quality review and is
not task acceptance.

The maintained packed-worker entrypoint is:

```sh
scripts/submit.sh \
  --stage image-migration-revalidation \
  --source data/c05-portable-runtime-006 \
  --out runs/synthesis-c05-image-migration-revalidation-007 \
  --run-name cap-synthesis-c05-image-migration-revalidation-007 \
  --concurrency 1 \
  --dispatch-limit 1 \
  --tier interactive
```

Submit-side preflight validates the actual reviewed bundle and archives every
manifest-declared input member. The worker verifies the same manifest and
receipt hashes, then performs a read-only provider lookup that requires the
active `cap-verifier-908214b9e11806813050` cache to expose the exact reviewed
Dockerfile recipe in provider `build_info`. It does not create, delete, or
replace a cache during preflight. The worker then invokes the standalone
executor with `uv run --project … SCRIPT.py` and a 14,400-second complete gate
timeout. Daytona and model credentials remain process environment variables;
no registry credential is staged or requested.
