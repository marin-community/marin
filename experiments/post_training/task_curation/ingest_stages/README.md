# atlas-ingest and atlas-repair

These commands qualify one source through the same sealed artifact graph. Each
has its own eligibility gate, checked before a queue claim, issue comment or
payload transfer:

- `atlas-ingest` accepts an unknown source, absent from both the Atlas catalog and
  its maintained pending intake and metadata exclusions.
- `atlas-repair` accepts an existing, immutable pinned Excluded source with typed
  Reasons. Its configured dataset and revision must match the declaration.

An Available, Pending, Deprecated, unpinned or reasonless source cannot enter
repair. A known source cannot enter ingest. These commands do not change the
status taxonomy. Declaration status follows the measured receipts.

## Configure one source

Work in a dedicated Marin worktree and run commands from its root. A reviewed
config lives in `ingest_stages/configs/`; its schema is `IngestConfig` in
`ingest_config.py`. For example:

```json
{
  "source_id": "Nvidia:Nemotron-RL-litmus-bench-v0.2/train",
  "upstream": {
    "repository": "nvidia/Nemotron-RL-litmus-bench-v0.2",
    "revision": "c180425444ebcc6db8dafdb897e51a94927df62a",
    "path": "train.jsonl"
  },
  "intended_use": ["train"],
  "converter_module": "experiments.post_training.task_curation.datasets.litmus.convert",
  "cluster": "cw-us-east-02a",
  "mode": "ingest"
}
```

Each config has exactly one canonical intended use: `train` or `eval`. Separate
uses need separate configs and source identities. Counts, output paths, registry
pointers, license decisions and acceptance claims are measured or resolved from
authoritative state. The converter exports `INGEST_CONVERTER`, a
`RowConverterBinding` containing a row converter, source format, task format,
grader environment and positive control. Every input, including existing
TaskSpecs and Harbor tasks, passes through QUICK TaskSpec conversion and dedup.

The companion `configs/code-contests-repair.json` requalifies the existing
CodeContests converter output. It preserves the task and grader contracts and
records the new ingestion identity. A dry run supplies no grader verdict;
actual controls require reproducible environments, including hashed dependency
locks. An unpinned workspace install must be fixed before trusting those controls.

## Preflight and dry-run

Install the workspace's normal CPU/test environment with `uv sync --extra cpu
--group test`. Infrastructure credentials (`HF_TOKEN`, `CW_KEY_ID`,
`CW_KEY_SECRET`) belong in the operating environment. Issue writes use
`GITHUB_TOKEN`, or the authenticated `gh` session. Never put credentials in config.

```bash
uv run --no-sync atlas-ingest preflight \
  --config experiments/post_training/task_curation/ingest_stages/configs/litmus-train.json
uv run --no-sync atlas-ingest dry-run \
  --config experiments/post_training/task_curation/ingest_stages/configs/litmus-train.json
uv run --no-sync atlas-repair dry-run \
  --config experiments/post_training/task_curation/ingest_stages/configs/code-contests-repair.json
```

Both local modes capture small catalog, license, queue and registry metadata.
They type-check the stages and invoke the regional worker's exact argv with
local metadata references. This exercises parsing, imports, source attestation,
eligibility, converter loading, GPU placement and graph construction. It does
not run graders, scan dataset payloads, call models, mutate the queue or submit
jobs. The JSON output includes every stage fingerprint and all planned job names.
Production stage behavior is covered separately by behavior tests and a regional run.

The equivalent module entrypoint is:

```bash
uv run --no-sync python -m experiments.post_training.task_curation.ingest_stages \
  ingest dry-run --config experiments/post_training/task_curation/ingest_stages/configs/litmus-train.json
```

`--queue <path>` overrides the maintained `CANDIDATES.md` intake queue.
Ingest requires exactly one matching row and refuses conflicting pins or issue
links. Repair derives its work claim from the existing declaration and Reasons;
it leaves the new-source queue unchanged. `--watchdog-feed <path>` and
`--landing-log <path>` select the shared alert feed and atlas-push landing log.
No local scheduler or background polling service is installed.

## Run and resume

Land the converter, config and command code through `atlas-push` first. `run`
requires a clean checkout and proof that its exact source revision landed.
Register the planned job names with the coordinator before submission.

```bash
uv run --no-sync atlas-ingest run \
  --config experiments/post_training/task_curation/ingest_stages/configs/litmus-train.json
```

Replace `atlas-ingest` with `atlas-repair` for the repair config. The host freezes
execution metadata before claiming work. It submits interactive CPU roots in
`cw-us-east-02a`, where all upstream downloads, conversion, census, overlap,
lexical scans and private publication run. Dataset payloads never pass through
the workstation. SAMPLE has its own root on `cw-us-east-08a` for the pinned default
reviewer's GB200 requirements; only its selected panel crosses to that region.

The stages are claim, license, upstream staging, conversion, dedup, census,
fingerprint/overlap, backend/oracle controls, lexical scan, SAMPLE and declaration.
The license assessment precedes acquisition. Population resolution checks the
exact pin, digest and declared count before opening shards. Fingerprint inputs
map file URIs to SHA-256; locators stay in the data region. Overlap uses the
maintained CI implementation. Reference controls must pass and noop controls
must fail. Unknown or unsupported grading outcomes refuse qualification.

Training contamination uses the existing full raw archive and mechanical
retention publishers. Evaluation keeps its measured benchmark population.
Changed retained bytes receive a fresh overlap survey. Previous populations
remain immutable evidence; only the new exact pin and locators are registered.
No later survey discovers populations by a staging-folder convention.

SAMPLE is informational. At most 5% no-verdict observations qualify a SAMPLE
receipt; provider failure or an inconclusive panel records no passing receipt
and does not block admission. Corrupt observations fail verification.

Re-running the same code and config reuses its frozen execution and sealed stage
receipts. Changed or missing evidence refuses reuse. A changed config or source
revision derives a new graph identity. Source registration and license/index
updates are immutable documents selected through Git pointers. `atlas-push`
checks their expected base inside its landing lock, then runs catalog tests,
source-fingerprint CI and the declaration audit. A moved shared pointer requires
rebasing and a new survey; entries cannot be dropped by replacing an index.
The requester receives a policy-marked pickup and terminal comment when its queue
row identifies an open issue.

See [INVENTORY.md](INVENTORY.md) for the maintained-helper mapping and historical
script retirement policy.
