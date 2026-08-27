# Splitting #8023 into a stack

[#8023](https://github.com/marin-community/marin/pull/8023) is one pull request carrying 100 files
and +20.6k lines against `origin/main`. This plan turns it into a four-layer stack plus three
independent pull requests, so that nothing has to be dropped to make the review tractable.

It is a planning document. No branch is restructured and nothing is pushed by it.

Its input is [pdf-campaign-partition.md](pdf-campaign-partition.md), which decides *what ships*. This
document decides *how what ships is delivered*. Where the two disagree, the partition manifest is
authoritative on content and this one is authoritative on layering.

Three decisions are taken as given and are not re-argued here: the router v2 training chain does not
ship, `converter_pool` is dropped, and cluster validation happens after the rewrite rather than
before.

## Summary

Seven pull requests: a four-layer stack that ends at #8023, and three independent pull requests that
never enter the stack because nothing depends on them.

```
main
 └── 1. flash-attn-4 constraint          (new branch)    ~6 lines
  └── 2. CUDA vLLM kernel artifacts      (new branch)    ~45 lines
   └── 3. datakit reuse + coordinator sizing (new branch) ~120 lines
    └── 4. the PDF pipeline              mark/pdf_pipeline  = #8023

independent, against main, in parallel:
     A. brokered serving-path concurrency limits          ~70 lines
     B. zephyr plain-http chunk scans                     ~20 lines
     C. grug MoE eval dtype cast                          ~6 lines
```

An eighth, `experiments/grug/moe/launch_pdf_compare.py`, needs a decision before it can be placed —
see [Open decisions](#open-decisions).

## The layers

File sets below are derived from `git diff origin/main...mark/pdf_pipeline`. The branch tip moves
while another agent curates it, so layers 1–3 and A–C are given as exact paths (they are stable and
the curation does not touch them) and layer 4 is given as a rule plus the partition manifest's SHIP
table.

### 1. Constrain flash-attn-4 so the workspace resolves

**Files.** `pyproject.toml` (5 added lines in `constraint-dependencies`), `uv.lock` (one line:
`{ name = "flash-attn-4", specifier = ">=4.0.0b16" }`).

`levanter[gpu]` requires `flash-attn-4`, which publishes only pre-releases in its 4.x line. Under
`prerelease = "explicit"` the specifier in `lib/levanter/pyproject.toml` does not opt in on its own,
because the requirement carries an extra (`flash-attn-4[cu13]`) and the opt-in is keyed on the bare
package name. `origin/main` today cannot re-resolve its own lock. That is a defect on `main` with no
PDF content, and it is the smallest reviewable change in the whole branch.

**Why it is a unit.** Six lines, one root cause, one verifiable claim: `uv lock` on `origin/main`
fails and succeeds with this entry.

**Why it is at the bottom rather than standalone.** Every layer above needs it in its ancestry.
Layer 4 changes `lib/marin/pyproject.toml` and therefore `uv.lock`; without the constraint present,
the lock and the manifests disagree and the lock check fails. Being at the bottom does not delay it:
`gh stack merge <pr> --yes` merges up to and including a given pull request, so this can land the
moment it is approved.

### 2. Let the CUDA vLLM launcher install prebuilt kernel artifacts

**Files.** `lib/marin/src/marin/inference/config.py`, `lib/marin/src/marin/inference/vllm_server.py`,
`lib/marin/src/marin/inference/vllm_backend.py`.

`VllmEngineConfig` gains `uv_with_packages` and `uv_extra_index_urls`, `IsolatedCudaVllm` threads
them into its `uvx --with` / `--index` command line and into `cache_identity`, and `vllm_backend`
passes them through. The motivating case is FlashInfer: CoreWeave images have no nvcc, so the kernels
have to arrive prebuilt from `https://flashinfer.ai/whl/cu130/` rather than being JIT-compiled at
startup. `__post_init__` rejects the fields on any non-CUDA launcher, and `cache_identity` folds the
index URLs in because distinct indexes serve different builds of the same requirement.

**Why it is a unit.** Serving-path code with no PDF content, reviewed by whoever owns
`marin.inference`. It has an existing test home: `tests/inference/test_vllm_server.py` and
`tests/inference/test_vllm_cache.py` already cover the launcher and its cache identity, and the
`cache_identity` change is exactly the kind of thing that wants a case there.

**Dependency.** Nothing below it is required. It is in the stack because layer 4 depends on it:
`experiments/datakit/build_pdf_source/ocr_extract/fleet.py` sets `uv_with_packages=FLASHINFER_PACKAGES`
and `uv_extra_index_urls=(FLASHINFER_INDEX,)`. Merged without this layer, layer 4 raises `TypeError`
at import of the fleet config.

Note that `VllmLauncherWithEnvironment` and the `start()`/`serve()` split, which the partition
manifest lists under this heading, are already on `origin/main`. Only the kernel-artifact plumbing
remains.

### 3. Let downstream pipelines reuse the reference pipeline's plumbing

**Files.** `lib/marin/src/marin/datakit/normalize.py`,
`lib/marin/src/marin/datakit/decon.py`,
`lib/marin/src/marin/processing/classification/consolidate.py`,
`experiments/datakit/decontam/config.py`,
`experiments/datakit/reference_pipeline.py`,
`experiments/datakit/cluster/quality/fast_transformer/train.py`.

Two changes that arrive together because they have the same shape and the same consumer.

The first is coordinator sizing. `normalize_to_parquet`, `decon_to_parquet`,
`build_all_source_drop_sets` and `consolidate` construct their own `ZephyrContext` and never expose
`coordinator_resources`, so every caller inherits Zephyr's 1 GB default. Over a source spread across
thousands of small files that default is an OOM kill at exit 137 near the end of a stage, discarding
the whole run. The three `experiments/datakit/` call sites that already size their coordinator
(`embeddings/harrier/pipeline.py`, `store/datakit_store.py`, `scripts/verify_fuzzy_dups_*.py`) do it
by building the context themselves, which is only available to callers that own the context. This
layer adds the parameter to the four step-level entry points and threads it through.

The second is de-privatisation of three helpers so a second pipeline can call them without copying
them: `normalize._make_split_writer` becomes `make_split_writer`,
`fast_transformer.train._binary_metrics` and `._save_scorer` become `binary_metrics` and
`save_scorer`, and the bloom step name plus the decontam parameters move out of
`reference_pipeline.py` into the `decontam/config.py` that already exists to hold shared decontam
policy. That last move is the load-bearing one: Marin step identity is name plus parameters, so a
second pipeline that wants the ~270 MB eval bloom as a cache hit rather than a rebuild must use
byte-identical values, and the only safe way to guarantee that is a single definition both import.

**Why it is a unit.** One sentence describes it: make the reference pipeline's building blocks
callable from outside it. The coordinator parameter is a defect fix with a named symptom; the rest is
a mechanical refactor with no behaviour change, which is the cheapest possible thing to review. It
touches the reference pipeline, so it wants the datakit owner's eyes, and burying it under 20k lines
of PDF code is how it currently avoids them.

**Dependency.** Layer 4 imports `make_split_writer` (`extract_ocr.py`), `binary_metrics` and
`save_scorer` (`quality/train_pdf_scorer.py`), and `BLOOM_STEP_NAME`, `EVAL_ROOT`, `NGRAM_LENGTH`,
`OVERLAP_THRESHOLD`, `ESTIMATED_DOC_COUNT`, `FALSE_POSITIVE_RATE` and `FLAGGED_SAMPLE_SIZE`
(`dedup.py`). Merged without this layer, layer 4 fails at import in five modules. It also passes
`coordinator_resources=` to `normalize_step` and `decon_step` from nine call sites, which would be
`TypeError`.

There is a case for splitting this into two — the defect fix and the refactor — and I argue against
it below.

### 4. Build the PDF-to-document pipeline for the focus crawl (#8023)

**Files.** The SHIP set of [pdf-campaign-partition.md](pdf-campaign-partition.md): the surviving
modules under `experiments/datakit/build_pdf_source/**`, their tests under `tests/datakit/`,
the `pdf` extra in `lib/marin/pyproject.toml`, the corresponding `uv.lock` entries, and
`infra/ci/select_tests.py`.

The file set is not frozen here on purpose. Another agent is curating `mark/pdf_pipeline` against the
manifest right now, and the tree has already moved under this analysis — `extract_inspector.py` and
`tests/datakit/test_extract_inspector.py` exist that did not when the manifest was written, and the
Docling tests are gone. The rule is what is stable: **layer 4 is whatever remains on
`mark/pdf_pipeline` after the manifest's CAMPAIGN rows are removed, minus the paths layers 1–3 and
A–C claim.**

Three paths currently on the branch belong in none of the seven and should be deleted rather than
assigned, because their only consumers are CAMPAIGN:

- `lib/marin/src/marin/inference/converter_pool.py` and `tests/inference/test_converter_pool.py` —
  dropped by decision.
- `lib/marin/src/marin/inference/broker.py` (`stats()`) and the `BrokerStats` /
  `BrokerStatsProvider` additions in `lib/marin/src/marin/inference/types.py`. Their only callers are
  `converter_pool.py` and `extract_fleet.py`, and both retire. Shipping them puts an unreachable
  method and two unused types into `marin.inference`. They should come back with the monitor that
  reads them.

One rename needs reverting for the same reason. Commit `a191db3c56` renamed
`worker._inference_error_response` to `inference_error_response` and documented it as "shared by the
forwarding worker and the converter pool". With the converter pool dropped, nothing outside
`worker.py` calls it, and pull request A (below) would be carrying an unmotivated public-API change.
Restore the private name.

**Why it is a unit.** It is the pipeline: an eight-step DAG whose stages share a record contract, a
routing decision and a schema, wired by `pipeline.py`. Splitting it by stage produces layers that
neither run nor test in isolation. If the curated diff is still large enough to need a further cut,
the seam that exists is `quality/`: `pipeline.py` imports `quality_label.py` at the top level but
nothing in the DAG imports the scorer-training chain (`build_oracle_sample.py`, `build_labels.py`,
`train_pdf_scorer.py`), which only produces the model `quality_label.py` loads by path. That chain
plus its two tests could become a layer 3.5. `quality/route_v2_features.py` cannot: `classify.py`
imports it directly.

### A. Size the brokered serving path's concurrency limits

**Files.** `lib/marin/src/marin/inference/proxy.py`,
`lib/marin/src/marin/inference/worker.py`,
`lib/marin/src/marin/inference/dashboard_server.py`.

Three library defaults an order of magnitude below the configured concurrency. `InferenceProxy` parks
a thread per in-flight request in `forward_raw_request`, and anyio's default `to_thread` limiter is
40 threads, so a brokered fleet was capped at 40 in flight regardless of how many workers sat behind
the broker; a 4-GPU serve drained at ~16 pages/s while the engines reported ~38 running. The worker's
forwarding `httpx.Client` and the dashboard's `/v1` `httpx.AsyncClient` both took httpx's default
100-connection pool, below both `max_in_flight` and a batch engine's `max_num_seqs`. Each is fixed by
sizing the limit from the caller's own budget. The same commit demotes per-request logging in the
proxy and worker to DEBUG, keeping INFO for batches that carry a dropped response or a non-200
status, because at hundreds of requests per second the request-id formatting is measurable work in
the proxy's event loop.

This is a documented defect class rather than three coincidences: `https://echo.oa.dev/wiki/259`
records four instances in this repository, the three here plus a fourth in an offline labelling job
with no relationship to serving. The pull request body should cite it.

**Why it is independent.** No layer imports anything this adds. Layer 4 needs the behaviour to run
the OCR fleet at its operating point, but not to import, test, or merge. `tests/evals/test_inference_proxy.py`
already exercises `InferenceProxy` and is where a regression test for the limiter belongs.

**Caveat.** As it stands this pull request also carries the `_inference_error_response` rename
discussed above. Drop it.

### B. Allow plain-http endpoints in polars chunk scans

**Files.** `lib/zephyr/src/zephyr/parquet_scan.py`, `lib/zephyr/tests/test_shuffle.py`.

`scan_parquet` builds an explicit `storage_options` block for CoreWeave virtual-hosted addressing.
`object_store` refuses a non-TLS endpoint unless told, and an explicit `storage_options` block does
not inherit `AWS_ALLOW_HTTP` from the environment, so the in-cluster LOTA endpoint — which is plain
http — failed every scan that went through this path. It killed a fleet extraction at the reduce
stage twice.

**Why it is independent.** It is a two-file correctness fix in `lib/zephyr`, reviewed by the zephyr
owner, and it has a direct test in `lib/zephyr/tests/test_shuffle.py`. Nothing above it imports
anything new. It should merge on its own timetable and does not belong in a PDF stack.

The other half of this work — `polars_io.py`, `external_sort.py` and `shuffle.py` from commit
`f0ecd62da` — has already reached `origin/main` through a merge. Only the `aws_allow_http` port
remains outstanding.

### C. Cast the eval model to the compute dtype

**Files.** `experiments/grug/moe/train.py`, `tests/test_grug_variant_contracts.py`.

`build_tagged_evaluator`'s loss closure receives the float32-parameter model from the callback and
evaluates it directly, while the train step evaluates in the compute dtype. Attention backends
including `gpu_fa4_cute` accept only bf16/fp16, so tagged evaluation crashes under any mixed-precision
policy that uses one. The fix threads `trainer.mp` into the builder and casts. Five lines of source
and one line of test.

**Why it is independent.** No PDF content and no dependency in either direction. It is a bug fix that
belongs on `main` today, ahead of anything else here.

## What each squash-merge message must say

`marin-community/marin` squash-merges, so each pull request description becomes a commit message on
`main`. Each should carry the fact a future reader needs and stop.

1. **flash-attn-4.** That `main` cannot re-resolve its own lock, and the exact reason: the extra on
   `flash-attn-4[cu13]` means the pre-release opt-in is not keyed to the bare name that
   `prerelease = "explicit"` checks.
2. **CUDA kernel artifacts.** That CoreWeave images have no nvcc, so FlashInfer must arrive prebuilt;
   and that the index URLs are in `cache_identity` because different indexes serve different builds
   of the same requirement.
3. **datakit reuse.** The exit-137 symptom and that it is not a function of shard count, plus the
   step-identity argument for why the bloom parameters have to be one definition rather than two
   equal ones.
4. **PDF pipeline (#8023).** What the pipeline is, its shape, its measured operating point, and the
   decisions the reports settle — extractor choice, router feature set, VLM operating point. It must
   also state what is deliberately not here, because reviewers of #8023 have already seen the wider
   change: the router v2 training chain, the Docling route and the converter pool are campaign
   history on `mark/pdf_processing`.
5. **A, concurrency.** The measured ceiling per site and the ratio that identifies the class, with
   `https://echo.oa.dev/wiki/259` linked.
6. **B, zephyr.** That an explicit `storage_options` block does not inherit `AWS_ALLOW_HTTP`, and
   that the failure mode is a dead reduce stage rather than a scan error.
7. **C, grug.** Which backends reject float32 and why the callback hands over an uncast model.

## What #8023 inherits

**#8023 stays on `mark/pdf_pipeline` and becomes layer 4, the top of the stack.** Its number, its
history, its review thread and its CI history are preserved. The three layers beneath it get new
branches and new pull requests.

This is also the only arrangement that avoids force-pushing a branch that is already under review.
`mark/pdf_pipeline` never gets rewritten; it only gains commits (the curation deletions the other
agent is landing) and a new base. GitHub computes a pull request's diff from the merge base of head
and base, so retargeting #8023's base from `main` to layer 3's branch shrinks its displayed diff to
layer 4's content alone, with no history rewrite — provided layer 3's branch is an ancestor of
`mark/pdf_pipeline`. Merging each layer branch forward into the one above it is what establishes
that, and merge commits are free here because they disappear in the squash.

Its description has to be rewritten. The current body describes the whole campaign; after the split
it must describe only the pipeline, and it becomes the squash commit message for the largest change
of the seven. Reviewers who already commented on #8023 should get a comment (prefixed `🤖` if an
agent writes it) explaining that the change was split, linking the three layers beneath and the three
independent pull requests, so that resolved review threads are not silently orphaned.

Its CI does not get cheaper. `.github/workflows/unified-unit.yaml` passes
`github.event.pull_request.base.sha` to `infra/ci/select_tests.py`, which diffs `base...HEAD` — so
each *lower* layer's CI narrows to that layer's own tests, which is a real gain. But #8023 sits on top
of everything, so its diff against layer 3 is still the whole PDF tree, and the merge commit CI runs
against a tree containing all four layers. The split buys reviewer attention, not runner minutes.

No workflow in `.github/workflows/` filters `pull_request` by base branch, so every stacked pull
request gets the full check suite despite not targeting `main`. Verified across all 35 workflows.

## Mechanics

### Prerequisites

```bash
gh extension install github/gh-stack     # already installed: v0.1.0
git config rerere.enabled true           # skips init's confirmation prompt
```

`remote.pushDefault` is not needed — this checkout has exactly one remote, `origin`.

Every `gh stack` invocation must be non-interactive: `view` always with `--json`, `submit` always
with `--auto`, and `init`/`add`/`checkout` always with explicit positional arguments. Without those
the command opens a TUI or prompts and hangs.

### Building the stack

Work in a worktree, not the shared checkout, and do not touch `mark/pdf_pipeline` until the
curating agent has finished.

The layer branches are cut by extracting each layer's paths from `mark/pdf_pipeline` onto a branch
rooted at `origin/main`, then merging each one forward. Because the layer content is being lifted out
of an existing branch rather than written fresh, `git checkout <ref> -- <paths>` against a branch
created from `origin/main` is more reliable than cherry-picking: several of these layers are subsets
of commits that also contain PDF content (`fe2bb6d04c` carries both the constraint and the `pdf`
extra; `a191db3c56` carries both the concurrency fixes and the rename to revert).

```bash
git fetch origin
gh stack init --base main \
  infra/flash-attn-4-constraint \
  inference/cuda-kernel-artifacts \
  datakit/shared-pipeline-plumbing

# Layer 1
gh stack bottom
git checkout mark/pdf_pipeline -- pyproject.toml
# hand-edit uv.lock to the single constraint line, or regenerate with `uv lock`
git add pyproject.toml uv.lock
git commit -m "[infra] Constrain flash-attn-4 so the workspace resolves"

# Layer 2
gh stack up
git checkout mark/pdf_pipeline -- \
  lib/marin/src/marin/inference/config.py \
  lib/marin/src/marin/inference/vllm_server.py \
  lib/marin/src/marin/inference/vllm_backend.py
git add lib/marin/src/marin/inference
git commit -m "[inference] Install prebuilt kernel artifacts in the CUDA vLLM env"

# Layer 3
gh stack up
git checkout mark/pdf_pipeline -- \
  lib/marin/src/marin/datakit/normalize.py \
  lib/marin/src/marin/datakit/decon.py \
  lib/marin/src/marin/processing/classification/consolidate.py \
  experiments/datakit/decontam/config.py \
  experiments/datakit/reference_pipeline.py \
  experiments/datakit/cluster/quality/fast_transformer/train.py
git add lib experiments
git commit -m "[datakit] Let downstream pipelines reuse the reference pipeline's plumbing"

gh stack submit --auto --open
```

`gh stack submit` cannot set a title or body. Fix both afterwards with
`gh pr edit <n> --title ... --body-file ...` and add the `agent-generated` label.

Then attach #8023 as the top layer. Do this with `gh stack link`, not `gh stack add`, and retarget
the base explicitly:

```bash
git checkout mark/pdf_pipeline
git merge datakit/shared-pipeline-plumbing    # makes layer 3 an ancestor; no force-push
git push origin mark/pdf_pipeline             # fast-forward
gh pr edit 8023 --base datakit/shared-pipeline-plumbing
gh stack link infra/flash-attn-4-constraint \
              inference/cuda-kernel-artifacts \
              datakit/shared-pipeline-plumbing \
              mark/pdf_pipeline
```

`gh stack link` is the right tool here for a specific reason: it is API-driven, keeps no local
tracking state, and pushes branch arguments **non-force and atomically**. `gh stack push` and
`gh stack submit` push with per-branch `--force-with-lease`. On brand-new branches that is a no-op,
which is why `submit` is fine for the initial creation of layers 1–3; on `mark/pdf_pipeline`, which
already backs an open pull request, it is exactly the force-push the repository forbids.

A, B and C do not enter the stack at all:

```bash
git switch -c inference/serving-concurrency-limits origin/main
# ... extract paths, commit, push, gh pr create --base main
```

### Updating a lower layer mid-review, without force-pushing

This is where `gh stack`'s own workflow and this repository's policy disagree, and the policy wins.
`gh stack rebase --upstack` and `gh stack sync` both cascade-rebase and then force-push every branch
above the one you changed. `AGENTS.md` forbids rebasing and force-pushing a pull request branch and
prescribes merging `origin/main` in instead.

Use plain git merges for every post-creation update:

```bash
# Review feedback lands on layer 2.
git switch inference/cuda-kernel-artifacts
# ... edit, commit
git push origin inference/cuda-kernel-artifacts        # fast-forward, no force

# Carry it up the stack. Merge, never rebase.
git switch datakit/shared-pipeline-plumbing
git merge inference/cuda-kernel-artifacts
git push origin datakit/shared-pipeline-plumbing

git switch mark/pdf_pipeline
git merge datakit/shared-pipeline-plumbing
git push origin mark/pdf_pipeline
```

Each pull request's displayed diff stays correct: GitHub recomputes from the merge base, and merging
the base branch in moves the merge base forward. `gh stack view --json` will report
`needsRebase: true` for the merged branches, because it tests whether the base is an ancestor by
linear history. That flag is expected and should be ignored — do not act on it by running
`gh stack rebase`.

The same merge-forward pattern absorbs `main` moving: `git merge origin/main` into layer 1, then
cascade upward.

### When a layer merges

Squash-merge collapses layer 1 into a single commit on `main` whose SHA does not appear in any
branch above. Do **not** run `gh stack sync` to recover from this — its squash-merge recovery path is
`git rebase --onto`, which force-pushes. Instead:

```bash
git fetch origin
git switch inference/cuda-kernel-artifacts
git merge origin/main            # layer 1's content arrives via main; the duplicate commits are inert
git push origin inference/cuda-kernel-artifacts
# cascade upward as above
```

GitHub retargets a pull request's base to the merged base's own base when the base branch is deleted,
so #8023's base walks down to `main` on its own as the stack drains. Confirm each retarget rather
than assuming it; `gh pr view <n> --json baseRefName` is the check.

For the merge itself, `gh pr merge` does not work on stacked pull requests. Use
`gh stack merge <pr-number> --yes --squash`, which merges bottom-to-top up to and including that pull
request, all-or-nothing.

### If stacks are not enabled on the repository

`gh stack submit` and `gh stack link` exit 9 if `marin-community/marin` does not have stacked pull
requests enabled. I could not determine this from the tree, and there is no read-only probe — the
first `submit` is the test.

This is not fatal. A GitHub stack is ordinary pull requests with non-`main` bases plus a grouping
object; the review scoping comes entirely from the base branch. On exit 9, create each layer with
`gh pr create --base <branch-below>` and link the layers to each other by hand in their descriptions.
Everything else in this plan is unchanged, including the merge order.

## Risks and ordering constraints

**`uv.lock` is the only file two layers touch.** Layer 1 adds one constraint line; layer 4 adds the
`pdf` extra's resolution. Nothing else in the seven touches it. Keeping A, B and C out of the stack is
partly for this reason — the fewer branches carrying lock deltas, the fewer conflicts when `main`
moves. If `main` lands an unrelated lock change, layer 1 takes the merge and the rest inherit it.

**Nothing in the stack leaves `main` broken if merged alone.** Each of layers 1–3 is additive: new
constraint entry, new optional dataclass fields, new optional keyword arguments plus renames whose
call sites all move in the same commit. Layers 2 and 3 introduce parameters with no in-tree consumer
until layer 4, which reads as dead configuration but is not a breakage. That is the honest cost of the
split and should be stated in each description.

**No layer's tests require a higher layer.** Verified against the current diff: layers 1–3 and A–C
add no tests that import `build_pdf_source`, and layer 4's tests import only downward. The reverse
risk is real, though — layer 4's tests fail without layers 2 and 3 present, which is why it is the
top and not a sibling.

**Merge order is forced in one place only.** Layer 4 after layers 1–3, for the import reasons given.
A, B and C have no ordering relationship to anything, including each other. Within 1–3 the order is
conventional rather than required; 1 is first because it is the one thing all of them want, and 2
before 3 only because 2 is smaller.

**`infra/ci/select_tests.py` raises CI cost repo-wide.** Layer 4 adds `pdf` to `UV_EXTRAS["marin"]`,
so every marin-leg unit run after it merges installs the PDF toolchain. As written that comment names
`pymupdf, docling, openvino, xgboost-cpu`, but the partition manifest retires Docling, OpenVINO and
NNCF. The extra must be curated before this lands or `main`'s CI pays for wheels nothing imports. This
is a review point on layer 4, not a layering problem.

**The curation is not finished.** `mark/pdf_pipeline` still carries `converter_pool.py`,
`test_converter_pool.py`, the `broker.stats()` addition and `tests/datakit/test_route_agreement.py`,
all of which the decisions or the manifest exclude. The stack cannot be built until the curating agent
has landed the deletions, because layer 4 is defined as the remainder.

**Cluster validation is deferred by decision**, so the stack will be opened against a pipeline that has
not been re-run end to end since the curation. Layers 1–3 are individually verifiable without a
cluster: layer 1 by `uv lock`, layer 2 and 3 by unit tests. Layer 4 is not, and its description should
say so rather than imply a validated run.

## How many pull requests, and why seven

**Seven** — four stacked, three independent — with an eighth pending a decision.

The argument against fewer. Three or four total would mean folding A, B and C into their nearest
neighbours: the concurrency fixes into layer 2 (both are `marin/inference/`), the zephyr fix into
layer 3 (both are pipeline infrastructure), the grug fix into layer 4 (both are `experiments/`). Each
of those merges produces a squash commit that has to explain two unrelated things, which is the exact
failure #8023 exhibits at scale. It also sends three changes to reviewers who have no interest in
them: the zephyr owner would be reading FlashInfer plumbing to get to a two-file `storage_options`
fix.

The argument against more. The two further cuts available are splitting layer 3 into its defect fix
and its refactor, and splitting layer 4 by pipeline stage. The first is not worth it — six files,
~120 lines, one sentence of intent, and the refactor half exists only to serve the defect fix's
consumer. The second is worse than not splitting: `pipeline.py` wires the stages into one DAG and the
tests cross module boundaries, so a per-stage layer neither runs nor tests, and every layer would need
rebasing every time the curation moves a file. The one further cut that is defensible is
`quality/`'s scorer-training chain, held in reserve if layer 4 is still unreviewable after curation.

The argument for the specific shape. Review overhead is not additive across independent audiences.
Seven pull requests with four reviewers, none of whom reads more than their own area, is less total
attention than one pull request that forces every reviewer through 20.6k lines to find their 70. The
stack exists only where a real import dependency exists; everything else is parallel, so wall-clock is
set by the slowest review rather than by the sum.

## Open decisions

1. **`experiments/grug/moe/launch_pdf_compare.py` (380 lines).** It has no dependency on the PDF
   pipeline — it imports nothing from `build_pdf_source` and reads its corpus from a hard-coded S3
   path. It does depend on C, since it sets `_GPU_ATTENTION = "gpu_fa4_cute"` with
   `mp="params=float32,compute=bfloat16,output=bfloat16"`, which is the configuration C's cast fixes.
   The complication is that its `PDF_FINAL_DIR` points at
   `common_crawl_focus_2026_22_pdf_ocr_all_e4e8dda6` — the all-routes study corpus, produced by
   `extract_ocr_all.py` and `finalize_ocr_all.py`, which the partition manifest classifies CAMPAIGN.
   So it measures a corpus the shipped pipeline does not produce. Three options: ship it as an eighth
   pull request stacked on C; keep it campaign-only; or ship it repointed at the shipped pipeline's
   output once that output exists. This needs a call.
2. **Does layer 3 split?** The recommendation is no. If the datakit owner would rather review the
   coordinator-sizing defect separately from the de-privatisation refactor, it splits cleanly along
   `lib/` versus `experiments/` — but `make_split_writer` sits on the `lib/` side of that line and
   belongs with the refactor.
3. **Are stacked pull requests enabled on `marin-community/marin`?** Undeterminable from the tree;
   the fallback is described above.
4. **Who reviews layers 2 and A?** Both are `marin.inference`. If they are the same person, there is a
   case for making A a fourth stack layer between 2 and 3 to give that reviewer one thread instead of
   two. I have kept them separate because A has no dependency in either direction and gains nothing
   from waiting.

## What I could not determine

- Whether `marin-community/marin` has GitHub stacked pull requests enabled.
- The final file set of layer 4, because the curation is in flight. The rule is given instead of a
  list, and the exact paths of layers 1–3 and A–C are given because they are stable under it.
- Whether `tests/datakit/test_route_agreement.py`, still present on the branch, survives the
  curation. The manifest sends it to CAMPAIGN; if it stays, layer 4 carries a test for a retired
  target.
- Whether branch protection on `main` requires a review before merge, which would set whether
  `gh stack merge --yes` can drain the stack in one call or has to wait per layer. `gh stack merge`
  checks only that each pull request is open and not a draft, and cannot bypass merge requirements.
