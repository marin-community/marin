# Per-User GCS Storage Prefix

_Why are we doing this? What's the benefit?_

Marin's GCS paths today flow through a single regional `MARIN_PREFIX` and every user writes into the same `experiments/` namespace, so there is no per-user attribution and no foundation for per-user quotas or usage reports. Model checkpoints — the dominant storage cost and the easiest artifact to leave behind by accident — sit inside hash-named step output dirs that nobody owns. We add a thin fsspec filesystem (`users://`) that resolves the same path against a per-user "personal" layer first and the existing shared root second; training step factories opt in by pointing their `override_output_path` at a `users://` URL. The executor, `StepSpec`, and downstream consumers stay unchanged — they just see fsspec URLs.

## Background

`marin_prefix()` ([`lib/rigging/src/rigging/filesystem.py:136`](https://github.com/marin-community/marin/blob/ea00d29f1/lib/rigging/src/rigging/filesystem.py#L136)) is the single hook every output path flows through. The closest in-repo precedent is `MirrorFileSystem` ([`filesystem.py:804`](https://github.com/marin-community/marin/blob/ea00d29f1/lib/rigging/src/rigging/filesystem.py#L804)) — an fsspec filesystem (`mirror://`) that resolves cross-region reads by probing the local marin bucket, scanning other regional buckets, and copying on first access under a `TransferBudget`. The personal-vs-shared lookup proposed here is the same algorithm (probe layer 0, fall back to layer 1, resolve to a concrete URL), but applied across prefixes inside one bucket and with no copy step — the goal is attribution, not transparent fetch. Iris already has a complete user-identity surface (`IrisClient.submit(user=)`, `iris job run --user`, `get_job_info().user`, `getpass.getuser()` fallback) per `.agents/projects/20260301_iris_user_job_names.md`. `StepSpec.override_output_path` ([`step_spec.py:59`](https://github.com/marin-community/marin/blob/ea00d29f1/lib/marin/src/marin/execution/step_spec.py#L59)) already exists as a per-step path override and accepts arbitrary URL strings. Full digest in [`research.md`](./research.md).

## Challenges

The hard part is keeping executor changes minimal. The executor's cache-hit predicate at [`executor.py:1448`](https://github.com/marin-community/marin/blob/ea00d29f1/lib/marin/src/marin/execution/executor.py#L1448) probes `{output_path}/.executor_status` and decides "step is done" purely from that file. If checkpoints live under `users://` but `.executor_status` lives at the shared root, a teammate's run would short-circuit my executor while my eval can't find their checkpoint. We resolve this by making the whole training step output dir live behind `users://` (status file + checkpoints + logs), so the layered lookup naturally answers both "is this step done for *me*" and "where are *my* checkpoints" with the same probe order.

The executor needs *one small change* to support this: today `Executor.compute_version` ([`executor.py:1504`](https://github.com/marin-community/marin/blob/ea00d29f1/lib/marin/src/marin/execution/executor.py#L1504)) hard-codes the executor's global `self.prefix` when joining `{prefix}/{name}_{hash}`. We add an optional `output_path_prefix` field on `ExecutorStep` and a one-line preference: `prefix = step.output_path_prefix or self.prefix`. After that change, the executor itself is unaware of `users://` — it joins the URL like any other prefix and fsspec handles the layering.

The second challenge is making sure `users://` doesn't accidentally swallow paths that aren't supposed to be per-user. We address this by making it strictly opt-in: only training step factories set `output_path_prefix="users://"` on the `ExecutorStep` they return. Ferries, datasets, evals, scratch — all continue to use `gs://`/`marin_prefix()` directly, unchanged.

## Costs / Risks

- **One more fsspec protocol to maintain.** Mirror set the precedent; `users://` follows the same shape, but it's another moving piece in `rigging`.
- **Implicit cross-user data dependencies.** Once my pipeline resolves a step to a teammate's `users/{them}/` dir, my downstream runs depend on that dir existing. If they delete it, my next run breaks (recoverable by re-running the upstream step, which then writes to my own dir). Acceptable for a small team where deletion is rare and recovery is cheap; would not scale to dozens of users without a real cache layer.
- **`.executor_status` SUCCESS without artifacts.** The cache hit is gated only on the status token. If a teammate's `.executor_status` is `SUCCESS` but their `checkpoints/` dir was partially deleted (lifecycle, manual `gsutil rm`, etc.), the executor short-circuits and downstream eval/SFT fails at file-open time, not at cache-check time. Recoverable by re-running the upstream step explicitly. A bulletproof fix (probe for a sentinel artifact, e.g. `checkpoints/.complete`) is plausible but out of scope for v1.
- **Cross-region guard surface.** `_infer_gcs_regions` (`executor.py:208`) only inspects `gs://` URLs; a `users://` URL slips through silently. Spec mandates `UserSharedFS` expose a `to_gs_url()` helper that callers (the region check, the cross-region transfer budget) invoke before scheme-matching.
- **Glob cost grows with team size.** A single `users/*/...` glob is ~100ms today (~10s of users); at 100+ users it would still be cheap but no longer free. If the team grows substantially we'd want an index instead of a glob.
- **Identity-resolution edge cases.** Running training outside an Iris context (local dev, ad-hoc scripts) falls back to `getpass.getuser()`; service-account contexts where `getpass` returns the same value across machines (`runner`, `nobody`) silently collide. Mitigation: a `MARIN_USER` env var inserted before the `getpass` step in the resolver chain lets service accounts and CI override explicitly.

## Design

### `UserSharedFS` (`users://`) — the whole feature

A new fsspec filesystem in `rigging`, sibling to `MirrorFileSystem`, that resolves a relative path against three layers in order:

- **My personal (writable):** `{prefix}/users/{me}/{path}`
- **Other users (read-only scan):** `{prefix}/users/*/{path}` — single GCS glob across other user dirs
- **Shared (read-only):** `{prefix}/{path}`

Read semantics: probe my personal first; on miss, glob `users/*/{path}` and return any match; on miss, probe shared; on miss, raise `FileNotFoundError`. With a small team (~10-50 marin users) the glob is a single low-latency GCS list call (~100ms) — no per-user fanout. Whichever layer hits is the concrete `gs://...` URL returned to the caller.

Write semantics: always to my personal. The other-user and shared layers are structurally read-only — writes never land there.

The two-mode behavior (read resolves via layers; write goes to personal) means a teammate's existing SUCCESS is reused transparently, while a teammate's FAILED status forces me to re-run and my writes land in my own dir. No promotion step, no double storage during transition — cross-user reuse is just the read path.

Layers are configured at instantiation: `prefix` resolves through `marin_prefix()`; `user` resolves through the Iris identity chain (`get_job_info().user` → `getpass.getuser()`) with an explicit override accepted in the constructor. The filesystem is registered as `users://` via `fsspec.register_implementation`.

`UserSharedFS` ships as a sibling to `MirrorFileSystem`, not a refactor. Mirror copies on miss across bucket boundaries with a transfer budget and a distributed lock; `UserSharedFS` does pure routing across prefixes inside one bucket. The two share a one-paragraph algorithm and almost no code — pulling out a `LayeredReadFS` base now would be premature abstraction by the standard of `AGENTS.md`. If a third use case shows up, refactor then.

### How training opts in

After the executor change above, every `ExecutorStep` carries an optional `output_path_prefix` that overrides the executor's global prefix for that step's `{prefix}/{name}_{hash}` path. Setting it to `"users://"` is the entire plumbing — hash-keyed caching is preserved, the resulting `output_path` is `users://{name}_{hash}`, and every artifact written by the step (`.executor_status`, checkpoints, logs) lands behind the layered fs.

V1 wires this in two places in `experiments/defaults.py`:

- **`default_train`** ([line 504](https://github.com/marin-community/marin/blob/ea00d29f1/experiments/defaults.py#L504)) and **`default_dpo`** ([line 821](https://github.com/marin-community/marin/blob/ea00d29f1/experiments/defaults.py#L821)) — the two ExecutorStep-producing leaf factories. Pass `output_path_prefix="users://"` to the `ExecutorStep` constructor. `default_sft` (line 749) and `simulated_epoching_train` (line 323) inherit transitively because they route through `default_train`.
- **`resolve_lm_train_config`** ([line 611](https://github.com/marin-community/marin/blob/ea00d29f1/experiments/defaults.py#L611)) — the inline-submission path used by `train()` / `prepare_lm_train()` bypasses ExecutorStep and calls `compute_output_path()` directly. Pass `output_path_prefix="users://"` (which is forwarded into the throwaway ExecutorStep that `compute_output_path` constructs).

Because the routing decision lives in the factories rather than on the executor or on `StepSpec` itself, opting out for a one-off (e.g. a "born shared" canonical training run intended for everyone) is just passing `output_path_prefix=None` through the factory.

A teammate's eval reading `users://{step}_{hash}/checkpoints/` resolves through the layered probe — my personal first, then any other user with a successful run of the same hash, then shared. The executor cache check at `executor.py:1448` becomes cross-user automatically because the `.executor_status` URL is layered: if any user has SUCCESS for this hash, we skip the recompute.

### Identity-failure mode

If the resolution chain produces no username — no explicit override, no Iris context, and `getpass.getuser()` raises (stripped containers with no `USER`/`LOGNAME` env vars and a UID not in `/etc/passwd`) — `UserSharedFS` raises on first access. We hard-fail rather than degrade because silently routing personal data into the shared root defeats attribution; a clear error at job start is cheaper than discovering the misroute weeks later. The hard-fail case is genuinely rare (it requires both Iris-less execution *and* a missing OS identity); the more common "ran outside Iris" path resolves via `getpass.getuser()` and works.

### Reporting (follow-up, not v1 implementation)

Per-user storage attribution becomes a GCS-inventory scan over `users/{user}/` prefixes, modeled on `scripts/ops/egress_report.py`. V1 ships the directory convention; the scanner job is a follow-up PR.

### Out of scope for v1

- Any promotion mechanism (the cross-user read scan replaces it — see below).
- Lifecycle rules on `users/*` (no auto-deletion).
- Hard write-time quota enforcement.
- Routing non-training step outputs (datasets, evals, ferries, scratch) through `users://`.
- A `scope` field on `StepSpec` (not needed under this design — opt-in is at URL-construction time).

## Testing

- **Unit on `UserSharedFS` resolution.** Stub the underlying fs with `memory://`; assert reads probe personal first, then glob across other users, then shared; that writes always go to personal regardless of where reads resolved; that identity override works; that identity-failure raises; and that `_resolve_path` returns the URL of whichever layer hit. Include the FAILED-at-teammate case: resolution returns teammate's path, executor reads FAILED, re-run writes to my personal.
- **Executor unchanged.** Snapshot test on `compute_output_path()` for existing experiments confirms no path drift for non-training steps.
- **Integration via a training step.** Run a small training executor step end-to-end with `override_output_path = "users://..."` against a test bucket; assert `.executor_status` and checkpoints land under `users/{user}/`, second invocation hits the personal layer, and a manually-promoted copy in the shared layer is found via fallback when the personal layer is purged.

## Open Questions

- **Confirm existing pretraining-checkpoint layout.** The "no cutover needed" property relies on currently-running pretraining checkpoints living at `gs://marin-{region}/checkpoints/...` (i.e., reachable via layer-3 shared-root fallback). If any active run targets a custom subprefix outside the regional bucket root, those owners need a one-time `output_path_prefix=None` opt-out for the resume cycle. Reviewers closer to the active training queue should sanity-check this assumption.
- **Future opt-in candidates beyond training.** V1 routes only `default_train` / `default_dpo` through `users://`. Obvious follow-ups: eval-result bundles (small but proliferate), sampler outputs (medium-sized, sometimes shared), tokenization caches (large, currently shared and content-addressable). Worth deciding now which of these should join the next rollout vs. wait for evidence of attribution gaps.
