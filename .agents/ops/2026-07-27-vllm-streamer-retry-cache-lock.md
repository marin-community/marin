---
date: 2026-07-27
system: vllm
severity: degraded
resolution: fixed
pr: https://github.com/marin-community/marin/pull/7674
issue: https://github.com/marin-community/marin/issues/6503
---

## TL;DR

- A Grug H100x8 evaluation failed after RunAI model streamer workers reported `Could not receive runai_response from libstreamer due to: b'File access error'`.
- The existing transient-startup retry classified the read failure correctly.
- Teardown kept the managed compilation-cache lock because the killed vLLM process group still existed, so the retry failed before launch with `vLLM compilation cache ... is already active on this host`.
- Teardown now ignores zombie/dead group members after killing vLLM, allowing the managed-cache lock to close before the retry.
- The Grug entry now uses distributed streaming, RunAI 0.16.1 bounds in-flight object-store reads,
  and the CUDA launcher gives the S3 low-speed check 10 seconds.
- A one-instance acceptance run streamed 15.6 GiB per rank in 14.63–14.75
  seconds, reached `vLLM environment ready`, and returned HTTP 200 to its
  first Harbor requests. The zombie regression and existing retry suite passed
  locally.

## Original problem report

The `grug-agentic-s3-step1903` / `grug-opencode-id` command from the evaluation
runbook failed during model startup. Three data-parallel workers reported the
RunAI `File access error`; the parent evaluation then reported that inference
finished before registering an endpoint.

## Investigation path

1. Iris job `/loom/eval-20260727-174623-grug-agentic-s3-step1903-ca26` failed after its inference child ran for five minutes.
2. The child log showed DP1, DP2, and DP6 failing in `runai_model_streamer.file_streamer.request_ready_chunks`. Other ranks continued loading until vLLM shut down.
3. `lib/marin/src/marin/inference/vllm_server.py:717` already classified this log marker as `TransientStartupError`, and the retry loop scheduled attempt two.
4. Teardown logged `Keeping vLLM compilation cache because process group 459 still exists`. The next attempt failed in `lib/marin/src/marin/inference/vllm_cache.py:191` while acquiring the same deterministic cache lock.
5. A replacement H100x8 serve showed the vLLM API, engine, and worker processes sharing one process group. This made zombie processes after `SIGKILL` the leading explanation for the group continuing to exist without being able to use cache files.
6. A local process-group regression reproduced the retained cache with one unreaped zombie. It passed after the liveness check inspected Linux procfs states instead of relying only on `killpg(..., 0)`.

## User course corrections

- The session was monitoring all documented evaluation commands when the user redirected it to make vLLM streaming more robust to S3 errors. The investigation moved from command validation to the failed retry path.

## Root cause

The RunAI S3 read fault was already retryable, but retry cleanup and managed
compilation-cache ownership disagreed about when a process group was gone.
`VllmServerHandle.stop()` used `killpg(..., 0)`, which reports zombie members
as existing. It therefore retained the cache lock after killing the group, and
the next attempt could not prepare the deterministic cache workspace.

## Fix

`lib/marin/src/marin/inference/vllm_server.py` now inspects `/proc/*/stat` on
Linux and treats a process group containing only `Z` or `X` entries as stopped.
It falls back to the existing conservative `killpg(..., 0)` check when procfs
cannot be inspected or on another platform. Active workers still retain the
cache.

`tests/inference/test_vllm_server.py` creates a real process group with an
unreaped zombie and verifies that `VllmServerHandle.stop()` releases the
managed cache. The full vLLM server and cache test selection passed: 18 tests.

## Streamer configuration follow-up

The failed serve made each of eight data-parallel ranks stream the full 124.9
GiB export. Rank load times ranged from 57 to 229 seconds. The equivalent
Snowball factory and backend parity test already pass
`--model-loader-extra-config '{"distributed":true}'`; the Grug YAML omitted it.

RunAI 0.16.0 submits all chunks for a read to the object-store client at once
and fails the read when any chunk exhausts the AWS client's retries. RunAI
0.16.1 adds a completion-driven in-flight window before the S3 client. It still
has no second retry loop around a failed chunk.

The 0.16.0 wheel bundles AWS SDK for C++ 1.11.584. Its `aws-c-s3` client uses a
standard retry strategy with five retries when RunAI leaves the strategy
unset. `AWS_MAX_ATTEMPTS` is read by the C++ SDK but is not transferred to the
CRT strategy's retry count. `AWS_RETRY_MODE=standard` selects a CRT standard
strategy whose default is three retries, so these variables are not used for
hardening.

The RunAI S3 client's `RUNAI_STREAMER_S3_REQUEST_TIMEOUT_MS` controls the
low-throughput failure interval. RunAI 0.16.x defaults it to 1,000 ms, while
AWS SDK 1.11.584 clamps the effective CRT interval to at least three seconds.
The CUDA launcher now defaults to 10,000 ms and preserves a value already
present in the inference environment. It also enables warning/error output on
stderr so a later `File access error` includes the underlying S3 exception in
the Iris job log.

### Background research brief

- Effort: low
- Stop rule: exact pinned sources established the request fanout, timeout, and retry behavior.
- Date: 2026-07-27

Evidence:

- The failed Iris job logged eight full 124.9 GiB streams and three RunAI read
  failures.
- RunAI 0.16.0 `client.cc` says one failed subrange fails the entire read and
  leaves a failed-read retry as future work.
- RunAI PR #156 bounds object-store submissions and shipped in 0.16.1.
- AWS SDK for C++ 1.11.584 maps the generic retry mode to a CRT strategy but
  does not map its maximum-attempt value. The bundled `aws-c-s3` commit defaults
  to five retries.

Negative results:

- `AWS_MAX_ATTEMPTS` does not increase this S3 CRT client's retries.
- `AWS_RETRY_MODE=standard` can reduce the retry budget.
- RunAI memory limits address host-memory pressure, not this S3 read fault.

The minimum live experiment launched
`grug-agentic-s3-step1903` / `grug-opencode-id --limit 1`. All eight ranks used
distributed streaming and read 15.6 GiB in 14.63–14.75 seconds, instead of
each reading 124.9 GiB in 57–229 seconds. vLLM registered its endpoint and
Harbor received HTTP 200 responses from `/v1/chat/completions`. If S3 faults
persist, the next bounded experiment is `RUNAI_STREAMER_CONCURRENCY=4` against
the same command.

## How OPS.md could have shortened this

In `lib/iris/OPS.md` under Process Inspection & Profiling, add a procfs-based
process-state command for minimal inference images that do not contain `ps`.
It should print PID, parent PID, process group, session, and state. This would
distinguish live descendants from zombie-only groups during any task teardown
investigation.

## Artifacts

- Iris root job: `/loom/eval-20260727-174623-grug-agentic-s3-step1903-ca26`
- Failed inference child: `/loom/eval-20260727-174623-grug-agentic-s3-step1903-ca26/inference-e7ae7ec6fdb54d79a8309476560bda4a`
- Successful replacement inference child: `/loom/eval-20260727-175503-grug-agentic-s3-step1903-3be6/inference-6e6c82a0a2b94a2fabfbeb7c1d786293`
- Streamer acceptance root: `/loom/eval-20260727-184149-grug-agentic-s3-step1903-5fdf`
- Streamer acceptance inference child: `/loom/eval-20260727-184149-grug-agentic-s3-step1903-5fdf/inference-7334c6cc15e54f018bd6d5bfc4030320`
- Monitoring state: `scratch/20260727-1743_monitoring_state.json`
