# Iris placement-collision + vfio-race recurrence on 2026-05-07

Two related iris issues likely both in play. Below: comment text for each, ready to paste.

---

## For #5470 (currently CLOSED — needs reopen)

URL: https://github.com/marin-community/marin/issues/5470
After posting, reopen and add the `agent-generated` label.

---

🤖 Reopening — same bug recurred on 2026-05-07 19:13 UTC, ~20 hours after #5490 merged. Same signature: two of three concurrently-dispatched v5p-64 gangs landed on the **identical 8-host set** and thrashed for 707 preemption cycles each (the documented v5p-64 8-task fingerprint).

Cluster: `marin`. Priority `interactive`, region `us-east5`, TPU `v5p-64`. All three jobs are 1e22 midtraining resumes (`MIDTRAIN_RESUME_OUTPUT_PATH` contract; checkpoint state preserved, no compute lost in the failure itself).

Affected jobs (all FAILED, train_lm preemptions / worker_failed):

- `/ahmedah/delphi-1e22-p33m67-lr0p83-v5p64-resume-20260507-190726` — 707 / 7 of 8
- `/ahmedah/delphi-1e22-p50m50-lr0p5-v5p64-resume-20260507-190908`  — 707 / 7 of 8
- `/ahmedah/delphi-1e22-p50m50-lr0p67-v5p64-resume-20260507-191041` — 707 / 7 of 8

Host overlap (gold-standard confirmation):

```
lr0.83 ∩ lr0.5: 8/8 hosts identical:
  10.202.0.182, 10.202.0.222, 10.202.0.225, 10.202.0.240,
  10.202.0.255, 10.202.1.8, 10.202.1.9, 10.202.1.11
lr0.83 ∩ lr0.67: 0 shared
lr0.5  ∩ lr0.67: 0 shared
```

Timeline (UTC) — note this is the **dispatch-after-failure** race that the description of #5490 explicitly targets:

```
19:11:56.883  lr0.83 train_lm submitted
19:13:06.097  lr0.83 train_lm started        (gang gets host set A)
19:13:17.967  lr0.67 train_lm submitted
19:13:22.767  lr0.5  train_lm submitted
19:13:26.097  lr0.83 task 0 SIGSEGV at +20s  (failure window opens)
19:15:38.518  lr0.5  train_lm started        (re-uses host set A → collision)
19:17:40.975  lr0.67 train_lm started        (gets host set B → clean)
```

There's a **2.3-minute gap** between lr0.83's task 0 SIGSEGV and lr0.5's dispatch — that should have been more than enough for `StopTasks` RPCs to evict lr0.83's still-running cohort siblings, but iris re-allocated the same 8 hosts to lr0.5 anyway, and they immediately hit the JAX coordinator port-8476 collision and thrashed for 707 cycles.

Per-task root errors (consistent with the `_requeue_coscheduled_siblings` race pattern):

- lr0.83 task 0: `Exit code 139: killed by SIGSEGV. stderr: @ 0x564903bbbcc7 _PyEval_EvalFrameDefault`
- lr0.5  parent: `connectrpc.errors.ConnectError: Request timed out`
- lr0.67 task 1: `RPC: /tensorflow.Coordination...` (JAX coordinator)
- All other tasks on each job: `Coscheduled sibling /…/train_lm/0 failed` cascade

Cross-ref: #5258 may also be in play here (vfio busy on host reuse → JAX `PollForError` on the second tenant). The lr0.67 task 1 error in particular looks more like #5258's symptom than #5470's. So the failure today is plausibly **both bugs together**: #5470 puts two gangs on the same hosts, #5258 makes those hosts hang at libtpu init.

Asks:

1. Verify the running controller binary actually contains #5490 (merge commit `0e8ea1720` on `main`). It is possible the controller was restarted but on an older build.
2. If yes — the fix has a gap. The lr0.83 → lr0.5 timeline above is probably enough to localize it: a 2.3-minute window after task 0's SIGSEGV was insufficient to evict the cohort from set A before the allocator re-issued the same 8 hosts to lr0.5.
3. Until fixed, I'm serializing submissions one-at-a-time and waiting for `train_lm` to be `running` for ≥30s before the next submit.

Diagnostic repro:

```bash
for j in /ahmedah/delphi-1e22-p33m67-lr0p83-v5p64-resume-20260507-190726 \
         /ahmedah/delphi-1e22-p50m50-lr0p5-v5p64-resume-20260507-190908 \
         /ahmedah/delphi-1e22-p50m50-lr0p67-v5p64-resume-20260507-191041; do
  uv run iris --controller-url=http://localhost:10000 --cluster=marin job summary "$j/train_lm" | head -4
done

for cell in p33m67-lr0p83 p50m50-lr0p5 p50m50-lr0p67; do
  uv run iris --controller-url=http://localhost:10000 --cluster=marin \
    job logs "/ahmedah/delphi-1e22-${cell}-v5p64-resume-20260507-*/train_lm" \
    --max-lines 3000 --level info \
    | grep -oE "advertise_host=10\.[0-9.]+" | sort -u
done
```

Cross-ref: in-repo postmortem `.agents/ops/iris_placement_bug.md` (5 incidents now, including this one), original fix PR #5490.

---

## For #5258 (currently OPEN — just add comment)

URL: https://github.com/marin-community/marin/issues/5258

---

🤖 New occurrence on 2026-05-07 19:13 UTC on `marin` cluster, v5p-64. Likely co-firing with #5470 (placement collision) — the placement bug put two gangs on the same host set, and at least one of the surviving tasks shows the exact `PollForError` symptom this issue describes.

Affected jobs:

- `/ahmedah/delphi-1e22-p33m67-lr0p83-v5p64-resume-20260507-190726` — task 0 `SIGSEGV` at +20s after libtpu init began (suspect `/dev/vfio` busy on at least one host); tasks 1–7 `worker_failed` from coscheduled-sibling cascade.
- `/ahmedah/delphi-1e22-p50m50-lr0p5-v5p64-resume-20260507-190908` — same 8-host set as lr0.83, dispatched 2.3 min after lr0.83's task-0 SIGSEGV. Could not init.
- `/ahmedah/delphi-1e22-p50m50-lr0p67-v5p64-resume-20260507-191041` — task 1 explicit error: `RPC: /tensorflow.Coordination...` → matches the `PollForError` symptom in this issue's TL;DR.

All three were terminated by iris with `failures=1, preemptions=707, worker_failed=7/8`. None of the slices hosting the dead containers were terminated — iris kept the bad VMs in the pool and recycled them on subsequent task attempts, exactly the behavior this issue describes.

Trigger sequence (matches "preempt-then-reuse" reproducer steps in the original report):

```
19:13:06.097  lr0.83 train_lm started on host set A
19:13:26.097  lr0.83 task 0 SIGSEGV (+20s)  — likely first vfio-busy host bites
19:15:38.518  lr0.5  train_lm dispatched onto SAME host set A (#5470 collision)
              ↑ tasks hit libtpu init issues / PollForError on the same suspect hosts
```

`tpu_health.detect_tpu_init_failure` did not match — slice was not terminated, hosts remained in the pool. Possible signal to add to the matcher: `RPC: /tensorflow.Coordination...` / `PollForError` paired with `_PyEval_EvalFrameDefault` SIGSEGV at libtpu init time.

No compute was lost (these were resume jobs that died at boot before loading checkpoints), but it blocked the user's sweep for ~10 minutes and required manual host triage.

Cross-ref: today's evidence and a deeper timeline are also on #5470 (which I'm reopening). Postmortem `.agents/ops/iris_placement_bug.md` (Marin repo, midtrain_data branch).
