---
date: 2026-08-27
system: iris
severity: degraded
resolution: investigating
pr: https://github.com/marin-community/marin/pull/8656
issue: https://github.com/marin-community/marin/issues/8734
---

## TL;DR

- A production v4-2048 job remained pending with `Coscheduling: need 256 workers in same 'tpu-name' group, largest group has 254`.
- The slice had 255 registered Iris workers. One expected worker was absent, and another had empty TPU metadata.
- TPU host `t1v-n-25bf471b-w-74` registered as Iris `worker-0`, colliding with the real `worker-0`. The two hosts alternated registrations from different IP addresses.
- `lib/iris/src/iris/cluster/worker/worker.py:229-231` maps a missing TPU worker index to zero, which turns incomplete metadata into a valid but duplicate worker identity.
- No training tasks started. Recovery requires replacing the degraded slice; no live mutation had been performed at the time of this entry.

## Original problem report

The user reported: "one host of the 256 is unhealthy Iris says" for `/held/iris-run-job-20260827-194841`.

## Investigation path

1. `iris job list --json --prefix /held/iris-run-job-20260827-194841` showed the coordinator running and all 256 training tasks pending. The scheduler reported that the largest `tpu-name` group had 254 workers and that the v4-2048 scale group was at its one-slice limit.

2. The autoscaler status showed one ready v4-2048 slice with `capacity_status=degraded`: 256 VM records, 255 healthy Iris workers, and missing worker `marin-tpu-v4-reserved-2048-us-central2-b-20260827-1951-20cad6fa-worker-74`.

3. Read-only controller SQL found 255 registered workers on the slice. Of those, 254 reported TPU name `t1v-n-25bf471b`; one reported an empty TPU name and worker index. The empty-metadata record was Iris `worker-0`, running on GCE instance `t1v-n-25bf471b-w-74` at `10.130.1.252`.

4. Controller logs showed `worker-0` repeatedly registering from `10.130.1.252` and `10.130.1.210` in alternating ten-minute intervals. Both physical hosts used the same Iris identity, so each registration replaced the other controller record.

5. `iris process status` resolved `worker-0` to hostname `t1v-n-25bf471b-w-74`. The expected `worker-74` endpoint did not exist. Worker logs showed repeated `No contact from controller, resetting` cycles, consistent with the identity collision.

6. Source inspection found the identity fallback at `lib/iris/src/iris/cluster/worker/worker.py:229-231`: when a slice ID exists but `hardware.tpu_worker_id` is empty, Iris silently chooses worker index zero.

7. A final active-task join found six unrelated CPU tasks bin-packed onto healthy workers in the degraded TPU slice. Replacing the slice without draining those tasks would interrupt jobs owned by `dlwh`, `marin`, `michaelryan`, and `runner`.

## Root cause

One TPU host lacked the TPU worker-index metadata used to derive its Iris identity. Iris treated the missing index as zero and registered host `w-74` as `worker-0`. That collided with the real `worker-0`, leaving `worker-74` absent and only 254 registered workers sharing the correct TPU name. The scheduler correctly refused to start a 256-way coscheduled job on that topology.

The code-level amplifier was `lib/iris/src/iris/cluster/worker/worker.py:229-231`. A missing index on a multi-host slice should not produce the valid identity `worker-0`; it should fail registration or use an explicit, validated worker identity.

## Fix

No recovery or code change had been applied. Immediate recovery requires draining the six CPU co-tenants, then deleting and recreating the degraded v4-2048 slice. Stopping and resubmitting only the pending job would retain the same degraded slice.

A follow-up code fix should reject missing TPU worker-index metadata when `config.slice_id` identifies a multi-host slice. The worker must not default such failures to index zero.

## How OPS.md could have shortened this

- Add a coscheduling diagnostic under `Scheduler & Autoscaler`: compare the expected slice worker IDs with registered `workers`, then group registered workers by `md_tpu_name`. A full VM count with a smaller TPU-name group indicates worker-registration or metadata drift rather than TPU capacity shortage.
- Add a worker-identity collision signal under `Troubleshooting`: repeated `worker_registered` events for one worker ID from alternating addresses indicate two daemons sharing an identity. Check `md_tpu_worker_id`, `md_tpu_name`, and GCE hostname before treating it as a generic unhealthy host.

## Artifacts

- Iris job: https://iris.oa.dev/#/job/%2Fheld%2Firis-run-job-20260827-194841
- Experiment issue: https://github.com/marin-community/marin/issues/8734
- Affected slice: `marin-tpu-v4-reserved-2048-us-central2-b-20260827-1951-20cad6fa`
- Affected TPU host: `t1v-n-25bf471b-w-74`
