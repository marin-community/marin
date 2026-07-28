---
date: 2026-07-28
system: iris
severity: experiment-launch-blocked
resolution: mitigated
pr: https://github.com/marin-community/marin/pull/7667
issue: https://github.com/marin-community/marin/issues/7705
---

# Main Iris stopped receiving the cw-us-east-08a federation heartbeat

## Summary

The main `marin` Iris controller could not deliver NEST-BURN-002 to
`cw-us-east-08a` on 2026-07-28. Its `ListPeers` response marked the peer
unreachable and retained an availability observation from 14:23:59 UTC.
Direct RPCs to the 08a controller remained healthy, and direct submissions
allocated three 16-node GB200 gangs normally.

The experiment was unblocked by terminating two roots before handoff and
submitting them directly to 08a. No controller was restarted. The main-to-peer
control path remains stale at the time of this record.

## Impact

Two root jobs remained in `QUEUED_HANDOFF` despite the peer having sufficient
capacity. The failed handoff delayed the experiment by about five minutes but
allocated no GPU. Any other job relying on main-to-08a federation may also
remain queued against stale capacity.

## Evidence

- Main-cluster `iris rpc controller list-peers` reported:
  - `peer_id=cw-us-east-08a`;
  - `reachable=false`;
  - `last_contact_ms=1785248639845` (14:23:59.845 UTC);
  - GB200 availability observation
    `observation_epoch_ms=1785248639819`;
  - stale capacity `784/808`.
- At 15:55 UTC, the same response was still unchanged while other peers had
  current 15:54 UTC observations.
- The main controller and all 1,043 workers reported healthy.
- Direct `cw-us-east-08a` controller RPCs, job submission, log retrieval,
  thread profiling, and task exec all succeeded.
- Directly submitted 16-node jobs acquired their gangs and ran. This rules out
  an 08a controller, scheduler, or capacity outage as the immediate cause.
- The affected roots were
  `/power/nest-burn-002-e256-100b-b128-r3-coord` and
  `/power/nest-burn-002-fixed25-100b-b128-r3-coord`. Both were stopped while
  still queued for handoff, so neither can later start as a duplicate.

## Mitigation

Submit the complete root job tree directly to `cw-us-east-08a` until the
parent's peer heartbeat is current again. Do not pin a federated child job:
Iris federates root trees, and children remain local to the coordinator that
creates them.

Before direct fallback, terminate the still-queued federated root and verify
its state is terminal. This prevents a delayed handoff from starting a
duplicate gang.

## Follow-up

- Alert when a configured federation peer's last contact exceeds two normal
  heartbeat intervals.
- Include the age of the availability observation in `job list` reasons; a
  stale free count currently reads like a live capacity constraint.
- Determine why the main controller stopped contacting only the 08a peer
  while direct access remained healthy. The current evidence localizes the
  fault to the parent-to-peer control path but does not identify the failing
  network, authentication, or refresh component.
