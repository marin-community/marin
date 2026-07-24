---
date: 2026-07-22
system: coreweave
severity: outage
resolution: fixed
pr: https://github.com/marin-community/marin/pull/7533
issue: none
---

## TL;DR

- `cw-rno2a` lost the `kueue-webhook-service` endpoint while `kueue-controller-manager` entered `CrashLoopBackOff`.
- The observed restarts were not OOM kills. They exited with code 1 after the Kubernetes API client rate limiter delayed event writes past Kueue's 10-second leader-election renewal deadline.
- The failure appeared during a resync with approximately 4,843 Pods and 3,080 Kueue Workloads. Kueue was using its 20-QPS, 30-request burst defaults.
- Iris now renders 100 QPS and a 200-request burst by default for every Kueue installation. A targeted Pulumi update applied those values to `cw-rno2a`; the replacement Pod became ready with zero restarts and restored the webhook endpoint.
- Full previews also found the existing Cloudflare federation CNAME in every CoreWeave stack orphaned from the Pulumi program. All four records are declared and protected again.

## Original problem report

Grafana fired `WebhookEndpointsEmpty cw-rno2a` for `kueue-system/kueue-webhook-service`, followed by `ControlPlaneCrashLooping cw-rno2a`. A previous Kueue failure had been caused by an OOM, so the operator asked whether memory pressure had returned and requested a Pulumi correction plus reload.

## Investigation path

1. The live Deployment requested and limited Kueue to 2 GiB. Its only Pod, `kueue-controller-manager-78558f65f7-q92qg`, was unready with 21 restarts, and the webhook Endpoint contained only a not-ready address.

2. Kubernetes recorded the latest termination as `Reason: Error`, `Exit Code: 1`, not `OOMKilled`. The Pod ran on a healthy node with approximately 1.5 TiB of allocatable memory. The cluster did not expose the Metrics API, so no historical working-set measurement was available.

3. Previous-container logs ended with `client rate limiter Wait returned an error: context deadline exceeded`, `Failed to renew lease`, and `Could not run manager` with `error: leader election lost`. A second restart at 22:54 UTC ended the same way.

4. The manager ConfigMap had no `clientConnection` block, leaving Kueue v0.18.0 at its 20-QPS and 30-request burst defaults. The cluster held approximately 4,843 Pods and 3,080 Kueue Workloads. Restart resyncs produced event POST delays above 11 seconds, longer than the 10-second leader renewal deadline.

5. `lib/iris/src/iris/cluster/platforms/k8s/kueue_manifests.py` gained a `clientConnection` renderer. The first recovery applied 100 QPS and a 200-request burst through `cw-rno2a`'s provisioning config while retaining the existing 2 GiB memory limit.

6. An untargeted Pulumi preview proposed deleting `iris-cw-rno2a.oa.dev`, an unrelated Cloudflare CNAME still present in stack state. The Kueue recovery used a target restricted to the Helm release; no DNS or NodePool resource changed. Later previews found the same drift in the other three CoreWeave stacks.

7. The targeted update completed in 43 seconds. Deployment revision 6 created `kueue-controller-manager-77cbc9bcd7-d5hcb`, and `kueue-webhook-service` published its ready endpoint at `10.0.1.139`.

8. `infra/pulumi/src/iac/coreweave/dns.py` restored the exact Cloudflare component, provider, and record URNs already in state. Each cluster config now declares the hostname and live target already recorded in its Pulumi state.

## User course corrections

- After the targeted Kueue rollout succeeded, the operator asked for the orphaned Cloudflare CNAME to be fixed before finishing. Full-stack previews showed that the same correction was required for all four existing CoreWeave records.
- Because Iris control-plane load is similar across clusters, the operator asked to make 100 QPS and a 200-request burst the shared default rather than a `cw-rno2a` override.

## Root cause

Kueue's controller, event recorder, and leader-election operations shared the configured Kubernetes client rate limiter. At 20 QPS with a burst of 30, the `cw-rno2a` restart resync queued enough event writes to delay lease renewal beyond 10 seconds. Kueue exited on `leader election lost`, Kubernetes restarted it, and the fail-closed admission webhook temporarily had no ready endpoint.

The prior 2 GiB memory correction remained active. The sampled failures had explicit exit-code-1 leader-election logs and no `OOMKilled` termination reason. Kubernetes retains only the most recent terminated-container status, so terminations older than the two captured log samples were not independently classified.

## Fix

`lib/iris/src/iris/cluster/platforms/k8s/kueue_manifests.py` now renders these client rate limits by default for both CKS and upstream Kueue charts:

```yaml
clientConnection:
  qps: 100
  burst: 200
```

`infra/pulumi/src/iac/config.py` exposes the pair as a typed provisioning block with the same shared defaults, while retaining per-cluster overrides. Pulumi applied only the `cw-rno2a` Kueue Helm release during recovery; other clusters will receive the defaults through their normal Pulumi updates.

`infra/pulumi/src/iac/coreweave/dns.py` also declares each existing DNS-only Cloudflare record with deletion protection. The typed schema owns the shared `oa.dev` zone ID, while every CoreWeave config carries its hostname and allocated LoadBalancer target. Operators load `cloudflare-oa-dns-token` into `CLOUDFLARE_API_TOKEN` for previews and updates.

## How OPS.md could have shortened this

- Add a CoreWeave control-plane subsection under `lib/iris/OPS.md` "Troubleshooting" with these first checks: `kubectl describe pod`, `.status.containerStatuses[*].lastState`, and `kubectl logs --previous`. The termination reason distinguishes an OOM from a controller exit before resource changes are considered.
- Add `client rate limiter` plus `leader election lost` as a generic controller-runtime diagnostic. The runbook should direct the operator to compare API request delay with `leaderElection.renewDeadline`, count watched objects, and inspect the component's `clientConnection.qps` and `burst` configuration.
- Extend the Pulumi warning to reject any unexpected deletion, not only NodePool replacement or deletion. A targeted update is appropriate only after its preview names the exact recovery resource; the program and state drift still need reconciliation afterward.

## Artifacts

- `.agents/ops/2026-07-22-coreweave-kueue-leader-loss.md`
- `lib/iris/src/iris/cluster/platforms/k8s/kueue_manifests.py`
- `infra/pulumi/src/iac/coreweave/dns.py`
- Grafana alert: https://grafana.oa.dev/alerting/grafana/k8s-control-plane-crashloop/view?orgId=1
