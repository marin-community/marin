---
date: 2026-07-22
system: coreweave
severity: outage
resolution: fixed
pr: none
issue: none
---

## TL;DR

- `cw-rno2a` lost the `kueue-webhook-service` endpoint while `kueue-controller-manager` entered `CrashLoopBackOff`.
- The observed restarts were not OOM kills. They exited with code 1 after the Kubernetes API client rate limiter delayed event writes past Kueue's 10-second leader-election renewal deadline.
- The failure appeared during a resync with approximately 4,843 Pods and 3,080 Kueue Workloads. Kueue was using its 20-QPS, 30-request burst defaults.
- `provisioning.coreweave.kueue.client_connection` now sets 100 QPS and a 200-request burst for `cw-rno2a`. A targeted Pulumi update rolled the controller; the replacement Pod became ready with zero restarts and restored the webhook endpoint.
- The full preview also found an existing Cloudflare federation CNAME orphaned from the Pulumi program. The CNAME is declared and protected again, and an untargeted preview reports 29 unchanged resources.

## Original problem report

Grafana fired `WebhookEndpointsEmpty cw-rno2a` for `kueue-system/kueue-webhook-service`, followed by `ControlPlaneCrashLooping cw-rno2a`. A previous Kueue failure had been caused by an OOM, so the operator asked whether memory pressure had returned and requested a Pulumi correction plus reload.

## Investigation path

1. The live Deployment requested and limited Kueue to 2 GiB. Its only Pod, `kueue-controller-manager-78558f65f7-q92qg`, was unready with 21 restarts, and the webhook Endpoint contained only a not-ready address.

2. Kubernetes recorded the latest termination as `Reason: Error`, `Exit Code: 1`, not `OOMKilled`. The Pod ran on a healthy node with approximately 1.5 TiB of allocatable memory. The cluster did not expose the Metrics API, so no historical working-set measurement was available.

3. Previous-container logs ended with `client rate limiter Wait returned an error: context deadline exceeded`, `Failed to renew lease`, and `Could not run manager` with `error: leader election lost`. A second restart at 22:54 UTC ended the same way.

4. The manager ConfigMap had no `clientConnection` block, leaving Kueue v0.18.0 at its 20-QPS and 30-request burst defaults. The cluster held approximately 4,843 Pods and 3,080 Kueue Workloads. Restart resyncs produced event POST delays above 11 seconds, longer than the 10-second leader renewal deadline.

5. `lib/iris/src/iris/cluster/platforms/k8s/kueue_manifests.py:138` gained an optional `clientConnection` renderer. `lib/iris/config/cw-rno2a.yaml:41` set 100 QPS and a 200-request burst while retaining the existing 2 GiB memory limit.

6. An untargeted Pulumi preview proposed deleting `iris-cw-rno2a.oa.dev`, an unrelated Cloudflare CNAME still present in stack state. The Kueue recovery used a target restricted to the Helm release; no DNS or NodePool resource changed.

7. The targeted update completed in 43 seconds. Deployment revision 6 created `kueue-controller-manager-77cbc9bcd7-d5hcb`, and `kueue-webhook-service` published its ready endpoint at `10.0.1.139`.

8. `infra/pulumi/src/iac/coreweave/dns.py` restored the exact Cloudflare component, provider, and record URNs already in state. The live CNAME still resolved to `iris-cw-rno2a.208261-marin-rn02a.coreweave.app`, and the final untargeted preview reported all 29 resources unchanged.

## User course corrections

- After the targeted Kueue rollout succeeded, the operator asked for the orphaned Cloudflare CNAME to be fixed before finishing. This converted a one-off targeted-update workaround into a clean, non-destructive full-stack preview.

## Root cause

Kueue's controller, event recorder, and leader-election operations shared the configured Kubernetes client rate limiter. At 20 QPS with a burst of 30, the `cw-rno2a` restart resync queued enough event writes to delay lease renewal beyond 10 seconds. Kueue exited on `leader election lost`, Kubernetes restarted it, and the fail-closed admission webhook temporarily had no ready endpoint.

The prior 2 GiB memory correction remained active. The sampled failures had explicit exit-code-1 leader-election logs and no `OOMKilled` termination reason. Kubernetes retains only the most recent terminated-container status, so terminations older than the two captured log samples were not independently classified.

## Fix

`lib/iris/src/iris/cluster/platforms/k8s/kueue_manifests.py` now renders explicit client rate limits when both values are configured:

```yaml
clientConnection:
  qps: 100
  burst: 200
```

`infra/pulumi/src/iac/config.py` exposes the pair as a typed provisioning block, and `lib/iris/config/cw-rno2a.yaml` supplies the large-cluster override. Pulumi applied only the Kueue Helm release during recovery.

`infra/pulumi/src/iac/coreweave/dns.py` also declares the existing DNS-only Cloudflare record with deletion protection. The CoreWeave config carries its zone, hostname, and allocated LoadBalancer target. Operators load `cloudflare-oa-dns-token` into `CLOUDFLARE_API_TOKEN` for previews and updates.

## How OPS.md could have shortened this

- Add a CoreWeave control-plane subsection under `lib/iris/OPS.md` "Troubleshooting" with these first checks: `kubectl describe pod`, `.status.containerStatuses[*].lastState`, and `kubectl logs --previous`. The termination reason distinguishes an OOM from a controller exit before resource changes are considered.
- Add `client rate limiter` plus `leader election lost` as a generic controller-runtime diagnostic. The runbook should direct the operator to compare API request delay with `leaderElection.renewDeadline`, count watched objects, and inspect the component's `clientConnection.qps` and `burst` configuration.
- Extend the Pulumi warning to reject any unexpected deletion, not only NodePool replacement or deletion. A targeted update is appropriate only after its preview names the exact recovery resource; the program and state drift still need reconciliation afterward.

## Artifacts

- `.agents/ops/2026-07-22-coreweave-kueue-leader-loss.md`
- `lib/iris/config/cw-rno2a.yaml`
- `lib/iris/src/iris/cluster/platforms/k8s/kueue_manifests.py`
- `infra/pulumi/src/iac/coreweave/dns.py`
- Grafana alert: https://grafana.oa.dev/alerting/grafana/k8s-control-plane-crashloop/view?orgId=1
