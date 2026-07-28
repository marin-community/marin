---
date: 2026-07-28
system: coreweave
severity: degraded
resolution: investigating
pr: https://github.com/marin-community/marin/pull/7707
issue: none
---

# CoreWeave US-EAST-08A public VIP blackhole

## TL;DR

- Marin reported federation peer `cw-us-east-08a` as unreachable after its last successful contact around 14:24 UTC.
- The Iris controller, controller Service, Traefik pod, TLS certificate, ingress route, and source-IP allowlist remained healthy.
- CoreWeave public VIP `166.19.14.7` timed out from the Marin controller and other external networks but answered from inside US-EAST-08A.
- Cilium advertised `166.19.14.7/32` to the CoreWeave DPU over an established BGP session. A new VIP and a second node/DPU advertisement had the same external timeout.
- CoreWeave must restore upstream route propagation for the US-EAST-08A public LoadBalancer range. No Marin repository change can repair the missing external route.

## Original problem report

The Marin dashboard showed peer `cw-us-east-08a` as `unreachable`, with its
last contact two hours earlier. The peer advertised
`region=US-EAST-08A`, `device-type=gpu`, and `device-variant=gb200`.
Operators could still reach the cluster directly through the Kubernetes API
and an Iris controller port-forward.

Marin controller logs contained:

```text
Federation heartbeat to peer cw-us-east-08a failed: Request timed out
```

The first failure in the continuous incident window occurred at
2026-07-28 14:24:42 UTC:

```text
Federation heartbeat to peer cw-us-east-08a failed: Request failed: error sending request for url (https://iris-cw-us-east-08a.oa.dev/iris.cluster.ControllerService/ListBackends): client error (SendRequest): connection error: host unreachable
```

## Investigation path

1. `iris --cluster=marin cluster status` reported the Marin controller healthy
   at version `366b5d4009`, with 1090 of 1090 workers healthy.
   `iris --cluster=cw-us-east-08a cluster status` reached the peer through a
   Kubernetes port-forward and reported its controller healthy at version
   `a88653b976`.

2. Marin controller logs showed intermittent HTTP `503 Service Unavailable`
   responses before July 28. At 14:24 UTC the failure changed to `host
   unreachable`, followed by one timeout per heartbeat interval.

3. DNS resolved `iris-cw-us-east-08a.oa.dev` through the expected CoreWeave
   hostname to `166.19.14.7`. TCP connections to port 443 timed out from the
   Marin controller and from an unrelated external host. Equivalent public
   VIPs in `cw-us-east-02a` and `cw-rno2a` completed TCP connections and
   returned the expected allowlist `403`.

4. The Marin controller's observed egress IP was `34.27.183.11`. The live
   `iris-federation-ipallowlist` Middleware admitted
   `34.27.183.11/32` and `35.254.13.19/32`, matching
   `infra/pulumi/src/iac/config.py`. The failure occurred before Traefik could
   return an HTTP response, so the allowlist was not the blocking layer.

5. The `iris-controller` pod, `iris-controller-svc`, Traefik pod, Ingress, and
   certificate were ready. Traefik had zero restarts over 12 days.
   `iris-controller-svc:10000/health` returned `200`. Requests through the
   Traefik cluster IP and through `166.19.14.7` from inside the cluster returned
   `403`, proving that the controller backend, route, middleware, and local VIP
   forwarding worked.

6. A parallel public LoadBalancer Service allocated `166.19.14.14`. It
   advertised the same healthy Traefik endpoint but timed out from both the
   Marin controller and another external network. Changing only this recovery
   Service from `externalTrafficPolicy: Local` to `Cluster` did not change the
   result.

7. Cilium on node `g8feb7e` reported its BGP session to DPU
   `169.254.100.0:179` as `established`, with over 309 hours of uptime. Its
   advertised routes included `166.19.14.7/32` and the recovery
   `166.19.14.14/32`, both with next hop `10.186.204.57`.

8. Traefik was temporarily scaled to two replicas. Pod anti-affinity placed the
   second replica on `g8fb2f2`; that node independently advertised both VIP
   routes to its DPU with next hop `10.186.204.91`. External connections still
   timed out. The extra replica and recovery Service were then removed.

## Root cause

CoreWeave's US-EAST-08A edge did not propagate or forward the cluster's public
LoadBalancer VIP routes to external networks. The Kubernetes data plane
advertised both tested `/32` routes to CoreWeave DPUs over established BGP
sessions from two nodes. Both routes remained reachable inside the cluster and
unreachable outside it.

The failing layer was downstream of Cilium's DPU advertisement and upstream of
external clients. Iris, Traefik, DNS, TLS, and the IP allowlist did not cause
the outage.

## Fix

No Marin-side fix was applied. The temporary LoadBalancer Service and second
Traefik replica did not bypass the CoreWeave edge failure and were removed.
CoreWeave must restore external routing for `166.19.14.7/32` and inspect route
export for tenant `208261`, cluster `marin-us-east-08a`, region
`US-EAST-08A`.

Grafana now monitors `ListPeers` from the Marin controller and pages after a
peer remains unreachable for five minutes. This reuses the production
federation heartbeat, so no Grafana egress address was added to the CoreWeave
allowlist.

## How OPS.md could have shortened this

- Extend `lib/iris/OPS.md` under `CoreWeave (GPU) Operations` with a public
  LoadBalancer triage sequence: compare controller Service, Traefik cluster IP,
  VIP from inside the cluster, and VIP from an external host.
- Add `cilium-dbg bgp peers` and
  `cilium-dbg bgp routes advertised ipv4 unicast peer <dpu-ip>` commands to the
  same section. An established session plus an advertised VIP that is externally
  unreachable identifies a provider edge escalation.
- State that a source-IP allowlist rejection returns HTTP `403`; a TCP timeout
  occurs before the Traefik middleware and should be debugged as routing or
  LoadBalancer health.

## Artifacts

- `.agents/ops/2026-07-28-coreweave-public-vip-blackhole.md`
