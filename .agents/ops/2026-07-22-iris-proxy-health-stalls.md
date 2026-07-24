---
date: 2026-07-22
system: iris
severity: degraded
resolution: mitigated
pr: none
issue: none
---

## TL;DR

- `cw-us-east-02a` emitted repeated readiness and liveness probe timeouts for its live, ready Iris controller Pod. The Pod was not stale and never restarted.
- The controller was proxying approximately 180 long-lived inference streams while its control loop repeatedly listed and decoded more than 1,500 Pods. The proxy event loop intermittently stopped answering `/health` for more than 10 seconds.
- The controller image included `uvloop`, but Iris's custom RPC-executor launcher constructed an asyncio selector loop directly and bypassed Uvicorn's `loop=auto` factory.
- The launcher now preserves Uvicorn's loop factory while retaining the 64-thread RPC executor. A local 1,600-stream benchmark improved median completion throughput from 74.1 to 96.9 streams per second and reduced median per-trial health p99 from 8.85 to 2.19 seconds, with no stream or probe errors.
- The Kubernetes probe remains at its existing 10-second timeout. Increasing it would hide genuine controller unresponsiveness; separating the proxy data plane remains the scaling path beyond this event-loop correction.

## Original problem report

Grafana showed this warning for `cw-us-east-02a`:

```text
Readiness probe failed: Get "http://10.0.5.136:10000/health": context deadline exceeded (Client.Timeout exceeded while awaiting headers)
iris
Pod/iris-controller-555c688864-fz6mm
Unhealthy
```

The operator asked whether the event was a false alarm or stale Pod, what caused it, how to adjust the probe, and how Iris could support as many as 1,600 concurrent proxied connections.

## Investigation path

1. `iris-controller-555c688864-fz6mm` was the current Deployment Pod and EndpointSlice target at `10.0.5.136`. It had been running since July 21 with zero restarts. Readiness and liveness failure event counts continued increasing during the investigation, while intermittent successes kept the Pod ready.

2. A loopback request to `/health` from inside the controller container also timed out. This ruled out a stale Grafana row and isolated the failure from Traefik and external routing. Grafana's warning-events panel retains raw Kubernetes events, but these events described a real transient stall.

3. The controller used approximately 3.1 GiB of its 64 GiB limit, the node reported no pressure, and sampled CPU throttling was absent. The event was not an OOM or CPU-quota failure.

4. The controller held approximately 421 inbound connections on port 10000. Of these, 180 arrived from Traefik and matched 180 outbound connections to one inference endpoint. The same interval contained 668 proxied completion requests and 1,094 Kubernetes exec requests.

5. The cluster held approximately 1,541 Pods and 537 Kueue Workloads. Kubernetes Pod and Workload LIST decoding took roughly 2.8 to 3.0 seconds on average and reached 6.1 seconds. This control-loop work contended for the Python GIL with the proxy server.

6. `py-spy` captured the controller-server thread in Starlette `StreamingResponse`, Uvicorn's `send`, and asyncio selector-transport writes. Logs emitted thousands of `asyncio socket.send() raised exception` messages per minute while clients disconnected under load.

7. The running image contained `uvloop 0.22.1` and `httptools`, and Uvicorn was configured with `loop=auto`. `_install_rpc_executor`, however, called `asyncio.new_event_loop()` and therefore always created `_UnixSelectorEventLoop`.

8. A regression test installed a marked Uvicorn loop factory and failed against the old launcher. Replacing the manual lifecycle with `asyncio.Runner(loop_factory=server.config.get_loop_factory())` selected the marked loop and retained the named `rpc-handler` executor.

9. A local benchmark drove real HTTP requests through Uvicorn, Starlette, `EndpointProxy`, HTTPX, and a streaming Uvicorn upstream. It alternated selector-loop and auto-loop trials at 200, 800, and 1,600 concurrent streams and issued health requests on the same proxy event loop.

10. At the 1,600-stream target, three production-parity trials completed with zero stream or health errors. The auto-selected uvloop improved median throughput by 30.8% and reduced median per-trial health p99 by 75.3%. Lower-load results were variable: the 200-stream run improved both measures, while a separate 800-stream run was flat on health and 13.4% slower on throughput.

## User course corrections

- The operator expanded the initial warning-event check into controller-load analysis, then asked about Python threading, subprocess proxying, and the knobs needed for 1,600 streams.
- After the event-loop defect was identified, the operator requested the best safe performance from the current single-process configuration, a reproducible local benchmark, a PR with measured results, and authorization to roll `cw-rno2a` after local validation.

## Root cause

The probe timeouts were genuine transient event-loop stalls, not stale monitoring state. Long-lived proxy streams and disconnect handling shared one Python process with a control loop that performs expensive Kubernetes object decoding. Under that load, the selector event loop could not schedule `/health` within the 10-second probe timeout.

Iris unintentionally disabled an optimization already present in its image. `_install_rpc_executor` replaced Uvicorn's server runner so synchronous Connect handlers used a 64-thread executor, but its manual `asyncio.new_event_loop()` also bypassed Uvicorn's configured loop factory. Uvicorn's `auto` setting therefore never selected the installed uvloop implementation.

Threads are appropriate for synchronous RPC handlers but do not move async HTTP forwarding or Python decoding off the GIL. Running multiple Uvicorn workers inside the existing controller would duplicate control loops and state ownership. A separate, horizontally scaled proxy Deployment is the correct future process boundary if one controller must sustain this load continuously.

## Fix

`_install_rpc_executor` now creates an `asyncio.Runner` with `server.config.get_loop_factory()`, sets the existing 64-thread default executor on that loop, and passes any pre-bound sockets through to `server.serve`. This restores Uvicorn's installed `uvloop` and `httptools` fast path without changing controller ownership or request semantics.

`benchmark_controller.py proxy` provides the local comparison. Its default scenario runs three balanced trials at 200, 800, and 1,600 streams, uses an uncapped HTTPX load client like the production proxy, matches the controller's 120-second keepalive, records the actual loop class, and reports throughput plus health latency and errors.

No probe timeout was increased. The 10-second timeout exposed a real data-plane/control-plane coupling problem, and the existing failure threshold prevented an unnecessary restart while the controller recovered between stalls.

## How OPS.md could have shortened this

- Distinguish the Grafana warning-events table from stateful alerts: verify the event's last timestamp and count, then inspect the owning Deployment, Pod UID, restart count, and EndpointSlice before treating a row as stale.
- For controller health timeouts, test `/health` from inside the Pod before changing ingress or probe settings. A loopback timeout identifies application-loop starvation.
- Record the controller's socket fan-out with `ss` or `/proc/<pid>/net/tcp`, correlate inbound ingress connections with outbound endpoint connections, and sample individual Python threads with `py-spy`.
- Confirm the actual event-loop implementation, not only the Uvicorn configuration. A custom runner can bypass `loop=auto` even when `uvloop` is installed.

## Artifacts

- `.agents/ops/2026-07-22-iris-proxy-health-stalls.md`
- `lib/iris/src/iris/cluster/controller/controller.py`
- `lib/iris/tests/cluster/controller/test_controller_server.py`
- `lib/iris/scripts/benchmark_controller.py`
- Kubernetes Pod: `iris/iris-controller-555c688864-fz6mm`
