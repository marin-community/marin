# Debugging log for federated endpoint proxy saturation

Determine why capability URLs for healthy endpoints on federated CoreWeave clusters hang at the Marin parent controller.

## Initial status

Issue #7448 reported that `/proxy/t` requests to `cw-us-east-02a` and `cw-rno2a` returned no bytes, while the same endpoint served requests inside its pod and parent-local endpoints remained reachable.

## Hypothesis 1

The parent controller tried to dial the pod IP mirrored from the peer, which is not routable across clusters.

## Results

The parent resolves a remote endpoint to `FederatedEndpointProxy`, which dials the peer controller URL and lets that controller dial the pod. It does not dial the mirrored pod IP. A local endpoint configured with `https://iris-cw-us-east-02a.oa.dev` as its upstream reached peer `/health` in 0.30 seconds and peer `/proxy` authentication in 0.35 seconds. The public load balancer, TLS path, catch-all ingress, peer controller, and `/proxy` route were reachable.

## Hypothesis 2

The valid federation bearer blocked in peer authentication or the received-job ownership query.

## Results

A live Qwen3-0.6B H100 endpoint reproduced the hang, but the parent never received response headers from the peer. Parent logs showed the dispatch at 21:05:34 and an empty `Proxy peer error` at 21:15:34, exactly the endpoint's 600-second timeout later. The peer-specific hypothesis did not explain why requests to both CoreWeave peers failed together.

## Hypothesis 3

Long-running inference requests exhausted the singleton federated proxy client's 100-connection pool. A later request waited for a pool slot for the full endpoint timeout.

## Results

Before the fix, `FederatedEndpointProxy` shared one `httpx.AsyncClient` across every peer and endpoint, with `_HTTPX_LIMITS` capped at 100 active connections and keepalive disabled. `httpx.PoolTimeout` renders as an empty string, matching the parent log. Historical parent logs contained a burst of the same empty error for `cw-rno2a`, and the fresh `cw-us-east-02a` reproduction emitted it after exactly 600 seconds. The successful transport control used the separate direct-endpoint client pool.

## Changes to make

- Add a real-HTTP regression test that holds 100 federated requests open and verifies a probe reaches the upstream without waiting for them.
- Remove the controller-wide active-connection cap while retaining disabled keepalive.
- Log transport exceptions with their representation so exceptions with an empty string remain identifiable.

## Results

The real-HTTP test failed on the existing code: 100 requests reached the peer and remained open, while the probe did not reach the peer within two seconds. The proxy now leaves `max_connections` unbounded, retains `max_keepalive_connections=0`, and logs unexpected transport exceptions with their representation.

The two proxy test files passed 58 tests. The Iris unit suite passed 2,839 tests with one skip. The repository suite passed 1,158 tests and failed three Arrow Flight tests because this Docker container discovered the host VM's `10.128.0.14` address through GCP metadata and could not connect to that address from its network namespace. Re-running the complete seven-test Arrow Flight file with metadata discovery disabled and the container hostname advertised passed.

## Future work

- [ ] Add proxy concurrency and pool-wait metrics so saturation is visible before requests time out.
