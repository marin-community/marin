# marin-finelog-server (pyext)

In-process [finelog](../..) server, exposed to Python via PyO3 and shipped as
the `marin-finelog-server` wheel — a standalone dist that consumers needing the
in-process server (e.g. the iris controller) depend on explicitly. The pure
`marin-finelog` client does not depend on it.

`finelog_server.EmbeddedServer` boots the same axum app the `finelog-server`
binary serves, on an owned tokio runtime bound to a local port. Callers talk to
it over the normal RPC contract (`finelog.client.LogClient` / proxies), so there
is exactly one server implementation. Iris's controller uses it as the local
log-server fallback when no external `/system/log-server` endpoint is set.

```python
from finelog_server import EmbeddedServer

with EmbeddedServer(log_dir="/tmp/logs") as server:
    print(server.address)  # http://127.0.0.1:<ephemeral>
    # ... talk to it with finelog.client.LogClient.connect(server.address)
```

The heavy server/store code lives in the `finelog` Rust crate; this crate is a
thin lifecycle shim. Build from source for development with
`python scripts/rust_mode.py dev && uv sync`.
