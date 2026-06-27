# buoy

A better viewer for [wandb](https://wandb.ai) runs, hosted on Iris. Pick a run,
buoy mirrors its metrics + config + TPU profile into a refetchable GCS cache, and
serves metric plots plus the **real xprof profile UI** in one place — turning
"look at this run's profile" into a link instead of a hand-download ritual.

wandb stays the source of truth; the GCS copy is a TTL-cleaned cache, not an
archive.

## Run it

```bash
buoy --cluster marin            # submit the service job, register /serve/buoy
# -> open https://iris.oa.dev/proxy/serve.buoy/
```

The service self-registers as an Iris endpoint and is reached through the
controller proxy. `WANDB_API_KEY` is auto-injected from your submitting shell.

## Architecture

```
SPA (static/)  ──►  Starlette API (app.py)  ──►  cache (GCS via cache.py)
                          │                            ▲
                          ├─ MirrorManager (mirror.py)─┘   async, poll for ready
                          └─ XprofManager (xprof.py) ──►  xprof subprocess (local logdir)
```

See `AGENTS.md` for module layout and the hard requirements baked in from the
prototype, and `.agents/projects/buoy/design.md` for the full design.

## v1 scope / deliberate follow-ups

This is the first prod cut. Intentionally deferred:

- **Frontend** ships as a no-build vanilla-JS SPA (plotly.js via CDN). The
  design's Vue + Vite tree is a follow-up.
- **Deploy** uses a self-registering Iris job (`buoy` CLI). The always-on
  k8s `ClusterIP` + cluster-config `endpoints` model is a follow-up.
- **Auth** relies on the controller proxy (no per-user session yet).
- **Mirror scope** covers history + config + summary + the `jax_profile`
  artifact; large `code`/`used_artifacts` are not mirrored.
