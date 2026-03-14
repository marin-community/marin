# Iris Operations

All Iris operations are available via the CLI:

```bash
uv run iris --help
```

## Connectivity

`iris --config=... cluster dashboard` establishes an SSH tunnel to the controller and prints the dashboard URL. Keep the terminal open.

Controller and worker both expose a `/health` endpoint:

- **Controller** (`http://localhost:10000/health`): `{"status": "ok", "workers": <int>, "jobs": <int>}`
- **Worker**: `{"status": "healthy"}`
