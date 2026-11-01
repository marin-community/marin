# Tau3 Pi and ACP image

Tau3 uses the public `ghcr.io/marin-community/rollout-images` package for its main
sandbox and retail and airline user/judge runtimes. The main recipe adds the Pi
and ACP dependencies to the canonical Tau3 image. It preserves
`tau2-bench[knowledge]` at commit `fc0055dc4e0a316c3f83133267fbd6faaa770992`
and its data. The runtime images retain their canonical digests, including the
existing server and task configuration.

[`dependencies.json`](dependencies.json) pins the parent digest, Node archive
and checksum, platform, runtime digests and recipe revision.
[`package-lock.json`](package-lock.json) locks Pi, pi-acp, acpx, the Pi MCP extension and their npm
dependencies. [`mcp.lock`](mcp.lock) and [`acp.lock`](acp.lock) lock the Python
dependencies with hashes. The image exposes `node`, `pi`, `pi-acp` and `acpx` on
`PATH`, MCP to `python3`, and the ACP SDK through
`/opt/harbor-acp-venv/bin/python`.
Pi loads the installed MCP extension from an absolute path in
`/root/.pi/agent/settings.json`; Harbor stages the task's MCP servers separately.
The extension is pinned to `pi-mcp-extension@1.10.0`; its SDK peers use
`@earendil-works/pi-ai@0.80.5` and `typebox@1.1.38`, matching Pi's own versions.

## Build or reuse

Authenticate Docker to GHCR with a token that can write packages. In the GHCR
package settings, set `rollout-images` visibility to **Public** before an Iris
worker uses it. The Dockerfile's repository source label links the package to the
repository; it does not change visibility. Stage the
canonical main, retail and airline manifests in `rollout-images`, preserving
their digests, before building the main recipe.

Preview the tag without contacting the registry:

```bash
uv run --no-project python -m experiments.evaluation.images.tau3.build --plan
```

Resolve or build the main image and save its immutable reference:

```bash
uv run --no-project python -m experiments.evaluation.images.tau3.build \
  --output artifacts/tau3-images.json
```

The command hashes the dependency locks and declared configuration into
`rollout-images:tau3-pi-<identity[:16]>`. An existing tag is reused. A missing tag
is built for `linux/amd64`, pushed with gzip layers, and read back from GHCR.
Authentication and other registry errors stop the command. The output JSON
records the complete dependency identity, main digest reference and canonical
runtime references. Use those immutable references in prepared task inputs.

`--repository` changes the main image's destination. `--base-image` accepts a
digest-pinned parent override and changes the dependency identity. The canonical
runtime references stay as declared in the manifest.

## Update dependencies

Edit `package.json`, `mcp.in`, `acp.in` or `dependencies.json`, then regenerate the
affected locks:

```bash
cd experiments/evaluation/images/tau3
npm install --package-lock-only --ignore-scripts --no-audit --no-fund
uv pip compile --generate-hashes --no-header --python-version 3.12 \
  --python-platform x86_64-unknown-linux-gnu mcp.in --output-file mcp.lock
uv pip compile --generate-hashes --no-header --python-version 3.12 \
  --python-platform x86_64-unknown-linux-gnu acp.in --output-file acp.lock
```

Bump `recipe_revision` when changing Dockerfile behavior. The builder hashes
dependencies and configuration; it does not hash repository source files.
