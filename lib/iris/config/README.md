# Iris cluster configs

`iris --cluster=<name>` resolves `<name>` against the top-level `*.yaml` files
here (plus `~/.config/marin/clusters`, which wins on conflict). Subdirectories
are not searched.

Naming convention:

- **Live clusters** — bare names. GCP: `marin.yaml`, `marin-dev.yaml`.
  CoreWeave: `cw-<region>.yaml` (e.g. `cw-us-east-02a.yaml`, `cw-rno2a.yaml`).
- **`ci-*.yaml`** — CI and test-harness configs; not real long-lived clusters.
- **`examples/`** — reference templates. Deliberately outside name resolution;
  pass an explicit path to use one.
