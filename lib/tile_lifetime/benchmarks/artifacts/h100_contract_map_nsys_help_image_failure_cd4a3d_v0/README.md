# H100 evidence image Nsight help failure

GitHub Actions run
[31483568727](https://github.com/marin-community/marin/actions/runs/31483568727)
used exact source `cd4a3d64f56a273d8a8acacf961befd65987a33c` to build the
manual-only H100 evidence image. The Docker context gate passed and the
build-mounted `h100_evidence_nsys_help.py` executed against the pinned Nsight
Systems 2026.1.3 binary. It rejected the real `--capture-range-end` help syntax
before image export. All five legacy image jobs were skipped. The workflow was
dispatched once and was not rerun.

`run.json` is the terminal machine-readable workflow snapshot.
`source-identity.json` binds the checked-in workflow, Docker, manifest, and
validator inputs. `log-identity.json` records the complete remote job log's byte
count and SHA-256; `failure-excerpt.log` is only the bounded exact failure
excerpt, not a replacement for that complete log. `failure-analysis.txt`
states the supported control-flow boundary.
