# Frozen agentic SFT screens

[Experiment #10213](https://github.com/marin-community/marin/issues/10213)
compared SWE-bench Verified random-100, TBLite-100, GSM8K and BFCL-Parity.
The committed `experiments/evaluation/task208-screens/screen-task-ids.json`
records the task identities and registry/source revisions. SWE uses the
campaign's existing random-100 subset; TBLite uses all 100 canonical tasks.
No task is resampled when comparing checkpoints.

Print the four-screen launch command:

```bash
uv run python -m experiments.evaluation.sweep_screens \
  --model-config experiments/evaluation/task208-screens/v0.yaml \
  --version <new-run-identity>
```

Add `--run` only after authorizing the inference allocation, and set
`MARIN_PREFIX` explicitly. The launcher requests one H100x8 serve on cw-rno2a
at interactive priority and does not retry a failed submission. Credentials
come from the existing evaluation launch environment.

These files preserve historical settings, not the entire executable runtime.
Original screens used policy revision `75a5690b`; subsequent launches used
`4d45932e39`, which fixed the FineStore legacy `_gen` reader without changing
scoring settings. Harbor used capability routing; a transient capability-proxy
502 burst required a selective GSM8K retry. Fresh mini-swe-agent installations
later needed Harbor `38742b0f` to include `mcp`. Record the actual runtime
revision and endpoint route on each new run. Current main's Iris GPU default
is NCCL WARN; the launch environment sets WARN explicitly too.

## Recover an artifact-failed record

```bash
uv run python -m experiments.evaluation.recover_sweep_record \
  --record <record-uri> --original-sha256 <sha256> \
  --assembly-commit <reader-commit> --output <new-local-directory>
```

This reads the active FineStore view, requires complete error-free scored
coverage, and writes an original-record backup, proposed replacement and
provenance locally. It performs no generation, modifies no source archive
and does not publish the proposed record. Review it before copying it to
storage.

For an already-completed selective retry, add `--retry-results <archive>`.
The audit requires identical response identities, unchanged successful
payloads and successful replacements of transport failures only. It clears
old timing/inference counters because they are not merged totals. Retry
creation and generation require separate authorization. In the original
GSM8K recovery, 1,207 responses were retained and 112 were regenerated.
