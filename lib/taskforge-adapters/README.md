# Taskforge adapters

`marin-taskforge-adapters` holds the code through which Taskforge (`lib/taskforge`) talks to a
specific provider or cluster. Each module implements a Protocol the library declares or constructs
what the library takes as an argument, so the library imports none of them.

```
src/taskforge_adapters/
  glm_client.py      GlmClient, the GLM-5.3 ModelClient: streamed chat completions through the openai
                     SDK, retries, infrastructure holds, and the endpoint sources (FixedEndpoint,
                     RelayEndpoint resolving an Iris GLM relay job)
  web.py             web_tools, the agent loop's web_search and web_fetch tools over Parallel's REST API
  parallel.py        ParallelWeb, the agentic session's Web over Parallel's REST API
  image_builder.py   IrisImageBuilder, the ImageBuilder that builds with kaniko and pushes with crane in
                     two Iris jobs; image_push.py is the push job's program
  finelog_ledger.py  FinelogLedger, the Ledger that writes spans to Finelog under an Iris task
  backends.py        machine_factories, which constructs the selected shellbox backends (ShellSim, the
                     local sandbox, Docker, Iris gVisor)
  build_process.py   run_build_in_process, one build in a child process under a wall clock; the child
                     constructs its GlmClient, image builder, backends and web tools from a JSON request
  testing.py         FakeGlm, a scripted fake of the GLM router as a local HTTP server, and run names
scripts/
  build_image_job.py        builds a directory's DockerBuild with IrisImageBuilder from a laptop
  cluster_rollout_probe.py  runs validation trials inside an Iris task with GLM from a relay job
```

The scripts document their submit commands in their docstrings. The capability experiment's drivers
(`experiments/post_training/capability_driven_envs`) run in this package's environment:
`uv run --package marin-taskforge-adapters --frozen python -m ...`.

## Testing

```bash
uv run --package marin-taskforge-adapters --frozen --group test pytest \
    -c lib/taskforge-adapters/pyproject.toml lib/taskforge-adapters/tests
cd lib/taskforge-adapters && uvx --from 'pyrefly>=1.0.0,<1.1.0' pyrefly check
```

Live tests carry `@pytest.mark.live_glm` and take the inputs `lib/taskforge/README.md` § Testing
describes.
