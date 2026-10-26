# Pi-RL development sandbox image

This is a **development image**, not an approved RL sandbox. It contains Python,
Node 22.19.0, and `@earendil-works/pi-coding-agent` 1.3.0. The Python and Node
base images are digest-pinned; npm dependencies are installed from the committed
`package-lock.json` during the Docker build. It contains no task files, model
weights, endpoint tokens, or credentials.

The `Pi RL development image` pull-request workflow builds this Dockerfile
without publishing and checks the Pi/recorder path in a network-isolated,
read-only container. It has `contents: read` only; passing that check is not an
image scan or permission to publish. The intended later release path is a
reviewed workflow that publishes a `dev-<commit>` tag to Marin's GHCR namespace,
records the OCI digest, scans the final image, and has Daytona pull disposable
snapshots from that digest. No such publication is configured yet.

Build locally with a `dev-` tag (do not push this tag as `latest`):

```sh
docker build -t pi-rl:dev-<attempt> docker/pi-rl
```

A Harbor Pi trial using this image must select
`agent.kwargs.tooling_source: preinstalled` and
`agent.kwargs.version: "1.3.0"`. Setup then checks the baked Node/Pi versions
and recorder dependency path and **fails rather than installing** apt, nvm, or
npm packages. Default Pi trials retain their existing per-trial install until
they explicitly opt in. The image is compatible with Shellbox's Linux amd64
registry-image path; use a digest-addressed `docker_image` in the Harbor task.

Local validation on 2026-10-22: `pi-rl:dev-20261022-sandbox-v2` ran an offline
version/dependency check, 73 Harbor Pi unit tests, and one full Docker-backed
SkyRL CPU policy → Pi → captured model turn → private grading → ETL smoke.
The Docker smoke used a temporary **localhost-only** registry to obtain a
digest-addressed test image. That registry has been removed. Reproduce it with
a local registry image digest using `RUN_HARBOR_PI_DOCKER=1`,
`PI_RL_DEV_IMAGE=localhost:<port>/pi-rl@sha256:<digest>`, and the opt-in
`test_pi_hydrated_docker.py` (`-m slow`). This is not a Daytona snapshot or a
GPU training result.

A disposable **Daytona eval-account** snapshot canary also ran on 2026-10-22.
The Daytona CLI pushed the locally prebuilt `dev-` image; Daytona did not build
the Dockerfile or install Pi. Snapshot `dev-pi-rl-beta-20261022-1630-v2`
started one sandbox with egress blocked. Inside it, Node and Pi passed exact
version and recorder-dependency checks, Pi made two requests to a synthetic
*loopback* OpenAI-compatible endpoint, executed one bash tool, and returned an
answer graded on the host. One observed sandbox create took 8.06 s; the Pi
mock-model turn took 0.45 s. These are **single-sample mock timings**, not real
worker ↔ Daytona model latency or a throughput benchmark. The sandbox and
snapshot were deleted and their absence confirmed. One unrelated eval-account
snapshot whose last use exceeded the authorized one-hour retention threshold
was also removed during the required quota preflight; provider-owned general
snapshots were untouched.

After rebasing onto Marin `origin/main`, a separate **full Daytona CPU-policy
worker smoke** passed on 2026-10-22: the prebuilt dev image was pushed to
snapshot `dev-pi-rl-beta-20261022-full-r1` in the eval account, then two
physical trials ran through the real SkyRL task worker, Harbor/Pi in-memory
capture, private exact-answer verifier, and trajectory ETL. Both the train and
eval trial verified; the train trial retained exact sampled policy token IDs
and logprobs, student-only masks, and reward 1.0. The single pytest invocation
took 91.41 s, including two short-lived HTTPS tunnel registrations (33.1 s and
34.3 s) used solely to let remote Daytona reach the synthetic CPU model on the
VM. The tunnel carried no private task data, production credentials, or model
weights. This is not a representative model-turn latency measurement. The
tunnels, trial sandboxes, and snapshot were removed; absence of the snapshot
and its sandboxes was checked. No durable registry image was published: the
local and transient-registry image digests do not attest to a retained image.

The earlier loopback-only Pi canary did **not** exercise worker capture or
private verification; the later CPU worker smoke did. Neither CPU test exercised
vLLM, a tool-using model trajectory, GPU optimizer update, or restart. The GPU
serving test below is a separate gate, not a training result. Before shared
training use, publish a separately tagged `dev-` image to an authorized
registry, use its immutable digest, scan and approve the final image, and
qualify the real Iris model endpoint without the external dev-test tunnel.
Do not use the RL account or promote these disposable snapshots to training on
the strength of these tests.

A 2026-10-22 attempt to submit one H100 `marin-serve iris` vLLM job to
`cw-rno2a` did **not** reach a GPU. CoreWeave rejected the first root handoff
because storage credentials were missing. A second submission sourced the new
`~/.config/marin/credentials/coreweave.env` file and used the newer Iris client
from Marin `c35dff75f`, which forwarded the keys as secrets rather than plain
environment variables. The peer still rejected the handoff as missing both keys.
At 20:08 UTC the Marin controller still reported version `7981e039d4 (held)`,
older than the two secret-handling changes (#10475, #10474). All three
CoreWeave peers' rollout records show the same controller image digest committed
between 19:18 and 19:25 UTC; they do not establish the parent was updated.
The parent rollout record's `6981c50544` image tag is the **git tree hash**
of that running `7981e039d4` commit, not a different revision. They agree.
[The #10474 handoff review](https://github.com/marin-community/marin/pull/10474#issuecomment-6348765318)
explicitly says this protocol has no mixed-version negotiation: both
controllers, submitter, and task image need to understand the new secret map.
It also found no cross-controller secret-handoff regression test. The observed
rejections are consistent with that version skew, though they do not isolate
which handoff step dropped the map. This is a submission-path blocker, not a
vLLM, Daytona, Pi, or model failure. No shared controller was restarted and no
GPU allocation was made at that point. No keys were passed in plain
environment variables and the federated root-job identity check was not bypassed.

## GPU-vLLM + Daytona eval worker smoke, 2026-10-22

At 20:22 UTC Marin's live controller reported `c7c14a252d` and healthy; all
three CoreWeave peers were reachable with committed rollout records. A bounded
federated CPU canary then confirmed that both newly supplied CoreWeave storage
keys reached a CoreWeave task as nonempty mounted files, without printing the
values. A first no-sync canary failed only because its bare task image lacked
the optional `rigging` Python module; the stdlib-only repeat succeeded. The
new keys were sourced for Iris submission, not added as plain job environment
variables.

One H100 `marin-serve iris` root job,
`/zack/pi-vllm-dev-20261022-202527`, served pinned
`Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca` with vLLM on
`cw-rno2a`. A model API probe confirmed `/tokenize`, exact prompt and selected
response IDs, and aligned token logprobs. The locally prebuilt Pi image was
pushed to the disposable **eval-account** Daytona snapshot
`dev-pi-rl-beta-20261022-gpu-full-v1`. The worker ran on the validation VM;
a temporary HTTPS tunnel carried the synthetic task's Pi requests to its
per-trial model proxy, which routed inference through the GPU vLLM endpoint.
This external tunnel is a dev-test expedient, not approved production ingress.

Two **independent physical** strict smoke trials (`gpu-v6` and `gpu-v7`)
passed in 54.23 s and 63.53 s respectively (pytest wall clock, not pure model
latency). Each Pi rollout generated one bash tool request and a follow-up GPU
model turn, privately verified the file-derived answer with score 1.0, and
returned two per-turn GRPO training rows. The two `1.0` rows *within each*
trial repeat one physical verifier result; they are **not** two graded trials.
For each served turn, the shared-worker ETL response and prompt IDs matched
the model client, the selected-token logprobs matched exactly, and student
loss masks and source tags aligned with served response tokens. Earlier
attempts included test-harness errors, a one-turn reward-0 answer, and one
correct two-turn trial whose test failed an overly strict final-text assertion.
None was admitted to an optimizer batch or manufactured into a GRPO trial.

The GPU serve job was cancelled after testing. The Daytona snapshot and any
sandbox belonging to it were removed, and the eval account had no remaining
non-general snapshots or Shellbox sandboxes at final inventory. This proves a
narrow GPU-*serving* to Daytona worker/capture/grade/ETL gate; it does **not**
prove CoreWeave GPU training, optimizer update, weight sync, checkpoint
restart, reliability under the eight-GPU recipe, or a durable GHCR image.
