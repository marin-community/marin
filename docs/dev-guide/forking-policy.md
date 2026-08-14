# Forking Policy

When Marin needs a modified version of an upstream package, we maintain a fork
under the `marin-community` GitHub organization rather than vendoring code into
the Marin monorepo. Each fork carries Marin's patches on top of an upstream base,
and an automated weekly process keeps it close to upstream. This page describes
how Marin pins a fork and how the refresh runs.

## How Marin pins a fork

Every fork is pinned under `config/external/`. The pins feed
`config/update-external.py`, which regenerates
`lib/marin/src/marin/external_dependencies.py`; nothing else imports a fork
revision directly. There are three pin kinds:

- Isolated uv lock (`evalchemy`, `harbor`, `MarinSkyRL`): the fork is a git
  dependency in `config/external/<fork>/uv.lock`. `uv run
  config/update-external.py <fork>` advances the lock and regenerates the pins.
- Descriptor SHA (`vllm` TPU stack, `tpu-inference`): the fork tips live in
  `config/external/vllm/tpu.toml`. This stack runs from an isolated `uvx`
  environment, so its `jax`/`jaxlib`/`libtpu`/`torch` come from the forks' own
  dependencies and there is no workspace lock change.
- Release wheel (`vllm` GPU): `config/external/vllm/gpu.toml` records a
  promoted, immutable wheel per architecture. See the GPU release pipeline below.

## The weekly refresh

`.github/workflows/ops-fork-ferry.yaml` runs every Monday at 08:00 UTC. It has one
matrix leg per fork unit (`tpu-vllm`, `vllm-gpu`, `evalchemy`, `harbor`). Each leg
mints a GitHub OIDC token, exchanges it for a short-lived Loom token, and launches
one Weaver session for that unit. There is no stored PAT.

`MarinSkyRL` is pinned and refreshable through the skill on demand, but is not yet
in the weekly rotation. A human runs the skill for a single fork the same way.

The session runs the `refresh-fork` skill
(`.agents/skills/refresh-fork/SKILL.md`), which owns the migration procedure:
select a new upstream base, replay Marin's overlays for overlay forks, stage the
result on a `<branch>-next` branch, re-pin Marin, run the fork's declared
end-to-end test, and on green open one draft Marin PR requesting the descriptor's
reviewer. On an unresolved external blocker it files a "can't migrate" issue
instead of a PR.

`config/external/migration.toml` is the per-fork descriptor. It records the
upstream repository, the pin kind, the fork branch the pin tracks, how to select a
new base, the validating e2e, the blocker assignee, and the nuances a refresh must
respect (a held torch revision, a tpu-inference known-good ceiling, or a
CUDA/torch/stable-ABI boundary that turns a bump into a migration).

## Validation

Each descriptor names one Marin e2e that runs before the PR opens:

| Fork | End-to-end |
|------|------------|
| `vllm` (TPU), `tpu-inference` | `experiments/evals/served_qwen3.py::QWEN3_TPU_INFERENCE` |
| `vllm` (GPU) | `tests/cluster/vllm/test_snowball_backend_parity.py` |
| `evalchemy` | `experiments/evaluation/configs/evalchemy/gsm8k-smoke.yaml` |
| `harbor` | `experiments/evaluation/configs/harbor/aime-smoke.yaml` |
| `MarinSkyRL` | `experiments/post_training/iceball_micro.py` |

When an e2e fails, the refresh reruns the same workload against Marin's current
pins on the old fork stack. It fixes only failures that pass on the old stack and
regress on the refreshed one. A workload already broken on the old stack is
recorded as a baseline failure and left for its own fix.

## The vLLM GPU release pipeline

The GPU pin resolves to a prebuilt wheel. The `marin-community/vllm`
fork builds an immutable cu130 wheel per architecture (x86_64 sm9.0, aarch64
sm10.0) through its own candidate and release workflows, validates the exact wheel
bytes on real GPUs, and publishes a GitHub release carrying
`marin-vllm-gpu-manifest.json`. The GPU overlay lives on the fork's `main`, which
the candidate build triggers on.

A refresh dispatches those workflows against the staged `main-next` branch, waits
for the promoted release, downloads the manifest, and re-pins with:

```sh
uv run config/update-external.py --promote-gpu-release marin-vllm-gpu-manifest.json
```

That command writes `gpu.toml` (release tag, source commit, version, torch
backend, and each arch's wheel URL and SHA-256) and regenerates the pins. Do not
hand-edit `gpu.toml`; the helper re-encodes the wheel URLs the way the pin
loader validates.

## Promotion

The refresh never force-moves a fork's stable branch. It stages on `<branch>-next`
and leaves the protected stable branch at the old tip; the draft Marin PR names
the `<branch>-next` to `<branch>` hard swap an admin performs after review. Because
the staged tip and the eventual stable tip are the same commit, the pins need no
change after promotion.

## Existing forks

| Fork | Repository | Tracks upstream | Pin |
|------|-----------|-----------------|-----|
| vllm (TPU) | [`marin-community/vllm`](https://github.com/marin-community/vllm) | [`vllm-project/vllm`](https://github.com/vllm-project/vllm) | descriptor SHA on the `tpu` branch (`tpu.toml`) |
| vllm (GPU) | [`marin-community/vllm`](https://github.com/marin-community/vllm) | [`vllm-project/vllm`](https://github.com/vllm-project/vllm) | release wheel from `main` (`gpu.toml`) |
| tpu-inference | [`marin-community/tpu-inference`](https://github.com/marin-community/tpu-inference) | [`vllm-project/tpu-inference`](https://github.com/vllm-project/tpu-inference) | descriptor SHA (`tpu.toml`) |
| evalchemy | [`marin-community/evalchemy`](https://github.com/marin-community/evalchemy) | [`mlfoundations/evalchemy`](https://github.com/mlfoundations/evalchemy) | isolated uv lock |
| harbor | [`marin-community/harbor`](https://github.com/marin-community/harbor) | [`harbor-framework/harbor`](https://github.com/harbor-framework/harbor) | isolated uv lock |
| MarinSkyRL | [`marin-community/MarinSkyRL`](https://github.com/marin-community/MarinSkyRL) | [`NovaSky-AI/SkyRL`](https://github.com/NovaSky-AI/SkyRL) | isolated uv lock |

## When to fork

Fork only when upstream has not accepted patches you need, the upstream release
cadence is too slow for a fix you need now, or you need a custom build such as
TPU-specific or GPU-specific wheels. Prefer upstreaming changes. A fork is ongoing
maintenance overhead even with the weekly refresh, and every retained patch needs
a reason and a drop condition so the overlay shrinks as upstream absorbs it.
