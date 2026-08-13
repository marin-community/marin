# Blank-`main` README template

Drop this on the `main` branch of a fork that carries more than one Marin pin (today
only `marin-community/vllm`), replacing whatever `main` held. Marin pins exact commits
and wheels on the per-pin branches below, so `main` holds nothing else. Adjust the
branch list and pin paths per fork.

---

# This fork is pinned by branch, not by `main`

`marin-community/vllm` carries two independent Marin pins, each our patches on a
different upstream base, so they cannot share one `main`:

- **`gpu`** — Marin's patches on vLLM upstream head. The fork's release pipeline builds
  this branch into the GPU wheel Marin pins in `config/external/vllm/gpu-release.toml`.
- **`tpu`** — Marin's patches on the tpu-inference-blessed base. Marin pins this as a
  git source in `config/external/vllm/tpu-forks.toml`.

`main` is intentionally blank. Marin resolves exact commits and wheels on the branches
above; each is refreshed on a `<branch>-next` staging branch and promoted with Marin's
`refresh-fork` skill. Do not build from or advance `main`.
