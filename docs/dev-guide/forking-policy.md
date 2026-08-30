# Forking Policy

Marin maintains five fork repositories and six documented consumption lanes.
The vLLM fork has separate GPU and TPU lanes. This guide maps those lanes, the
checks that cover them, and the human review boundary around a refresh.

The mutable operating contract lives in the
[`refresh-fork` skill](https://github.com/marin-community/marin/blob/main/.agents/skills/refresh-fork/SKILL.md).
The
[`migration.toml` descriptor](https://github.com/marin-community/marin/blob/main/config/external/migration.toml)
records each upstream, pin kind, staging branch, and Marin end-to-end test.
[`config/external/`](https://github.com/marin-community/marin/tree/main/config/external)
holds the exact selected revisions and artifacts, and
[`config/update-external.py`](https://github.com/marin-community/marin/blob/main/config/update-external.py)
generates the consumer-facing pins. Use those sources for current commands and
values.

## Forks and consumers

Git-based forks are locked to an exact commit and run from a dedicated uv
project. Wheel-based forks are installed into an isolated `uvx` environment
from immutable release URLs. Consumers never resolve an unpinned fork head at
runtime.

| Lane | Upstream | How Marin consumes it | Fork-to-fork use |
| --- | --- | --- | --- |
| [`evalchemy`](https://github.com/marin-community/evalchemy) | [`mlfoundations/evalchemy`](https://github.com/mlfoundations/evalchemy) | Exact Git source in an isolated uv lock | None |
| [`harbor`](https://github.com/marin-community/harbor) | [`harbor-framework/harbor`](https://github.com/harbor-framework/harbor) | Exact Git source in an isolated uv lock | MarinSkyRL also locks the Harbor fork as a Git source |
| [`MarinSkyRL`](https://github.com/marin-community/MarinSkyRL) | [`NovaSky-AI/SkyRL`](https://github.com/NovaSky-AI/SkyRL) | Exact Git source in an isolated uv lock | Its optional vLLM environment uses Marin's GPU vLLM release wheels |
| [`vllm` GPU](https://github.com/marin-community/vllm) | [`vllm-project/vllm`](https://github.com/vllm-project/vllm) | Exact released wheel URL and SHA-256 for each supported architecture | MarinSkyRL can use the same release line |
| [`vllm` TPU](https://github.com/marin-community/vllm) | [`vllm-project/vllm`](https://github.com/vllm-project/vllm) | One exact public vLLM wheel requirement | The vLLM wheel metadata selects the exact tpu-inference companion wheel |
| [`tpu-inference`](https://github.com/marin-community/tpu-inference) | [`vllm-project/tpu-inference`](https://github.com/vllm-project/tpu-inference) | Installed transitively from the TPU vLLM wheel; there is no second Marin pin | Its nightly replaces the companion with tpu-inference HEAD while retaining Marin's selected vLLM wheel |

## Five distinct operations

These operations share checks and artifacts, but they start from different
inputs and produce different evidence.

| Operation | Trigger | Input and result | What success establishes |
| --- | --- | --- | --- |
| [Stable-head pin updater](https://github.com/marin-community/marin/blob/main/.github/workflows/ops-external-dependencies.yaml) | Every six hours, at minute 17 | Reads the stable heads of Evalchemy, Harbor, and MarinSkyRL; opens or updates an exact-lock Marin PR and auto-merges it after green checks | Marin follows an already-maintained stable fork head. It does not select a new upstream base or build wheels. |
| Fork PR checks | Pull request, push, or manual dispatch in the fork | Tests the fork's owned delta with CPU, packaging, metadata, and small fixture checks | The checked-out fork revision passes that repository's declared PR contract. Most PR jobs do not run Marin or an accelerator. |
| [Weekly refresh launcher](https://github.com/marin-community/marin/blob/main/.github/workflows/ops-fork-ferry.yaml) | Monday at 08:00 UTC; the skill also supports manual runs | Launches `tpu-vllm`, `vllm-gpu`, `evalchemy`, and `harbor` sessions. Each session selects a newer upstream base and ends with a qualified draft PR, an explicit no-op, or a blocker issue. | A green launcher run proves that the sessions were launched. The session receipt and draft PR hold the qualification result. |
| Scheduled nightly | Daily or manual workflow in each fork | Tests the current checked-out fork head with a larger workload, live service, or accelerator | That exact head passed the named nightly lane. A staged refresh is covered only when the nightly ran its exact revision or artifact. |
| Stable promotion | Human action after review | Moves a branch-based fork's stable branch to the reviewed staged commit and restores isolated selectors to the stable branch at the same SHA | Consumers can follow the reviewed stable head. Promotion creates no new test evidence. |

Refresh qualification reruns the fork's ordinary PR checks against the staged
candidate and adds the Marin end-to-end test declared in `migration.toml`. A
green refresh covers that candidate and that end-to-end path. It does not cover
every nightly workload. A green nightly covers its checked-out head; it does
not cover another staged SHA. A green PR check usually has no live Marin or
accelerator evidence.

MarinSkyRL uses the same branch-based refresh path on demand. It is outside the
Monday rotation.

## Qualification by lane

The skill, descriptor, receipts, and no-op or blocker routing form one control
plane for every lane. Each fork keeps its own smoke tests because the useful
failure signal differs by package and artifact. The fork workflows own
repository-level checks, while the descriptor owns one cross-repository Marin
gate. Scheduled nightlies add broader or longer coverage against the current
fork head.

| Unit | Ordinary fork checks | Refresh qualification | Scheduled nightly |
| --- | --- | --- | --- |
| vLLM GPU | [Marin CI](https://github.com/marin-community/vllm/blob/main/.github/workflows/marin-ci.yaml) checks the overlay, packaging, and release contract without a GPU | [Candidate](https://github.com/marin-community/vllm/blob/main/.github/workflows/marin-gpu-candidate.yaml) and [release](https://github.com/marin-community/vllm/blob/main/.github/workflows/marin-gpu-release.yaml) workflows build immutable x86_64 and aarch64 wheels, exercise the exact bytes on H100 and GB200, and then Marin runs Snowball backend parity | [H100 source nightly](https://github.com/marin-community/vllm/blob/main/.github/workflows/marin-nightly.yaml); it does not replace the released-wheel gate |
| vLLM TPU pair | [vLLM](https://github.com/marin-community/vllm/blob/main/.github/workflows/marin-ci.yaml) and [tpu-inference CI](https://github.com/marin-community/tpu-inference/blob/main/.github/workflows/marin-ci.yaml) cover their CPU-testable overlays and release metadata | One prerelease contains exactly two wheel assets; fresh resolution verifies the metadata edge, then Marin serves Qwen3 through the exact pair on TPU | [tpu-inference nightly](https://github.com/marin-community/tpu-inference/blob/main/.github/workflows/marin-e2e-nightly.yaml) keeps Marin's selected vLLM wheel and installs tpu-inference HEAD before the TPU service test |
| Evalchemy | [Marin CI](https://github.com/marin-community/evalchemy/blob/main/.github/workflows/marin-ci.yaml) and [PR e2e](https://github.com/marin-community/evalchemy/blob/main/.github/workflows/e2e-ci.yaml) checks cover the harness, graders, packaging, and small local cases | Marin runs the descriptor's small GSM8K smoke against the exact staged lock | [Larger GSM8K nightly](https://github.com/marin-community/evalchemy/blob/main/.github/workflows/e2e-nightly.yaml) includes score gates |
| Harbor | [Marin CI](https://github.com/marin-community/harbor/blob/main-next/.github/workflows/marin-ci.yaml) plus [Python tests](https://github.com/marin-community/harbor/blob/main-next/.github/workflows/pytest.yml) cover the Marin adapter, package, and Linux and Windows container paths | Marin runs the descriptor's small AIME smoke against the exact staged lock | [Live evaluation nightly](https://github.com/marin-community/harbor/blob/main/.github/workflows/marin-nightly.yaml) runs broader tasks and operational gates |
| MarinSkyRL | [CPU CI](https://github.com/marin-community/MarinSkyRL/blob/main/.github/workflows/cpu_ci.yaml) covers lint, packaging, launchers, and CPU training tests | An on-demand refresh runs the Iceball micro experiment against the exact staged lock | [GPU nightly](https://github.com/marin-community/MarinSkyRL/blob/main/.github/workflows/marin-nightly.yaml) runs H100 training and a GB200 rollout/train/broadcast cycle |

When a refresh end-to-end test fails, run the same workload against Marin's
current pin. A candidate-only failure is a refresh regression. A failure shared
by the candidate and current pin is recorded as a baseline or infrastructure
blocker; changing the fork candidate cannot establish a green qualification.

## Refresh lifecycle and review boundary

1. Select and freeze the upstream base, retained overlay, consumer head, and
   artifact inputs. If the selected base has not advanced and no metadata repair
   is needed, finish with an explicit no-op.
2. Review every retained patch against upstream. Rebase or replay is one
   possible Git step for a branch-based overlay. The refresh also includes
   staging, artifact production, Marin qualification, review, and promotion.
3. Prepare the lane's candidate:

   - Branch-based lanes stage the reviewed overlay on `<branch>-next` and pin
     Marin to that exact commit while the stable branch remains unchanged.
   - The GPU vLLM lane builds architecture-specific wheels from the staged
     branch, qualifies the exact bytes, publishes an immutable manifest, and
     pins those release URLs and hashes in `gpu.toml`.
   - The TPU group publishes exactly two wheels from reviewed source commits.
     The vLLM wheel requires the exact tpu-inference wheel, so Marin records one
     requirement in `tpu.toml`. The source commits are release evidence and do
     not use branch promotion.
   - MarinSkyRL follows the branch-based path only when an on-demand refresh is
     requested.

4. Run the fork checks and the lane's Marin end-to-end test. Preserve the exact
   commits, wheel digests, workflow runs, logs, and any current-pin control
   result in the draft PR.
5. Open a draft while every branch-based stable head remains at its old tip. Early
   agent-driven refreshes stay review-gated: a human checks the overlay choices,
   exact pins or artifacts, qualification receipt, and proposed promotion
   target.
6. For a branch-based lane, an administrator performs the lease-protected hard
   swap after review. The isolated lock then names the stable branch at the same
   commit before the Marin PR becomes ready and merges. GPU vLLM also promotes
   its staged source branch after its released-wheel evidence is reviewed.

Current promotion is this administrator action. The GitHub App has the needed
bypass capability, but no production workflow owns automatic stable-branch
promotion. [Issue #8323 tracks the planned automation](https://github.com/marin-community/marin/issues/8323#issuecomment-5465358406),
including lease protection and exact-SHA readback. Until that work lands, a
merged Marin pin PR does not move a fork branch automatically.

## When to create a fork

Create a fork when Marin needs an unmerged upstream patch, upstream's release
cadence cannot deliver a required fix, or Marin needs a custom artifact such as
a TPU or GPU wheel. Prefer upstreaming each retained patch. Record why it is
kept and the condition for dropping it so that later refreshes can remove work
that upstream has absorbed.
