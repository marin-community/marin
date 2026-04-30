---
name: add-dataset
description: Add a dataset to the datakit pipeline (lib/marin/src/marin/datakit/download/) and register it in sources.py. Use when given a pointer to a dataset (HuggingFace repo, URL, etc.) and asked to land a PR that wires it into datakit.
---

# Skill: Add a Dataset to Datakit

Datakit is the canonical `download -> (...) -> tokenize` pipeline (see `docs/design/2355_datakit.md`). This skill walks from a single pointer (typically an HF repo id) to a merge-ready PR that:

1. Adds a download/transform module under `lib/marin/src/marin/datakit/download/<name>.py`.
2. Registers the source in `lib/marin/src/marin/datakit/sources.py` with a token count.

**Cargo-cult an existing module rather than inventing a new shape.**

## 0. Inputs you must have before starting

If any of these is missing, **ask the user — do not guess**:

- **Pointer**: HF dataset id (e.g. `GAIR/daVinci-Dev`) or a non-HF URL.
- **Pinned revision**: 7-character HF commit SHA from the dataset's commit history. No `main`, no long SHAs, no branch names.
- **Marin name**: stable lowercase slug (e.g. `coderforge`, `davinci-dev/ctx-native`, `nsf_awards`). Slash-separated when there are sibling subsets.
- **Rough token count in billions** (Llama-3 tokenizer). The registry asserts it; downstream mixing weights it. Acceptable: a real count, an estimate from `bytes / 4`, or a placeholder explicitly flagged as such in the PR body. **Don't fabricate precision.**

Optional but useful:
- Whether the dataset is HF-gated (callers will need `HF_TOKEN`).
- Existing source-id field name (preserved as `source_id` post-normalize).

## 1. Inspect the dataset

Use the schema tool. Picking the wrong text field ships garbage downstream — this is the highest-leverage step.

```sh
uv run lib/marin/tools/get_hf_dataset_schema.py <hf_dataset_id>
```

The tool returns JSON with `splits`, `features`, `text_field_candidates`, and a `sample_row`. If the dataset has multiple configs the response is instead `{"error": "...", "available_configs": [...]}`; pick one and re-run with `--config_name <config>`. Datasets with custom loader scripts need `--trust_remote_code`. Gated datasets need `HF_TOKEN=... uv run ...`.

Text field selection priority: a field literally named `text` → fields with `text` in the name → string-typed fields. Always sanity-check against `sample_row` before committing to one — a `caption` field can be misleading if `body` is what you actually want.

Also capture:
- File layout on HF (top-level dirs, parquet/jsonl, file naming) — needed for `hf_urls_glob`.
- Whether each row is one document or rows must be joined (per-page books, multi-message trajectories).
- Structured columns that need rendering (when there's no scalar text field).
- Any existing meaningful row id (then pass `id_field=` to `normalize_step` so it's kept as `source_id`).

Pin the HF revision: open the dataset on HF Hub, click the commit history, copy the **7-char short SHA** of the commit you want to lock to.

## 2. Pick the pattern

### Pattern A — Plain HF parquet, no transform (simplest)

Use when raw HF parquet already has a scalar text column and one row = one document. The whole module is a thin wrapper around `hf_normalize_steps`:

```python
from marin.datakit.download.hf_simple_util import hf_normalize_steps

def my_dataset_normalize_steps() -> tuple[StepSpec, StepSpec]:
    return hf_normalize_steps(
        marin_name="my-dataset",
        hf_dataset_id="<org>/<repo>",
        revision="<7-char SHA>",
        # text_field="text",  id_field="id",  file_extensions=(".parquet",)
    )
```

Or, for a *family* of one-repo-per-subset (no shared download), build a table and project rows — see `common_pile.py` (27 entries) and `finepdfs.py` (19 language subsets).

### Pattern B — Download + custom transform + normalize (most common)

Use when raw rows must be rendered/joined into a single text doc:
- Multi-message rollouts (`swe_rebench_openhands.py`, `coderforge.py`, `superior_reasoning.py`, `davinci_dev.py`, `gpt_oss_rollouts.py`).
- Per-page books (`institutional_books.py`).
- Structured rows requiring Markdown templating (`davinci_dev.py` ctx-native).

Three-step chain: `download → processed → normalized`. See §3 for the canonical skeleton.

### Pattern C — Multi-subset family with shared download

Use when one HF repo has multiple disjoint subsets and you want each registered separately, sharing one expensive download. Cargo-cult `nemotron_v2.py` (the family download is `@cache`'d so all subsets reuse one staging) and the `_rows_nemotron(...)` projection in `sources.py`.

### Pattern D — Non-HF source

Today only `nsf_awards.py`. Build a `download_<name>_step` whose `fn` runs a Zephyr pipeline that fetches from the source API and writes parquet directly. Then a normal `normalize_step` over that. Use this only if there's no HF mirror.

## 3. Write the module

Place it at `lib/marin/src/marin/datakit/download/<name>.py`. The Pattern B skeleton:

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""<HF_DATASET_ID> download and transform.

<one-paragraph description of what each row looks like and how text is constructed>.
"""

import hashlib

from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext, counters, load_parquet

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "<org>/<repo>"
HF_REVISION = "<7-char SHA>"


def row_to_doc(row: dict) -> list[dict]:
    """Render one row into zero or one normalized records.

    Drop with a counter rather than raising on missing fields — partial
    upstream rows are common and we want the pipeline to keep going.
    """
    text = ...  # extract / render
    if not text:
        counters.increment("<name>/dropped")
        return []
    counters.increment("<name>/kept")
    return [
        {
            "id": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text": text,
            "source": HF_DATASET_ID,
        }
    ]


def transform(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(f"{input_path}/**/*.parquet")
        .flat_map(load_parquet)
        .flat_map(row_to_doc)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="<name>-transform", resources=ResourceConfig(cpu=1, ram="8g"))
    ctx.execute(pipeline)


def download_<name>_step() -> StepSpec:
    dl = download_hf_step(
        "raw/<name>",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        # hf_urls_glob=[...],  # only when restricting files
    )
    return StepSpec(
        name="processed/<name>",
        deps=[dl],
        fn=lambda output_path: transform(input_path=dl.output_path, output_path=output_path),
        hash_attrs={"version": "v1"},
    )


def <name>_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the full ``(download+transform, normalize)`` chain."""
    processed = download_<name>_step()
    return (
        processed,
        normalize_step(name="normalized/<name>", download=processed),
    )
```

### Common variations
- **JSONL inputs**: `from zephyr.readers import load_jsonl`; if files are at known paths use `Dataset.from_list([f"{input_path}/foo.jsonl", ...])`. See `gpt_oss_rollouts.py`, `superior_reasoning.py`.
- **Parquet with deeply nested structs** (multi-message rollouts): default 16 GiB workers OOM during full-row materialization. Use `load_parquet_batched` from `rollout_transforms.py` and bump `worker_resources` on `normalize_step` (see `davinci_dev.py` env-native at 64 GiB).
- **Gated HF dataset**: no code change — the module docstring should call it out, and callers must `export HF_TOKEN=...` before submitting. The HF auth path fails fast on 401/403 (see `lib/marin/src/marin/datakit/download/huggingface.py:_hf_auth_error`).
- **Existing meaningful source ID**: pass `id_field="<existing_field>"` to `normalize_step`. Normalize *always* recomputes the canonical `id` from `text` via xxh3_128; the upstream id becomes `source_id`. See `nsf_awards.py`. The sha256 inside `row_to_doc` is purely a placeholder so the intermediate parquet has *some* id — normalize overwrites it.
- **Multiple disjoint subsets of one repo** (e.g. ctx-native vs env-native): emit two `<name>_<subset>_normalize_steps` functions sharing the HF download via separate `hf_urls_glob` selectors. See `davinci_dev.py`.
- **Restricting files**: prefer `hf_urls_glob=[...]` over downloading the full repo. Cuts both staging time and storage cost.

### Required output schema for `row_to_doc`
- `id`: any string — overwritten by normalize.
- `text`: the document text (UTF-8 string). Normalize filters empty/whitespace-only out.
- `source`: a constant string identifying the dataset (downstream provenance / topic-by-source consumers rely on this — don't omit it).
- Anything else you want preserved is fine; normalize keeps all extra columns.

## 4. Register in `sources.py`

Open `lib/marin/src/marin/datakit/sources.py`.

1. Import the chain factory in alphabetical order with the other download imports:
   ```python
   from marin.datakit.download.<name> import <name>_normalize_steps
   ```

2. Add a row to `single_sources` (alphabetical) inside `all_sources()`:
   ```python
   ("<marin_name>", <name>_normalize_steps, <rough_token_count_b>),
   ```

3. For Pattern A *families* (one repo per subset): add a `_rows_flat(<factory>, {marin_name: count, ...})` block.

4. For Pattern C Nemotron-shaped families (shared `@cache`'d download): add a `_rows_nemotron(library_family, registry_family, {...})` block.

5. Append the new tuple to the `all_rows` tuple at the bottom of `all_sources()` if you added a new family.

**Do not** put new sources in the `# ---- Disabled sources ----` TODO block above `all_sources()`. That block is for sources that still cannot ferry; it is read by no one.

### Token count
- In billions of Llama-3 tokens.
- Sources, in order of preference: an authoritative count from the dataset card (verify the tokenizer family), Marin's own token-count-viewer, a `bytes / 4` estimate from raw text, and as a last resort a placeholder.
- If estimating, say so plainly in the PR body — don't pad decimals to imply precision the count doesn't have.

## 5. Validate locally

Cheapest end-to-end check that your wiring is correct (no GCS, no Iris, no actual download):

```sh
uv run python -c "
from marin.datakit.sources import all_sources
s = all_sources()['<marin_name>']
print(s.name, s.rough_token_count_b)
for step in s.normalize_steps:
    print(' ', step.name, step.hash_id[:12])
"
```

This rebuilds the dataclass, exercises your factory, hits the duplicate-name assertion in `all_sources()`, and proves the module imports cleanly.

For Pattern B, also run lint/format on touched files:

```sh
./infra/pre-commit.py --files \
    lib/marin/src/marin/datakit/download/<name>.py \
    lib/marin/src/marin/datakit/sources.py \
    --fix
```

If `row_to_doc` does any non-trivial rendering, add a unit test under `tests/datakit/download/test_<name>.py`. Pattern: pure function, hand-built input dict, assert text/counters. **No network, no real GCS** — see `tests/datakit/download/test_npm_registry_metadata.py` for an in-process HTTP-server fixture, or `conftest.py` for the shared mock fixtures.

A live ferry / actual download is a separate, follow-up step — usually owned by whoever runs the canonical run, not the PR. Don't block the PR on it.

## 6. Open the PR

Use the `pull-request` skill to draft and push (or `commit` then `gh pr create`).

PR description should cover:
- HF id and pinned revision.
- What each row is and how `text` is rendered (one or two sentences).
- Token count source: real / estimated / placeholder.
- Auth requirements (gated dataset → `HF_TOKEN` needed).
- Any non-default resources (e.g. 64 GiB workers for nested-struct rows).

Reference past PRs for tone and structure:
- #5252 — daVinci-Dev (multi-subset, Markdown rendering, gated)
- #5193 — manifest-backed game/music raw eval slices
- #5126 — UWF Zeek security eval slice
- #4516 — NSF Grant Abstracts (non-HF source)

## Common mistakes to avoid

- **Pinning `revision="main"` or a 40-char SHA.** HF list APIs accept the 7-char short SHA; long SHAs and branch names break the cache invariants and reproducibility.
- **Inventing a token count.** The `DatakitSource` dataclass requires it (no default). Either find a real number, estimate transparently from `bytes / 4`, or label it as a placeholder in the PR body.
- **Skipping `source` in the rendered record.** Provenance and topic-by-source code reads it.
- **Putting the source under the disabled-TODO block** in `sources.py`. New sources go in `all_sources()`'s `single_sources` (or a family block).
- **Reusing one HF repo across two `download_hf_step(name=...)` instances without sharing.** Pick one: `@cache` the factory (Nemotron pattern) or `override_output_path=` to share staging. Otherwise you pay the download twice and the staged dirs collide on hash.
- **Putting the canonical id in `row_to_doc`.** Normalize always overwrites `id` with `xxh3_128(text)`. If you need to preserve an upstream id, pass `id_field=` to `normalize_step` instead.
- **Globbing the whole repo when you only want a few files.** Use `hf_urls_glob` to restrict; staging is expensive.

## See also

- Design: `docs/design/2355_datakit.md`
- Testbed: `docs/design/datakit_testbed.md`
- Source registry: `lib/marin/src/marin/datakit/sources.py`
- Schema tool: `lib/marin/tools/get_hf_dataset_schema.py`
- Existing modules: `lib/marin/src/marin/datakit/download/`
- Helpers: `huggingface.py` (`download_hf_step`), `hf_simple_util.py` (`hf_normalize_steps`), `rollout_transforms.py` (`load_parquet_batched`, `strip_think_tags`)
- Normalize: `lib/marin/src/marin/datakit/normalize.py` (`normalize_step`, `normalize_to_parquet`, `DedupMode`)
- Tracking issue: marin#4272
