# Dataset Catalog

## Overview

`experiments/count_tokens.py` maintains a catalog of all pretraining data
sources used in Marin — where they come from, what domain they cover, whether
they're synthetic, and how many tokens they contribute after tokenization.

**Registry:** https://huggingface.co/datasets/marin-community/token-counts
**Viewer:** https://huggingface.co/spaces/marin-community/token-count-viewer

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `marin_name` | string | Short identifier used in Marin pipelines |
| `marin_tokens` | int | Token count after tokenization |
| `category` | string | Content domain: `web`, `code`, `math`, `multilingual`, `specialized` |
| `synthetic` | bool | Whether an LLM generated, rephrased, or translated the text |
| `pdf` | bool | Whether the source material was extracted from PDFs |
| `hf_repo` | string | Source HuggingFace dataset repo (e.g. `nvidia/Nemotron-CC-v2`) |
| `hf_subset` | string | HuggingFace dataset config/subset, if applicable |
| `transform_tldr` | string | What Marin does to the data before tokenization, if non-trivial |

## How the script works

1. For each dataset in the `DATASETS` dict, finds the tokenized directory under
   `gs://marin-us-central1/tokenized/<prefix>-<hash>/train/.stats.json`.
2. Reads token counts from the stats JSON.
3. Builds a CSV with all metadata from the `Dataset` dataclass.
4. Uploads to `marin-community/token-counts` via `huggingface_hub`.

## Running

```bash
# Locally (needs GCS access + HF write token)
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
uv run python experiments/count_tokens.py

# On Iris
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  -e HF_TOKEN ${HF_TOKEN} \
  -- python experiments/count_tokens.py
```

## Adding a new data source

1. Find the dataset on HuggingFace and read the dataset card.
2. Add an entry to the `DATASETS` dict with a `Dataset(...)`:
   - `gcs_prefix`: path under `gs://marin-us-central1/tokenized/`
   - `category`: one of `web`, `code`, `math`, `multilingual`, `specialized`
   - `synthetic`: `True` if LLM-generated or LLM-translated
   - `pdf`: `True` if sourced from PDF extraction
   - `hf_repo`: source HuggingFace repo
   - `hf_subset`: HF config/subset name, if applicable
   - `transform_tldr`: short description of any Marin-side transform
3. Run the script to verify it picks up the stats and the registry updates.

## Category taxonomy

5 content-domain categories, plus orthogonal `synthetic` and `pdf` flags:

| Category | What belongs here |
|---|---|
| `web` | Organic text from the web or curated corpora: Common Crawl (Nemotron CC), Common Pile subsets (academic papers, books, legal, forums, government docs, etc.), PDF-extracted documents |
| `code` | Source code and code-related content, both organic (GitHub, StackV2, PEPs) and synthetic (code QA, reviews, transpilation) |
| `math` | Math-focused content: web extractions, synthetic textbooks, competition problems, formal logic |
| `multilingual` | Translated or parallel text corpora (FineTranslations, Nemotron CC translated variants) |
| `specialized` | Synthetic data that doesn't fit code/math: cross-domain reasoning, wiki rewrites, economics, multiple choice, SFT general |

### How categories were originally determined

Each dataset was categorized by reviewing its HuggingFace dataset card — not
by Marin pipeline naming. Three families were researched:

1. **NVIDIA Nemotron family** — `nvidia/` org on HuggingFace: Nemotron-CC-v2,
   v2.1, CC-Code-v1, CC-Math-v1, Pretraining-Code-v2,
   Pretraining-Specialized-v1/v1.1, Pretraining-SFT-v1.
2. **Common Pile** — `common-pile/` org and arXiv:2506.05209.
3. **Standalone** — `HuggingFaceFW/finepdfs`, `HuggingFaceFW/finetranslations`,
   `AI-MO/NuminaMath-1.5`.

### Key classification decisions

- **`synthetic` is orthogonal to category.** A synthetic code QA dataset is
  `category=code, synthetic=True`. The category is always the content domain.
- **Quality filtering is not synthetic.** Nemotron CC organic subsets are
  filtered by quality classifiers but the text is human-written →
  `synthetic=False`.
- **Translated web text → `multilingual`**, not `web`. The translation is the
  defining characteristic for data mixing.
- **`web` is the catch-all** for organic text that isn't code, math, or
  multilingual. This includes academic papers, books, legal docs, forums, etc.
  from Common Pile.
- **`pdf`** is a format flag, not a category. PDFs span many domains. The flag
  lets you filter by extraction method.

## Maintaining the catalog

### When to update

- A new dataset is added to the Marin tokenization pipeline
- An existing dataset's processing changes (new transform, re-tokenization)
- A new HuggingFace source is integrated

### Where to look for new datasets

- `experiments/` for new experiment scripts with tokenization pipelines
- `lib/marin/src/marin/transform/` for new transform modules
- GCS: `gs://marin-us-central1/tokenized/` for new tokenized output dirs

### Verifying entries

- **HF repos**: check `https://huggingface.co/datasets/<hf_repo>` exists.
  Common orgs: `nvidia/`, `common-pile/`, `HuggingFaceFW/`, `AI-MO/`.
- **Transforms**: search `experiments/` and `lib/marin/src/marin/transform/`
  for processing applied to the dataset. Only document non-trivial transforms
  (filtering, stitching, format conversion) — not standard tokenization.

### Viewer

The Gradio viewer at `marin-community/token-count-viewer` reads directly from
the registry CSV. It updates automatically when the catalog script runs — no
separate deployment needed.
