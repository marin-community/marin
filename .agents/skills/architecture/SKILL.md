---
name: architecture
description: Marin architecture overview and repository structure reference. Use when you need to understand how Marin works, find a module, or orient yourself in the codebase.
---

# Skill: Marin Architecture

Marin is a framework for building reproducible language model training pipelines. At its core, Marin executes DAGs of steps using [Fray](https://github.com/marin-community/marin/tree/main/lib/fray) (dispatched onto [Iris](https://github.com/marin-community/marin/blob/main/lib/iris/OPS.md) on shared clusters) for distributed processing, with automatic versioning based on code and configuration. Pipeline: data curation → transformation → tokenization → training → evaluation.

## Core Architecture

**Executor Pattern**: Experiments are DAGs of `ExecutorStep` objects (`lib/marin/src/marin/execution/executor.py`). Output path = `<base>/<name>-<hash>` where hash covers versioned fields and dependencies. Only changed steps re-run.

**Fray/Iris Distribution**: Steps that need remote execution wrap their function with `remote()` (see `experiments/defaults.py`). Fray launches each remote step as a sub-job against the current cluster (Iris on shared infra, Local for laptop runs). Step-specific dependency groups are drawn from `pyproject.toml`.

**Entry Point**: Call `executor_main()` at the bottom of the script; launch the script itself as a CPU-only Iris job (`uv run iris --cluster=marin job run -- python -m experiments.<script>`) for cluster execution. See [`lib/iris/OPS.md`](https://github.com/marin-community/marin/blob/main/lib/iris/OPS.md) for the full launch reference.

## Repository Structure

```
marin/
├── pyproject.toml              # Dependencies, extras (gpu, tpu, cpu, rl, eval)
├── README.md, CLAUDE.md, AGENTS.md
│
├── lib/marin/src/marin/                  # Core library organized by function
│   ├── execution/              # DAG executor (executor.py, status_actor.py)
│   ├── run/                    # Legacy launcher stubs (slurm_run.py); submit via `iris job run` on shared clusters
│   ├── download/               # Dataset downloaders (huggingface/, ar5iv/, wikipedia/, nemotron_cc/, filesystem/)
│   ├── transform/              # Raw data → text (ar5iv/, stackexchange/, wikipedia/, conversation/, domain-specific)
│   ├── crawl/                  # Web crawling (fetch_links.py, minhash/, fineweb_edu/, open_web_math/)
│   ├── processing/             # Data processing (tokenize/, classification/, open_web_math/, pubmed/, wikipedia/)
│   ├── classifiers/            # Train quality classifiers (fasttext/, bert/, hf/, custom/)
│   ├── training/               # Model training (training.py, scaling_laws.py)
│   ├── rl/                     # Async PPO (rollout_worker.py, train_worker.py, replay_buffer.py, curriculum.py, environments/, weight_transfer/)
│   ├── evaluation/             # Evaluation (evaluators/, visualize.py)
│   ├── generation/             # LLM inference (inference.py, llm_generation.py, pipeline.py)
│   ├── markdown/               # HTML → markdown (markdown.py, guess_code.py)
│   ├── core/                   # Data types (data.py, conversation.py)
│   └── utilities/, validation/, cluster/, infra/, scaling_laws/, schemas/, web/
│
├── experiments/                # Experiment scripts
│   ├── defaults.py             # default_download, default_tokenize, default_train, default_eval
│   ├── models.py, llama.py     # Model configs
│   ├── simple_train_config.py, simple_sft_config.py
│   ├── pretraining_datasets/, midtraining_datasets.py, paloma.py
│   ├── exp*.py                 # Individual experiments (exp<issue_num>_<name>.py)
│   ├── tutorials/              # Tutorial experiments
│   ├── tootsie/                # Tootsie model experiments (8B, 32B, 70B)
│   ├── evals/                  # Evaluation runners (evals.py, task_configs.py, engine_configs.py, run_*.py)
│   └── posttrain/, dclm/, dolma/, pretraining_datasets/, multilingual_fineweb2_hq/, metrics/
│
├── tests/                      # Test suite
│   ├── integration_test.py     # Full pipeline smoke test (<10min, no GPU)
│   ├── processing/, rl/, evals/, deduplication/, data-curation/, snapshots/, tpu/, vllm/
│   └── quickstart-data/
│
├── docs/                       # Documentation (tutorials/, explanations/, references/, recipes/, reports/, design/, dev-guide/, model-cards/)
├── infra/                      # Cluster configs (configure_gcp_registry.py, configure_buckets.py). Iris cluster configs live under lib/iris/examples/.
├── scripts/                    # Utilities (iris/, training/, pm/, debug/, gpu_eval/)
└── docker/                     # Docker configs (marin/, levanter/)
```

## Key Concepts

**Steps and Versioning** (`lib/marin/src/marin/execution/executor.py`): Output path = `<base>/<name>-<hash>` where hash covers versioned fields + dependencies. Re-running with same config = no-op.

**Pipeline Stages**:
1. **Download** (`lib/marin/src/marin/download/`): Fetch datasets from HF Hub, S3, Wikipedia, arXiv
2. **Transform** (`lib/marin/src/marin/transform/`): Raw formats → text/markdown
3. **Quality Filtering** ([`lib/marin/src/marin/processing/classification/`](https://github.com/marin-community/marin/tree/main/lib/marin/src/marin/processing/classification)): Train classifiers for filtering
4. **Tokenize** (`lib/marin/src/marin/processing/tokenize/`): Text → tokens (sentencepiece/tiktoken)
5. **Train** (`lib/marin/src/marin/training/`): Levanter (JAX) on TPU/GPU
6. **Evaluate** (`lib/marin/src/marin/evaluation/`): lm-eval-harness or vLLM

**Cluster Infrastructure** ([`lib/iris/OPS.md`](https://github.com/marin-community/marin/blob/main/lib/iris/OPS.md)): Iris on GCP (TPU v4/v5e/v6e) and CoreWeave (H100 GPUs); on-demand controller + autoscaling preemptible workers. Submit jobs with `uv run iris --cluster=marin job run ...`.

**Default Helpers** (`experiments/defaults.py`): `default_download()`, `default_tokenize()`, `default_train()`, `default_eval()`

## Quick Reference

**Getting Started**: See `README.md` for installation and getting started guides.

**For Agents**: `.agents/skills/add-dataset/` • `.agents/skills/fix-issue/` • See `AGENTS.md` in repository root

**Core APIs**: See `lib/marin/src/marin/execution/executor.py` for executor API, `experiments/defaults.py` for default steps

**Infrastructure**: See `infra/README.md` for cluster setup and infrastructure overview
