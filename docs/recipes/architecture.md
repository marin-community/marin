# Marin Architecture

Marin is a framework for building reproducible language model training pipelines. At its core, Marin executes DAGs of steps using Ray for distributed processing, with automatic versioning based on code and configuration. Pipeline: data curation → transformation → tokenization → training → evaluation.

## Core Architecture

**Executor Pattern**: Experiments are DAGs of `ExecutorStep` objects (`lib/marin/src/marin/execution/executor.py`). Output path = `<base>/<name>-<hash>` where hash covers versioned fields and dependencies. Only changed steps re-run.

**Ray Distribution**: Steps can be normal or `@ray.remote` functions. Ray ships code to workers with step-specific dependency groups from `pyproject.toml`.

**Entry Point**: `executor_main()` or [`lib/marin/src/marin/run/ray_run.py`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/run/ray_run.py) for cluster execution.

## Repository Structure

```
marin/
├── pyproject.toml              # Dependencies, extras (gpu, tpu, cpu, rl, eval)
├── README.md, CLAUDE.md, AGENTS.md
│
├── lib/marin/src/marin/                  # Core library organized by function
│   ├── execution/              # DAG executor (executor.py, status_actor.py)
│   ├── run/                    # Job launchers (ray_run.py, slurm_run.py)
│   ├── download/               # Dataset downloaders (huggingface/, ar5iv/, wikipedia/, nemotron_cc/, filesystem/)
│   ├── transform/              # Raw data → text (ar5iv/, stackexchange/, wikipedia/, conversation/, domain-specific)
│   ├── crawl/                  # Web crawling (fetch_links.py, minhash/, fineweb_edu/, open_web_math/)
│   ├── processing/             # Data processing (tokenize/, classification/, open_web_math/, pubmed/, wikipedia/)
│   ├── classifiers/            # Train quality classifiers (fasttext/, bert/, hf/, custom/)
│   ├── training/               # Model training (training.py, scaling_laws.py)
│   ├── rl/                     # Async PPO (rollout_worker.py, train_worker.py, replay_buffer.py, curriculum.py, environments/, weight_transfer/)
│   ├── evaluation/             # Evaluation (evaluators/, visualize.py)
│   ├── datashop/               # LLM-based filtering (pipeline.py, dataset_processor.py, templates.py)
│   ├── generation/             # LLM inference (inference.py, llm_generation.py, pipeline.py)
│   ├── markdown/               # HTML → markdown (markdown.py, guess_code.py)
│   ├── core/                   # Data types (data.py, conversation.py)
│   └── utilities/, validation/, cluster/, infra/, speedrun/, scaling_laws/, schemas/, web/
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
│   └── speedrun/, posttrain/, dclm/, dolma/, pretraining_datasets/, multilingual_fineweb2_hq/, metrics/
│
├── tests/                      # Test suite
│   ├── integration_test.py     # Full pipeline smoke test (<10min, no GPU)
│   ├── processing/, rl/, evals/, deduplication/, data-curation/, snapshots/, tpu/, vllm/
│   └── quickstart-data/
│
├── docs/                       # Documentation (tutorials/, explanations/, references/, recipes/, reports/, design/, dev-guide/, model-cards/)
├── infra/                      # Ray cluster configs (marin-*.yaml, configure_gcp_registry.py)
├── scripts/                    # Utilities (ray/, training/, pm/, debug/, gpu_eval/)
├── docker/                     # Docker configs (marin/, levanter/)
└── data_browser/               # Web UI (server.py, src/, conf/)
```

## Key Concepts

**Steps and Versioning** (`lib/marin/src/marin/execution/executor.py`): Output path = `<base>/<name>-<hash>` where hash covers versioned fields + dependencies. Re-running with same config = no-op.

**Pipeline Stages**:
1. **Download** (`lib/marin/src/marin/download/`): Fetch datasets from HF Hub, S3, Wikipedia, arXiv
2. **Transform** (`lib/marin/src/marin/transform/`): Raw formats → text/markdown
3. **Quality Filtering** ([`lib/marin/src/marin/processing/classification/`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/processing/classification/)): Train classifiers for filtering
4. **Tokenize** (`lib/marin/src/marin/processing/tokenize/`): Text → tokens (sentencepiece/tiktoken)
5. **Train** (`lib/marin/src/marin/training/`): Levanter (JAX) on TPU/GPU
6. **Evaluate** (`lib/marin/src/marin/evaluation/`): lm-eval-harness or vLLM

**Cluster Infrastructure** (`infra/README.md`): Ray on GCP, on-demand head + preemptible TPU workers (v4/v5e/v6e), autoscaling 4-1024 workers, managed via `scripts/ray/cluster.py`

**Default Helpers** (`experiments/defaults.py`): `default_download()`, `default_tokenize()`, `default_train()`, `default_eval()`

## Quick Reference

**Getting Started**: See `README.md` for installation and getting started guides.

**For Agents**: [Add Dataset](add_dataset.md) • [Fix Issue](fix_issue.md) • See `AGENTS.md` in repository root

**Core APIs**: See `lib/marin/src/marin/execution/executor.py` for executor API, `experiments/defaults.py` for default steps

**Infrastructure**: See `infra/README.md` for cluster setup and infrastructure overview
