# Data Browser

Marin comes with a data browser that makes it easy to
view datasets (in various formats) and experiments produced by the executor.
It is basically a file browser that handles popular file formats like jsonl and parquet.

## Prerequisites

- Basic [installation](installation.md)
- Run an experiment, either [First Experiment](first-experiment.md) or [Executor 101](executor-101.md).

## Using the data browser

Install dependencies:

```bash
cd data_browser
uv venv
uv pip install -e .
npm install
```

The data browser takes a configuration file that specifies the root directory (in our examples, it's been `local_store`,
or `../local_store` if we're in the `data_browser` directory).  This is what's defined in `conf/local.conf`.

The data browser has two pieces:

- Frontend server (React)

- Backend server (Flask)

To start the data browser, we need to run both servers, in different shells:
```bash
DEV=true uv run python server.py --config conf/local.conf
npm start
```

Once the server is started, go to
[http://localhost:3000](http://localhost:3000) and navigate around to look at datasets and experiments.
You can click on **Experiment JSON** to get a nicer view of the experiment (the URL is also
printed out when you run the experiment).
