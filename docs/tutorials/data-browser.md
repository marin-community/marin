# Data Browser

Marin comes with a data browser that makes it easy to
view datasets (in various formats) and experiments produced by the executor.
It is basically a file browser that handles popular file formats like jsonl and parquet.

## Prerequisites

- Basic [installation](installation.md)
- Run an experiment, either [First Experiment](first-experiment.md) or [Executor 101](executor-101.md).

## Configuration Files

The data browser uses configuration files to specify what paths are accessible:

- `conf/local.conf` - For browsing local files (e.g., `../local_store`)
- `conf/gcp.conf` - For browsing GCP Storage buckets (requires authentication)
- `conf/docker.conf` - For Docker deployment

## Installation

Install dependencies:

```bash
cd data_browser
uv sync
npm install
```

**Note**: If you get `ModuleNotFoundError` when running the server, ensure dependencies are installed or run via uv:

```bash
DEV=true uv run server.py --config conf/local.conf
```

## Development Setup

The data browser consists of two components that need to run simultaneously:

1. **Backend server (Flask)** - Handles file access and API endpoints
2. **Frontend server (React)** - Provides the web interface

### Option 1: One-Command Dev Loop (Recommended)

Use the helper launcher to start both servers together:

```bash
cd data_browser
uv run python run-dev.py --config conf/local.conf
```

This script sets `DEV=true`, starts the Flask backend on the `port` defined in your config (defaults to `5000`; `conf/local.conf` uses `5050`), and runs `npm start` for the React app on port 3000 (or the next free port if 3000 is busy). The React dev server automatically opens your browser window, and it proxies every `/api/...` call to the backend port from your config, so you normally do all browsing at `http://localhost:3000` regardless of the backend port. Press `Ctrl+C` once to stop both services. Pass `--backend-only` if you only need the API, or `--config conf/gcp.conf` to point at a different dataset configuration.

### Option 2: Manual Control

If you prefer to manage the processes yourself, run them in separate terminals:

**Terminal 1: Backend Server**
```bash
cd data_browser
DEV=true uv run python server.py --config conf/local.conf
```

**Terminal 2: Frontend Server**
```bash
cd data_browser
npm start
```

### Option 3: API-Only Testing

When you only need the REST API (or the React server won't start), launch just the backend:

```bash
cd data_browser
DEV=true uv run python server.py --config conf/local.conf
```

**Access**: `http://localhost:<port>/api/view?path=local_store` (replace `<port>` with the value from your config; `5050` for `conf/local.conf`).

## Configuration Details

### Local Development (`conf/local.conf`)
```yaml
root_paths:
- ../local_store
port: 5050
```

### GCP Storage (`conf/gcp.conf`)
```yaml
root_paths:
- gs://marin-us-central2
- gs://marin-us-west4
# ... other buckets
blocked_paths:  # Optional: paths to block access to
- gs://marin-us-central2/private-data/
max_lines: 100
max_size: 10000000
```

**Note**: GCP configuration requires valid Google Cloud credentials (service account or gcloud auth).

## Troubleshooting

### React Server Won't Start
If you get "Connection refused" errors, the React dev server may not be running properly. You can still:

1. **Use API directly**: Access `http://localhost:<port>/api/view?path=YOUR_PATH` (match the config's port)
2. **Check React server**: Ensure `npm start` is running without errors
3. **Port conflicts**: Check if port 3000 is available

### Permission Errors
- For GCP buckets: Ensure you have valid Google Cloud credentials

## API Endpoints

- `GET /api/config` - Returns server configuration
- `GET /api/view?path=PATH&offset=0&count=5` - Browse files and directories
- `GET /api/download?path=PATH` - Download files
