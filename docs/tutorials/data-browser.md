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
uv venv
uv pip install -e .
npm install
```

**Note**: If you get `ModuleNotFoundError` when running the server, ensure dependencies are installed or run via uv:

```bash
DEV=true uv run python server.py --config conf/local.conf
```

## Development Setup

The data browser consists of two components that need to run simultaneously:

1. **Backend server (Flask)** - Handles file access and API endpoints
2. **Frontend server (React)** - Provides the web interface

### Option 1: Full Development Setup (Recommended)

Run both servers in separate terminals:

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

**Access**: [http://localhost:3000](http://localhost:3000)

### Option 2: API-Only Testing

If you only need to test the API endpoints or the React server won't start:

```bash
cd data_browser
DEV=true uv run python server.py --config conf/local.conf
```

**Access**: [http://localhost:5000/api/view?path=../local_store](http://localhost:5000/api/view?path=../local_store)

## Configuration Details

### Local Development (`conf/local.conf`)
```yaml
root_paths:
- ../local_store
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

1. **Use API directly**: Access `http://localhost:5000/api/view?path=YOUR_PATH`
2. **Check React server**: Ensure `npm start` is running without errors
3. **Port conflicts**: Check if port 3000 is available

### Permission Errors
- For GCP buckets: Ensure you have valid Google Cloud credentials

## API Endpoints

- `GET /api/config` - Returns server configuration
- `GET /api/view?path=PATH&offset=0&count=5` - Browse files and directories
- `GET /api/download?path=PATH` - Download files
