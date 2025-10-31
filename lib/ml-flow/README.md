# ml-flow

Apache Beam wrapper library for Marin data preprocessing pipelines.

## Overview

`ml-flow` provides a high-level Dataset API for building scalable data pipelines on Google Cloud Dataflow. It's designed to replace Ray-based preprocessing in Marin with Beam's superior auto-scaling and cost optimization.

## Key Features

- **Dataset API**: Fluent interface for reading, transforming, and writing data
- **FlexRS Support**: 40-60% cost savings on batch workloads
- **Auto-scaling**: Throughput-based scaling (1-1000 workers)
- **fsspec Integration**: Compatible with GCS, S3, local filesystems
- **Fault Tolerance**: Built-in retries and error handling

## Quick Start

```python
from ml_flow import Dataset, DataflowOptions

# Configure Dataflow
options = DataflowOptions(
    project="my-project",
    region="us-central1",
    use_flex_rs=True,  # 40-60% cost savings
)

# Read, transform, write
ds = Dataset.from_jsonl_files(
    "gs://input/**/*.jsonl.gz",
    pipeline_options=options.to_pipeline_options()
)
ds = ds.filter(lambda x: x["score"] > 0.5)
ds = ds.map(lambda x: {"text": x["text"].upper()})
ds.write_jsonl_gz("gs://output/data")
ds.run_and_wait()
```

## Local Testing

Use `DirectRunner` for fast local development:

```python
options = DataflowOptions(
    project="test",
    runner="DirectRunner",
)
```

## Installation

From within the Marin repository:

```bash
uv sync --extra beam --prerelease=allow
```

Or standalone:

```bash
cd lib/ml-flow
uv pip install -e .
```

## Development Status

**Phase 0**: Core library implementation (current)
- Basic Dataset API
- JSONL and text file I/O
- Local testing with DirectRunner

**Phase 1+**: Feature expansion
- Batch inference support
- Advanced transformations
- Full migration guides

## Testing

Run tests with:

```bash
cd lib/ml-flow
uv run --with pytest pytest tests/
```

## License

See LICENSE file in the Marin repository root.
