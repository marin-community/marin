# fray-rusty

A Rust-based implementation of distributed task coordination for the Fray framework.

This package provides high-performance components for task distribution and execution:

- **Coordinator**: Central task distribution service using Cap'n Proto RPC
- **Worker**: Task execution client that connects to the coordinator
- **Python Extension**: Python bindings for coordinator and worker functionality

## Components

- Cap'n Proto schema for efficient RPC communication
- Async runtime based on Tokio
- Python integration via PyO3
- Optimized for low-latency task distribution

## Building

This package uses maturin for building the Python extension:

```bash
uv run maturin develop
```

## Testing

```bash
uv run pytest
```
