# Executor API

!!! note "Executor Framework Moved to Thalas"
    The executor framework has been factored out into a separate package called [Thalas](https://github.com/marin-community/thalas).

    For complete API documentation, please see the [Thalas Executor API Reference](https://github.com/marin-community/thalas/blob/main/docs/references/executor-api.md).

## Migration Guide

Update your imports from `marin.execution` to `thalas`:

```python
# Old imports (no longer work)
from marin.execution import executor_main, ExecutorStep

# New imports
from thalas import executor_main, ExecutorStep
```

All other aspects of the API remain unchanged.
