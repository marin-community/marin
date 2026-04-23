# Hardware Resource Configuration

Marin uses Fray for scheduling and resource management (dispatching to Iris on shared clusters, or to a local backend for laptop runs). The `fray` library provides unified resource configuration types that translate to concrete cluster resource requests.

## ResourceConfig

The main entry point for resource configuration. Use the static factory methods to create configurations:

```python
from fray.cluster import ResourceConfig

# TPU configuration
tpu_config = ResourceConfig.with_tpu("v4-8")
tpu_multislice = ResourceConfig.with_tpu("v4-8", slice_count=2)

# GPU configuration
gpu_config = ResourceConfig.with_gpu("H100", count=8)
gpu_auto = ResourceConfig.with_gpu()  # auto-detect GPU type

# CPU-only configuration
cpu_config = ResourceConfig.with_cpu()
```

::: fray.v2.types.ResourceConfig

## Device Configurations

These are the underlying device types wrapped by `ResourceConfig`:

### CPU

::: fray.v2.types.CpuConfig

### GPU

::: fray.v2.types.GpuConfig

### TPU

::: fray.v2.types.TpuConfig
