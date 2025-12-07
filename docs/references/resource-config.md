# Hardware Resource Configuration

Marin uses Ray for scheduling and resource management. Ray provides a flexible resource model that allows you to specify
the resources that a task requires. The `fray` library provides unified resource configuration types.

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

::: fray.cluster.base.ResourceConfig

## Device Configurations

These are the underlying device types wrapped by `ResourceConfig`:

### CPU

::: fray.cluster.base.CpuConfig

### GPU

::: fray.cluster.base.GpuConfig

### TPU

::: fray.cluster.base.TpuConfig
