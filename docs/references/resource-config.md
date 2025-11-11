# Hardware Resource Configuration

Marin uses Ray for scheduling and resource management. Ray provides a flexible resource model that allows you to specify
the resources that a task requires. In Marin, we specify a few wrapper types for common hardware configurations.


## CPU-Only

::: marin.resources.CpuOnlyConfig

## GPU

::: marin.resources.GpuConfig

## TPU

::: marin.resources.TpuPodConfig
