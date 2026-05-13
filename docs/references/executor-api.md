# Executor API

This is the API documentation for the executor framework.

## Executor Entrypoint

::: marin.execution.executor_main
::: marin.execution.ExecutorMainConfig

## Executor and Steps

::: marin.execution.Executor
::: marin.execution.ExecutorStep

## Inputs and Outputs

::: marin.execution.InputName
::: marin.execution.output_path_of
::: marin.execution.this_output_path
::: marin.execution.OutputName
::: marin.execution.THIS_OUTPUT_PATH


## Versioning

::: marin.execution.VersionedValue
::: marin.execution.versioned
::: marin.execution.ensure_versioned
::: marin.execution.unwrap_versioned_value
::: marin.execution.get_executor_step

## Config Walking and Materialization

Helpers for inspecting and resolving placeholder-bearing configs without
running a full executor pipeline. `compute_output_path` and
`resolve_local_placeholders` run on the submitter; `materialize` is the
worker-side counterpart that submits any embedded `ExecutorStep`s and
substitutes resolved paths into the config.

::: marin.execution.walk_config
::: marin.execution.resolve_local_placeholders
::: marin.execution.compute_output_path
::: marin.execution.materialize
