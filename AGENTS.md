# AGENTS.md

## Guidelines for Coding Agents in Marin

This document provides a starting point for using coding agents (AI or human) in the Marin project.

- Begin with the agent-friendly recipes in `docs/recipes/`.
- The first step for dataset addition is schema inspection. See the [add_dataset.md](docs/recipes/add_dataset.md) recipe for details.
- You can help organize experiments using the [organize_experiments.md](docs/recipes/organize_experiments.md) recipe.
- Follow the rules and examples in each recipe to ensure compatibility and automation-friendliness.

## Coding Guidelines

- Always fix tests if you broke them.
- DO NOT fix tests by relaxing tolerances or hacking around them.
- NEVER SAY You're absolutely right!
- You never credit yourself in commits
- Always use `uv run` in python projects instead of `python`

- Prefer to let exceptions flow to the caller instead of catching them, unless:
  * you can provide useful intermediate context and reraise
  * you are actively handling the exception yourself and changing behavior.

- You prefer top-level functions vs methods when writing code that doesn't actively mutate state - aka functional style.
- Use classes and methods when appropriate to hide data
- Prefer top-level Python functions & fixtures for tests.
- Prefer early-exit (if not x: return None) when it will reduce nesting significantly.

- You never introduce hacks like `hasattr(m, "old_attr")`, you instead update code to have a consistent pattern. The only exception is if you are explicitly asked to update usage of a 3rd party library and are explicitly asked to add backwards compatibility for older versions of that dependency

## Deprecation

- Unless specifically requested by the user, you do _not_ introduce deprecation or fallback paths for code. You always update all usages of the code instead.

## Comments

You write detailed comments when appropriate to describe code behavior as a
whole, e.g. at the module or class level, or when describing some subtle
behavior.

You don't generate comments with obviously reflect the code, e.g.

<bad>
     # Use in-memory rollout queue
    rollout_queue = InMemoryRolloutQueue()
</bad>

<good>
# We have found that each instance of a FlightServer can provide approximately 1GB/s
# of throughput. As our typical VMs run with 200Gbps NICs, running 16 parallel servers
# should be sufficient to saturate the network.
</good>

## Planning
- When planning, you produce detailed plans including code snippets
- You ask questions up front when building a plan instead of guessing.

## Testing
- You always run the appropriate tests for your changes in e.g. the tests/ directory
- You use pytest features like fixtures & parameterization to avoid duplication and write clean code

> This file will be expanded as agent workflows and best practices evolve.
