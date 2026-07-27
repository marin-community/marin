# EP64 multi-process telltale startup

The EP64 training task must start four supervised JAX processes on each of 16
GB200 nodes without intermittent local HTTP-port collisions.

## Initial status

The 64-process job repeatedly failed in `iris.runtime.telltale.start` before
model construction. Multiple local ranks selected the same ephemeral port
between `find_free_port()` releasing its probe socket and uvicorn binding it.

## Hypothesis 1

Concurrent supervised ranks race because the kernel can return the same free
ephemeral port after each probe socket closes.

## Changes to make

Select from a disjoint 1,000-port range for each local-device partition. Keep
the existing kernel-selected ephemeral behavior for ordinary single-process
tasks.

## Results

The next 64-process EP64 launch initialized all four supervised processes on
each of 16 nodes and completed five training steps. Later profile retries failed
in JAX coordinator discovery, after telltale startup had completed; that retry
race is separate from the local HTTP-port collision.

## Future work

- [ ] Replace probe-then-bind with passing a bound socket into the server
      lifecycle if Iris needs to support multiple supervised processes sharing
      one local-device partition.
