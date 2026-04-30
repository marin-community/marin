# Zephyr Performance Improvements and Monitoring

Zephyr is fast enough to do our current data processing, but it takes ~10 hours to do a standard run. If we can make it an order of magnitude faster, we can enable much better experimentation and iteration.

We believe there is a lot of low-hanging fruit to speed up Zephyr (e.g. better parallelization, pipeline stages, etc.) but we don't have an easy way to pull data to test our theories or figure out other optimization strategies. We already capture cpu profiles for all Iris jobs every five minutes--we just need to analyze them.

We should make the low-hanging fruit optimizations while building the tooling to validate that it's working. That tooling will help us define the next round of improvements, should we need them.

## Goals
* At least a 10x performance improvement in Zephyr
	* TBD: pick a job (or jobs) to benchmark
* A workflow that anyone can run manually to understand the performance of their Zephyr job
	* This workflow can provide a way to determine how much time we're spending in Python and whether using something like Polars or Rust to speed things up further is warranted

## Design

### Low-hanging fruit
It feels more valuable to track these individual improvements (and future ones) in Github Issues under an Epic, so below is just a quick summary.

**Better Parallelization**
We can allow the system to parallelize the workload better by splitting input files (for splittable inputs like Parquet) so that we can run more shards (and therefore more workers) at a time

**Better Resource Utilization**
We currently use the same, over-provisioned, workers for every stage. We can do two things to use resources better,
1. Use multithreading for CPU-heavy stages to take better advantage of the available cores while sharing some of the fix-cost of large libraries in RAM
2. Use different worker-types with different resource allocations for different stages

**Pipelining Stages**
Today, each stage is a hard gate, requiring all shards to finish before starting the next stage. We can start the following stage when it's inputs are ready instead of waiting for all inputs to be ready.

We can also consider doing more stage fusion across stages, which might achieve the same goal.
### Monitoring & analysis workflow
We'll use a modified chain-of-thought approach where we use `lib/iris/scripts/job_profile_summary.py` to do a first-pass summarization of the CPU profiles for a job and then provide the output to an LLM to summarize and make recommendations.

`job_profile_summary.py` provides a first pass at how to pull the stats from a snapshot of the Iris database and summarize them. We can extend it using the recommendations in the [research doc](research.md), specifically:
* Figure out which snapshot to pull since iris prunes task stats
* Handle outliers, specifically have it output workers that have too many samples (maybe just a straggler) and a distribution of sample counts (to make sure one slow worker doesn't dominate)
* Retain enough source information for the LLM to do research in the codebase to determine fixes.
* Ensure output is agent-friendly or add an `--llm-summary` output parameter (top-50 stacks by sample count with percentages, normalized, under ~4,000 tokens)

Then build a `SKILL.md` to have the LLM analyze the output from `job_profile_summary.py`. It should:
* Classify the bottleneck type (CPU-bound in Python / CPU-bound in library code / GC / slow worker or data skew / coordinator overhead)
* Identify impact ceiling for fixing the bottleneck
* Produce structured recommendations with a confidence level and the preconditions each recommendation requires, and
* Identify gaps (e.g. CPU profile is insufficient) or anything else that it cannot be sure of without further information (suggesting changes to `job_profile_summary.py` if required)
## Challenges / Risks / Questions
* Increasing parallelism may cause stability issues with large number of shards / workers
* Parquet splitting requires reading file footers at plan time (one GCS roundtrip per file); for jobs with thousands of input files this adds latency to pipeline startup
* Stage pipelining only helps non-shuffle stages; Scatter→Reduce boundaries must remain hard barriers because reducers need output from all mappers
* The GIL could hinder multithreading
* Making sure we're not chasing incorrect recommendations / one-offs with an efficient validation step
## Open Questions
* What job(s) to use as a benchmark?
## Non Goals
* Tracking performance of specific workflows over time (e.g. [Issue #2122](https://github.com/marin-community/marin/issues/2122)).
* Automating the workflow so it runs on every Zephyr job or automatically generates tickets. Let's prove it's useful first, then we can decide whether to automate it.
## Future work
* We should be able look for memory leaks by applying a similar technique to the memory profiles Iris saves
* We should be able to, pretty easily, extend this to other jobs running on Iris
