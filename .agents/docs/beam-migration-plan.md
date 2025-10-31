# Apache Beam/Dataflow Migration Plan for Marin

**Status**: Approved
**Created**: 2025-10-30
**Target Completion**: ~20 weeks from start

---

## Executive Summary

This document outlines the complete plan for migrating Marin's data preprocessing infrastructure from Ray to Apache Beam/Dataflow. The migration creates a new library `lib/ml-flow` providing a Dataset-based API that wraps Apache Beam, enabling massive-scale batch processing with superior auto-scaling and cost optimization.

### Key Decisions
- **Library location**: `lib/ml-flow/` (standalone package)
- **Migration strategy**: Module-by-module (transform → crawl → classification)
- **Training/TPU**: Keep Ray for all training, RL, and TPU orchestration
- **Dataflow features**: Prioritize FlexRS for 40-60% cost savings on batch workloads

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Target Architecture](#target-architecture)
3. [API Design](#api-design)
4. [Migration Patterns](#migration-patterns)
5. [Migration Phases](#migration-phases)
6. [Testing Strategy](#testing-strategy)
7. [Cost & Performance](#cost--performance)

---

## Current State Analysis

### Ray Usage Patterns in Marin

Based on comprehensive codebase analysis, we identified **7 major patterns**:

#### 1. File-Level Batch Processing (Most Common)
**Pattern**: `@ray.remote` + `@cached_or_construct_output` + manual orchestration
**Files**: ~50+ transform scripts, crawl converters
**Example**: `transform/wikipedia/transform_wikipedia.py`, `crawl/common/convert_to_html.py`

```python
@ray.remote(max_calls=1)
@cached_or_construct_output(success_suffix="SUCCESS")
def process_file(input_path, output_path, config):
    # Transform file
    pass

# Manual orchestration with backpressure
pending = []
for file in files:
    while len(pending) >= max_in_flight:
        ready, _ = ray.wait(pending, num_returns=1)
        ray.get(ready[0])
    pending.append(process_file.remote(file, out_file))
```

#### 2. Ray Data Batch Inference
**Pattern**: `ray.data.read_json()` + `map_batches()` with actors
**Files**: `generation/inference.py`, `generation/pipeline.py`

```python
ds = ray.data.read_json(input_path)
ds = ds.map_batches(
    vLLMTextGeneration,
    concurrency=(1, 4),  # Min/max workers
    batch_size=32,
    **ray_resources
)
ds.write_json(output_path)
```

#### 3. Autoscaling Actor Pool
**Pattern**: Custom actor pool with dynamic scaling
**Files**: `processing/classification/autoscaler.py` (426 lines), `processing/classification/inference.py`

```python
class AutoscalingActorPool:
    """
    Three-threaded architecture:
    - Autoscaling monitor: Scale based on queue depth
    - Dispatcher: Dispatch to least-loaded actor
    - Result collector: Handle failures, resubmit
    """
    def _check_and_scale(self):
        utilization = (pending + active) / num_actors
        if utilization > 0.8:
            self._scale_up()
        elif utilization < 0.2:
            self._scale_down()
```

#### 4. Levanter Tokenization
**Pattern**: Wrapped Levanter cache building in Ray remote
**Files**: `processing/tokenize/tokenize.py`

```python
@ray.remote(num_cpus=0.1)
def tokenize(config):
    source = config.get_shard_source("train")
    tokenizer = AutoTokenizer.from_pretrained(...)
    cache = levanter.build_or_load_cache(
        source, batch_tokenizer, cache_path
    )
```

#### 5. Multi-Stage Pipelines
**Pattern**: Ray actors maintaining state across stages
**Files**: `datashop/pipeline.py` (MEDU pipeline)

```python
@ray.remote(max_restarts=-1)
class MEDUPipeline:
    def stage1_generate_descriptions(self):
        self.generated = self.llm.generate(prompts)

    def stage2_merge_hierarchically(self):
        while len(self.generated) > 1:
            # Tree-based merging
```

#### 6. Training/RL Orchestration
**Pattern**: Fixed pools with high retry limits for TPU resilience
**Files**: `rl/rl_job.py`, `training/training.py`

```python
@ray.remote(**tpu_kwargs, max_retries=100)
def train_worker():
    # Training on TPU pod
    pass
```

#### 7. Executor Framework
**Pattern**: DAG-based step execution with Ray
**Files**: `execution/executor.py`

```python
@dataclass
class ExecutorStep:
    name: str
    fn: ray.RemoteFunction
    config: ConfigT

class Executor:
    def run(self, steps):
        # Build DAG, execute with ray.get()
```

---

## Target Architecture

### New Library: `lib/ml-flow`

```
lib/ml-flow/
├── pyproject.toml              # Package definition
├── README.md                   # Usage guide
├── src/ml_flow/
│   ├── __init__.py
│   ├── dataset.py              # Core Dataset API (wraps PCollection)
│   ├── io.py                   # FileProcessor for file-level parallelism
│   ├── inference.py            # ModelInference for batch inference
│   ├── transforms.py           # Common transforms (tokenization, etc)
│   ├── execution.py            # BeamExecutorStep for executor integration
│   ├── options.py              # DataflowOptions configuration
│   └── utils.py                # Utilities (resumption, checkpointing)
└── tests/
    ├── test_dataset.py
    ├── test_io.py
    └── test_inference.py
```

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Marin Data Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────┐         ┌──────────────────────┐   │
│  │   Data Sources      │         │   Training (Ray)     │   │
│  │  - GCS buckets      │         │  - TPU pod mgmt      │   │
│  │  - HuggingFace      │         │  - Levanter          │   │
│  │  - Web crawls       │         │  - RL workers        │   │
│  └──────────┬──────────┘         └──────────────────────┘   │
│             │                                                 │
│             v                                                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │        lib/ml-flow (Apache Beam Wrapper)            │    │
│  │  ┌────────────────────────────────────────────┐     │    │
│  │  │ Dataset API                                 │     │    │
│  │  │  - from_files() / write_files()            │     │    │
│  │  │  - map() / filter() / map_batches()        │     │    │
│  │  │  - reshuffle() / group_by_key()            │     │    │
│  │  └────────────────────────────────────────────┘     │    │
│  │  ┌────────────────────────────────────────────┐     │    │
│  │  │ FileProcessor                               │     │    │
│  │  │  - process_directory() with resumption     │     │    │
│  │  └────────────────────────────────────────────┘     │    │
│  │  ┌────────────────────────────────────────────┐     │    │
│  │  │ ModelInference                              │     │    │
│  │  │  - RunInference with custom handlers       │     │    │
│  │  └────────────────────────────────────────────┘     │    │
│  └─────────────────────────────────────────────────────┘    │
│             │                                                 │
│             v                                                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │        Google Cloud Dataflow                        │    │
│  │  - Auto-scaling workers                             │    │
│  │  - FlexRS for batch (40-60% cost savings)          │    │
│  │  - Built-in fault tolerance                         │    │
│  └─────────────────────────────────────────────────────┘    │
│             │                                                 │
│             v                                                 │
│  ┌─────────────────────┐                                     │
│  │   Processed Data    │                                     │
│  │  - Dolma format     │                                     │
│  │  - Tokenized caches │                                     │
│  │  - Classified docs  │                                     │
│  └─────────────────────┘                                     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## API Design

### 1. Core Dataset API

```python
# lib/ml-flow/src/ml_flow/dataset.py

from typing import TypeVar, Callable, Any, Iterator
from dataclasses import dataclass
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

T = TypeVar('T')
U = TypeVar('U')

@dataclass
class DatasetOptions:
    """Configuration for dataset processing."""
    pipeline_options: PipelineOptions
    temp_location: str
    staging_location: str
    checkpoint_location: str | None = None
    resume: bool = True


class Dataset:
    """
    Dataset API that wraps Apache Beam PCollection.

    Design principles:
    - Lazy evaluation (Beam's DAG building)
    - Automatic checkpointing via Beam's built-in mechanisms
    - File-based I/O patterns (jsonl.gz, parquet)
    - Resumability through fault tolerance

    Example:
        ds = Dataset.from_files("gs://bucket/data/**/*.jsonl.gz")
        ds = ds.filter(lambda x: x["score"] > 0.5)
        ds = ds.map(lambda x: transform(x))
        ds.write_files("gs://bucket/output/")
        ds.run_and_wait()
    """

    def __init__(
        self,
        pcollection: beam.PCollection,
        options: DatasetOptions,
        schema: dict[str, Any] | None = None
    ):
        self._pcollection = pcollection
        self._options = options
        self._schema = schema
        self._pipeline = pcollection.pipeline

    @classmethod
    def from_files(
        cls,
        pattern: str,
        file_format: str = "jsonl.gz",
        options: DatasetOptions = None,
    ) -> "Dataset":
        """Read from file pattern (supports GCS, local, S3)."""
        pass

    def map(self, fn: Callable[[T], U], name: str | None = None) -> "Dataset":
        """Apply function to each element."""
        pass

    def flat_map(self, fn: Callable[[T], Iterator[U]], name: str | None = None) -> "Dataset":
        """Apply function that returns multiple elements."""
        pass

    def filter(self, predicate: Callable[[T], bool], name: str | None = None) -> "Dataset":
        """Filter elements based on predicate."""
        pass

    def map_batches(
        self,
        fn: Callable[[list[T]], list[U]],
        batch_size: int = 32,
        name: str | None = None
    ) -> "Dataset":
        """Apply function to batches of elements."""
        pass

    def reshuffle(self, name: str | None = None) -> "Dataset":
        """Force a shuffle/reshard operation for load balancing."""
        pass

    def group_by_key(
        self,
        key_fn: Callable[[T], str],
        name: str | None = None
    ) -> "GroupedDataset":
        """Group elements by key."""
        pass

    def with_resume_checkpoint(
        self,
        checkpoint_path: str,
        id_fn: Callable[[T], str]
    ) -> "Dataset":
        """Enable resumption by filtering already-processed IDs."""
        pass

    def write_files(
        self,
        output_path: str,
        file_format: str = "jsonl.gz",
        num_shards: int | None = None,
    ) -> "WriteResult":
        """Write to files with automatic sharding."""
        pass

    def run(self) -> "PipelineResult":
        """Execute the pipeline."""
        return self._pipeline.run()

    def run_and_wait(self) -> "PipelineResult":
        """Execute and wait for completion."""
        result = self._pipeline.run()
        result.wait_until_finish()
        return result
```

### 2. FileProcessor (Replaces @ray.remote Pattern)

```python
# lib/ml-flow/src/ml_flow/io.py

from typing import Callable, Any
import apache_beam as beam

class FileProcessor:
    """
    Process files in parallel, maintaining input/output structure.

    Replaces the common Ray pattern:
        @ray.remote
        @cached_or_construct_output(success_suffix="SUCCESS")
        def process_file(input_path, output_path):
            ...

        # Manual orchestration
        pending = []
        for file in files:
            while len(pending) >= max_in_flight:
                ready, _ = ray.wait(pending, num_returns=1)
            pending.append(process_file.remote(file, out))
    """

    @staticmethod
    def process_directory(
        input_pattern: str,
        output_path: str,
        process_fn: Callable[[str, str], Any],
        options: DatasetOptions,
        success_suffix: str = "SUCCESS",
        resume: bool = True,
    ) -> "PipelineResult":
        """
        Process all files matching pattern.

        Example:
            FileProcessor.process_directory(
                "gs://input/**/*.jsonl.gz",
                "gs://output/",
                process_fn=transform_html_to_text,
                options=options
            )

        Features:
        - Automatic parallelization (Dataflow scales workers)
        - Resumption via .SUCCESS markers
        - Fault tolerance (automatic retries)
        - No manual throttling needed
        """
        pipeline = beam.Pipeline(options=options.pipeline_options)

        # Match files
        files = (
            pipeline
            | "MatchFiles" >> beam.io.fileio.MatchFiles(input_pattern)
            | "GetMetadata" >> beam.Map(lambda match: match.metadata)
        )

        # Filter finished if resuming
        if resume:
            finished = _get_finished_files(output_path, success_suffix)
            files = files | "FilterFinished" >> beam.Filter(
                lambda meta: _output_path_for(meta.path, ...) not in finished
            )

        # Process each file
        results = (
            files
            | "ProcessFile" >> beam.Map(
                lambda meta: _process_single_file(
                    meta.path,
                    _output_path_for(meta.path, input_pattern, output_path),
                    process_fn,
                    success_suffix
                )
            )
        )

        return pipeline.run()
```

### 3. ModelInference (Replaces Ray Data + Actor Pool)

```python
# lib/ml-flow/src/ml_flow/inference.py

from typing import Callable, Any
import apache_beam as beam
from apache_beam.ml.inference.base import RunInference, ModelHandler
from dataclasses import dataclass

@dataclass
class InferenceConfig:
    """Configuration for batch inference."""
    model_path: str
    batch_size: int = 32
    max_batch_buffering_duration_ms: int = 1000
    min_batch_size: int = 1

    # Resource configuration (Dataflow)
    num_workers: int | None = None  # Auto-scale
    worker_machine_type: str = "n1-standard-16"
    worker_accelerator_type: str | None = None  # e.g., "nvidia-tesla-t4"
    worker_accelerator_count: int = 1


class ModelInference:
    """
    Batch inference wrapper for ML models.

    Replaces:
    - Ray Data map_batches with actors
    - AutoscalingActorPool (custom 426-line implementation)

    Uses Beam's RunInference with automatic:
    - Model loading per worker
    - Batching with configurable parameters
    - GPU/TPU attachment
    - Fault tolerance
    """

    @staticmethod
    def create_handler(
        model_name: str,
        model_loader: Callable[[], Any],
        predict_fn: Callable[[Any, list], list],
    ) -> ModelHandler:
        """
        Create a custom model handler.

        Example:
            def load_model():
                from marin.processing.classification import create_classifier
                return create_classifier(model_path, ...)

            def predict(model, batch):
                return model(batch)

            handler = ModelInference.create_handler(
                "fasttext-quality", load_model, predict
            )
        """
        class CustomModelHandler(ModelHandler):
            def __init__(self):
                self._model = None

            def load_model(self):
                if self._model is None:
                    self._model = model_loader()
                return self._model

            def run_inference(self, batch, model, inference_args=None):
                return predict_fn(model, batch)

            def get_metrics_namespace(self):
                return f"inference/{model_name}"

        return CustomModelHandler()

    @staticmethod
    def run_inference(
        dataset: Dataset,
        handler: ModelHandler,
        config: InferenceConfig,
    ) -> Dataset:
        """
        Run batch inference on dataset.

        Example:
            ds = Dataset.from_files("gs://input/*.jsonl.gz")
            handler = ModelInference.create_handler(...)
            results = ModelInference.run_inference(ds, handler, config)
            results.write_files("gs://output/")
        """
        pcoll = dataset._pcollection

        results = (
            pcoll
            | "RunInference" >> RunInference(
                model_handler=handler,
                inference_args={
                    "min_batch_size": config.min_batch_size,
                    "max_batch_size": config.batch_size,
                    "max_batch_duration_secs":
                        config.max_batch_buffering_duration_ms / 1000,
                }
            )
        )

        return Dataset(results, dataset._options)
```

### 4. DataflowOptions Configuration

```python
# lib/ml-flow/src/ml_flow/options.py

from dataclasses import dataclass
from apache_beam.options.pipeline_options import PipelineOptions

@dataclass
class DataflowOptions:
    """
    Configuration for Dataflow pipelines.

    Key features:
    - FlexRS for batch (40-60% cost savings)
    - Auto-scaling based on throughput
    - Machine type selection
    - GPU/TPU support
    """
    project: str
    region: str = "us-central1"
    temp_location: str = None
    staging_location: str = None

    # Auto-scaling
    max_num_workers: int = 1000
    autoscaling_algorithm: str = "THROUGHPUT_BASED"

    # FlexRS (flexible resource scheduling for batch)
    use_flex_rs: bool = True  # 40-60% cost savings

    # Resources
    machine_type: str = "n1-standard-16"
    disk_size_gb: int = 100

    # Execution
    wait_for_completion: bool = True

    def to_pipeline_options(self) -> PipelineOptions:
        """Convert to Beam PipelineOptions."""
        options_dict = {
            'runner': 'DataflowRunner',
            'project': self.project,
            'region': self.region,
            'temp_location': self.temp_location or f"gs://{self.project}-temp/dataflow/temp",
            'staging_location': self.staging_location or f"gs://{self.project}-temp/dataflow/staging",
            'max_num_workers': self.max_num_workers,
            'autoscaling_algorithm': self.autoscaling_algorithm,
            'machine_type': self.machine_type,
            'disk_size_gb': self.disk_size_gb,
        }

        if self.use_flex_rs:
            options_dict['flexrs_goal'] = 'COST_OPTIMIZED'

        return PipelineOptions(**options_dict)
```

---

## Migration Patterns

### Pattern 1: File-Level Processing

**Before (Ray)**:
```python
# transform/wikipedia/transform_wikipedia.py

@ray.remote(max_calls=1)
@cached_or_construct_output(success_suffix="SUCCESS")
def process_file_ray(input_path: str, output_path: str, config: Config):
    """Transform Wikipedia HTML to markdown."""
    docs = read_jsonl_gz(input_path)
    transformed = [transform_wikipedia_doc(doc, config) for doc in docs]
    write_jsonl_gz(output_path, transformed)

@ray.remote(num_cpus=0.1)
def run_transform(config: TransformConfig):
    """Orchestrate file processing."""
    files = fsspec_glob(config.input_path + "/**/*.jsonl.gz")

    # Manual throttling
    pending = []
    max_in_flight = 100

    for input_file in files:
        # Wait if too many pending
        while len(pending) >= max_in_flight:
            ready, pending = ray.wait(pending, num_returns=1)
            ray.get(ready[0])

        output_file = rebase_file_path(
            config.input_path, input_file, config.output_path
        )
        pending.append(process_file_ray.remote(input_file, output_file, config))

    # Wait for remaining
    ray.get(pending)
```

**After (Beam)**:
```python
# transform/wikipedia/transform_wikipedia.py

from ml_flow import FileProcessor, DataflowOptions

def process_file(input_path: str, output_path: str, config: Config):
    """Transform Wikipedia HTML to markdown."""
    docs = read_jsonl_gz(input_path)
    transformed = [transform_wikipedia_doc(doc, config) for doc in docs]
    write_jsonl_gz(output_path, transformed)

def run_transform(config: TransformConfig):
    """Orchestrate file processing."""
    options = DataflowOptions(
        project=config.gcp_project,
        region=config.region,
        use_flex_rs=True,  # 40% cost savings
    )

    FileProcessor.process_directory(
        input_pattern=config.input_path + "/**/*.jsonl.gz",
        output_path=config.output_path,
        process_fn=lambda inp, out: process_file(inp, out, config),
        options=options,
        resume=True,
    )
```

**Changes**:
- ❌ Remove `@ray.remote` decorator
- ❌ Remove `@cached_or_construct_output` (handled by FileProcessor)
- ❌ Remove manual throttling with `ray.wait()`
- ✅ Use `FileProcessor.process_directory()`
- ✅ Add FlexRS for cost optimization
- **Result**: ~60% less code, auto-scaling, 40% cost savings

---

### Pattern 2: Batch Inference with Actor Pool

**Before (Ray)**:
```python
# processing/classification/inference.py

from marin.processing.classification.autoscaler import AutoscalingActorPool

@ray.remote
def run_inference(config: InferenceConfig):
    """Run classification inference with autoscaling."""

    # Create autoscaling actor pool (custom 426-line implementation)
    task_queue = Queue()
    result_queue = Queue()
    pool = AutoscalingActorPool(
        AutoClassifierRayActor,
        model_name_or_path=config.model_name,
        attribute_name=config.attribute_name,
        model_type=config.model_type,
        task_queue=task_queue,
        result_queue=result_queue,
        autoscaler_config=AutoscalingActorPoolConfig(
            min_actors=1,
            max_actors=10,
            scale_up_threshold=0.8,
            scale_down_threshold=0.2,
        ),
    )

    # Read files and enqueue batches
    for file in files:
        for batch in read_batches(file, batch_size=512):
            task_queue.put(batch)

    # Collect results
    while num_collected < num_batches:
        result = result_queue.get()
        write_result(result)

    pool.shutdown()
```

**After (Beam)**:
```python
# processing/classification/inference.py

from ml_flow import Dataset, ModelInference, InferenceConfig, DataflowOptions

def run_inference(config: InferenceConfig):
    """Run classification inference with autoscaling."""

    # Configure Dataflow
    dataflow_opts = DataflowOptions(
        project=config.gcp_project,
        use_flex_rs=False,  # Inference needs fast turnaround
        max_num_workers=20,
    )

    dataset_opts = DatasetOptions(
        pipeline_options=dataflow_opts.to_pipeline_options(),
        resume=True,
    )

    # Create dataset
    ds = Dataset.from_files(
        config.input_path + "/**/*.jsonl.gz",
        options=dataset_opts
    )

    # Enable resumption
    ds = ds.with_resume_checkpoint(
        config.output_path,
        id_fn=lambda x: x.get("id")
    )

    # Create model handler
    def load_classifier():
        from marin.processing.classification.classifier import create_classifier
        return create_classifier(
            config.model_name,
            config.attribute_name,
            config.model_type
        )

    def predict(model, batch):
        return model(batch)

    handler = ModelInference.create_handler(
        config.model_name, load_classifier, predict
    )

    # Run inference (Beam handles batching, scaling, fault tolerance)
    results = ModelInference.run_inference(
        ds,
        handler,
        InferenceConfig(
            model_path=config.model_name,
            batch_size=512,
        )
    )

    # Write results
    results.write_files(config.output_path, file_format="jsonl.gz")
    results.run_and_wait()
```

**Changes**:
- ❌ Remove `AutoscalingActorPool` (426 lines → 0 lines)
- ❌ Remove manual queue management
- ❌ Remove custom scaling logic
- ✅ Use Beam's `RunInference` with auto-scaling
- ✅ Declarative batching configuration
- **Result**: ~70% less code, simpler, better scaling

---

### Pattern 3: Multi-Stage Pipeline (MEDU)

**Before (Ray)**:
```python
# datashop/pipeline.py

@ray.remote(max_restarts=-1)
class MEDUPipeline:
    """Multi-stage pipeline with state."""

    def __init__(self, model_name, corpus_contents, ...):
        self.llm = vLLMProvider(model_name)
        self.generated_descriptions = []

    def stage1_generate_descriptions(self):
        """Generate benchmark descriptions from corpus samples."""
        prompts = [create_prompt(doc) for doc in self.corpus_samples]
        self.generated_descriptions = self.llm.generate(prompts)

    def stage2_merge_hierarchically(self):
        """Hierarchically merge descriptions."""
        while len(self.generated_descriptions) > 1:
            pairs = chunk(self.generated_descriptions, 2)
            merged = [self.llm.generate(merge_prompt(p)) for p in pairs]
            self.generated_descriptions = merged

    def stage3_label_documents(self, input_path, output_path):
        """Use final description to label documents."""
        final_desc = self.generated_descriptions[0]
        # Process documents with final description
```

**After (Beam)**:
```python
# datashop/pipeline.py

from ml_flow import Dataset, DataflowOptions

def run_medu_pipeline(config: MEDUConfig):
    """Multi-stage MEDU pipeline."""

    options = DataflowOptions(project=config.gcp_project)
    dataset_opts = DatasetOptions(pipeline_options=options.to_pipeline_options())

    # Stage 1: Generate descriptions from corpus
    corpus_ds = Dataset.from_files(config.corpus_contents, options=dataset_opts)

    descriptions = (
        corpus_ds._pcollection
        | "GenerateDescriptions" >> beam.ParDo(
            GenerateBenchmarkDescriptionFn(config.model_name)
        )
        | "CollectDescriptions" >> beam.combiners.ToList()
    )

    # Stage 2: Hierarchical merging
    final_description = (
        descriptions
        | "HierarchicalMerge" >> beam.Map(
            lambda descs: hierarchically_merge_descriptions(
                descs, config.model_name
            )
        )
    )

    # Stage 3: Label documents using final description (as side input)
    input_ds = Dataset.from_files(config.input_path, options=dataset_opts)

    labeled = (
        input_ds._pcollection
        | "LabelDocuments" >> beam.ParDo(
            LabelDocumentsFn(),
            beam.pvalue.AsSingleton(final_description)  # Side input
        )
    )

    Dataset(labeled, dataset_opts).write_files(config.output_path)


class GenerateBenchmarkDescriptionFn(beam.DoFn):
    """Generate descriptions (DoFn for worker-local model)."""

    def __init__(self, model_name):
        self.model_name = model_name
        self._llm = None

    def setup(self):
        """Load model once per worker."""
        self._llm = vLLMProvider(self.model_name)

    def process(self, doc):
        prompt = create_prompt(doc)
        yield self._llm.generate([prompt])[0]


def hierarchically_merge_descriptions(descriptions: list[str], model_name: str):
    """Hierarchical merging (runs on single worker for aggregation)."""
    llm = vLLMProvider(model_name)

    while len(descriptions) > 1:
        pairs = [descriptions[i:i+2] for i in range(0, len(descriptions), 2)]
        merged = [llm.generate(merge_prompt(p))[0] for p in pairs]
        descriptions = merged

    return descriptions[0]
```

**Changes**:
- ❌ Remove Ray actor state management
- ✅ Use Beam DoFns with `setup()` for model loading
- ✅ Use side inputs for passing aggregated data
- ✅ Explicit stage separation
- **Result**: More explicit data flow, better testability

---

### Pattern 4: Training Integration (No Change)

**Current (Ray)**:
```python
# training/training.py

@ray.remote(num_cpus=0.1)
def run_levanter_train_lm(config: TrainLmOnPodConfig):
    @ray.remote(**hw_config.as_remote_kwargs(), max_calls=1)
    def train_lm_task():
        train_lm.main(train_config)

    return run_on_pod_resumable(train_lm_task, hw_config, ...)
```

**After (No Change)**:
```python
# training/training.py - STAYS THE SAME

# Training stays on Ray for:
# - TPU pod management (Levanter integration)
# - Fine-grained TPU resource control
# - Existing working infrastructure
```

**Decision**: Keep all training on Ray. Beam handles only data preprocessing.

---

## Migration Phases

### Phase 0: Library Setup (Weeks 1-2)

**Deliverables**:
1. Create `lib/ml-flow` package structure
2. Implement core `Dataset` API
3. Implement `FileProcessor`
4. Unit tests with DirectRunner (local testing)
5. Basic integration test with small dataset on Dataflow
6. Documentation and examples

**Success criteria**:
- All unit tests pass with DirectRunner
- Single file processing test works on Dataflow
- API matches design document

---

### Phase 1: `src/marin/transform/` (Weeks 3-8)

**Scope**: ~40 transform scripts (legal, conversation, medical, etc.)

**Target files**:
```
transform/conversation/transform_conversation.py
transform/conversation/transform_preference_data.py
transform/legal/transform_australianlegalcorpus.py
transform/legal/transform_hupd.py
transform/legal/transform_multilegalpile.py
transform/legal/transform_edgar.py
transform/medical/lavita_to_dolma.py
transform/stackexchange/transform_stackexchange.py
transform/wikipedia/transform_wikipedia.py
transform/ar5iv/transform_ar5iv.py
... (30+ more)
```

**Migration approach**:
1. **Week 3**: Migrate 5 simple transforms (conversation, legal)
2. **Week 4**: Migrate 10 medium complexity (stackexchange, medical)
3. **Week 5**: Migrate 10 complex transforms (wikipedia, ar5iv with HTML processing)
4. **Week 6**: Migrate remaining transforms
5. **Week 7**: Parallel validation (Ray vs Beam, compare outputs)
6. **Week 8**: Performance testing and FlexRS optimization

**Validation strategy**:
```python
# For each transform:
# 1. Run Ray version on sample (1GB)
# 2. Run Beam version on same sample
# 3. Diff outputs (should be identical)
# 4. Compare cost and runtime
```

**Success criteria**:
- All transforms migrated and passing tests
- Output parity with Ray versions (bit-identical where possible)
- 30%+ cost reduction via FlexRS
- Documentation updated

---

### Phase 2: `src/marin/crawl/` (Weeks 9-12)

**Scope**: WARC processing, outlink extraction, deduplication

**Target files**:
```
crawl/fetch_links.py
crawl/get_outlinks_from_html.py
crawl/convert_responses_parquet_to_warc.py
crawl/deduplicate_outlinks.py
crawl/deduplicate_outlinks_against_cc.py
crawl/sample_from_unique_outlinks.py
crawl/count_outlinks.py
crawl/count_tokens.py
crawl/common/convert_to_html.py
crawl/fineweb_edu/convert_fineweb_edu_to_html.py
crawl/open_web_math/convert_open_web_math_to_html.py
... (15+ files)
```

**Migration approach**:
1. **Week 9**: WARC HTML extraction (convert_to_html.py patterns)
2. **Week 10**: Outlink processing and sampling
3. **Week 11**: Deduplication (exact and bloom filter)
4. **Week 12**: Quality filtering pipelines (FineWeb-Edu, OpenWebMath)

**Technical considerations**:
- Large WARC files: Use Beam's file streaming
- Bloom filters: Load as side input
- BigQuery dedup: Use Beam's BigQuery connector

**Success criteria**:
- All crawl operations migrated
- Deduplication matches Ray version
- Cost analysis shows FlexRS benefit (40-60% savings)

---

### Phase 3: `src/marin/processing/classification/` (Weeks 13-17)

**Scope**: Replace AutoscalingActorPool with Beam RunInference

**Target files**:
```
processing/classification/inference.py (main migration)
processing/classification/autoscaler.py (DELETE - 426 lines)
processing/classification/consolidate.py
processing/classification/classifier.py (adapt for Beam)
```

**Migration approach**:
1. **Week 13**: Implement `ModelInference` wrapper
2. **Week 14**: Adapt FastText classifier for Beam
3. **Week 15**: Adapt BERT classifier for Beam
4. **Week 16**: Migrate consolidation logic
5. **Week 17**: Load testing and auto-scaling validation

**Technical considerations**:
- Model loading: One-time per worker via DoFn.setup()
- Batching: Use Beam's batching with appropriate buffer sizes
- Checkpointing: Use `with_resume_checkpoint()`
- GPU support: Configure worker_accelerator_type

**Validation**:
```python
# Test auto-scaling behavior:
# 1. Start with small workload (observe scale-up)
# 2. Increase load (observe more workers)
# 3. Reduce load (observe scale-down)
# 4. Compare to Ray's AutoscalingActorPool behavior
```

**Success criteria**:
- Classification inference migrated
- Auto-scaling comparable to Ray's AutoscalingActorPool
- Model loading efficient (once per worker)
- Delete 426 lines of custom autoscaler code

---

### Phase 4: `src/marin/generation/` (Weeks 18-19, PARTIAL)

**Scope**: Data prep only (NOT vLLM inference)

**Target files**:
```
generation/chunk_utils.py (chunking strategies)
generation/dataset.py (sampling, score extraction)
```

**NOT migrating** (keep on Ray):
```
generation/inference.py (vLLM inference stays on Ray)
generation/llm_generation.py (vLLM provider stays)
generation/pipeline.py (vLLM integration stays)
```

**Migration approach**:
1. **Week 18**: Migrate chunking transformations to Beam
2. **Week 19**: Migrate dataset sampling and score extraction

**Rationale for partial migration**:
- vLLM inference requires specific GPU/TPU management
- Ray provides better control for long-running inference servers
- Data prep is batch-friendly and benefits from Beam
- Keep inference on Ray for now, revisit later

**Success criteria**:
- Chunking and sampling migrated to Beam
- vLLM inference still on Ray (working as before)
- Clear separation between prep (Beam) and inference (Ray)

---

### Phase 5: Integration & Optimization (Weeks 20-22)

**Focus**: End-to-end pipelines, executor integration, cost optimization

**Tasks**:

1. **Executor Integration**:
   - Implement `BeamExecutorStep`
   - Update `Executor` to handle Beam steps
   - Test mixed Ray/Beam pipelines

2. **End-to-End Testing**:
   - Run full transform → classify → tokenize pipeline
   - Validate output quality
   - Performance benchmarking

3. **Cost Optimization**:
   - FlexRS tuning for batch workloads
   - Machine type selection per workload
   - Auto-scaling parameter tuning
   - Cost comparison: Ray vs Beam

4. **Documentation**:
   - Migration guide for future patterns
   - API documentation
   - Performance tuning guide
   - Cost analysis report

**Success criteria**:
- Full pipelines running on Beam
- 40%+ cost reduction on batch workloads
- Complete documentation
- Team trained on new patterns

---

## Testing Strategy

### Unit Tests (DirectRunner)

```python
# tests/test_dataset.py

def test_dataset_map():
    """Test Dataset.map() operation."""
    with TestPipeline() as p:
        ds = Dataset.from_files("test_data/*.json", options=test_options)
        result = ds.map(lambda x: {"value": x["value"] * 2})
        # Assert results

def test_dataset_filter():
    """Test Dataset.filter() operation."""
    # Similar structure

def test_resumption():
    """Test checkpoint-based resumption."""
    # Run pipeline twice, verify skip on second run
```

### Integration Tests (DataflowRunner)

```python
# tests/integration/test_file_processing.py

def test_transform_pipeline_small():
    """Test file processing on small dataset."""
    options = DataflowOptions(
        project="test-project",
        region="us-central1",
    )

    FileProcessor.process_directory(
        input_pattern="gs://test-bucket/input/*.jsonl.gz",
        output_path="gs://test-bucket/output/",
        process_fn=test_transform_fn,
        options=options,
    )

    # Validate outputs
    assert output_matches_expected()
```

### Validation Tests (Ray vs Beam Comparison)

```python
# tests/validation/test_transform_parity.py

def test_wikipedia_transform_parity():
    """Ensure Beam and Ray produce identical outputs."""

    # Run Ray version
    ray_output = run_ray_transform(test_config)

    # Run Beam version
    beam_output = run_beam_transform(test_config)

    # Compare
    assert outputs_identical(ray_output, beam_output)
```

### Performance Tests

```python
# tests/performance/test_scaling.py

def test_autoscaling_behavior():
    """Test that Dataflow scales workers appropriately."""

    # Monitor worker count over time
    # Verify scale-up under load
    # Verify scale-down when idle

def test_cost_comparison():
    """Compare cost: Ray vs Beam."""

    # Run same workload on both
    # Measure cost and runtime
    # Verify expected savings
```

---

## Cost & Performance

### Expected Cost Savings

**FlexRS (Flexible Resource Scheduling)**:
- **Batch workloads**: 40-60% cost reduction
- **Trade-off**: May wait up to 6 hours for resources
- **Ideal for**: Transform pipelines, nightly processing, non-urgent tasks

**Auto-scaling**:
- **Ray**: Fixed cluster (pay even when idle)
- **Beam**: Scale to zero (pay only for active processing)
- **Savings**: ~30% on average workload

**Combined**: ~50-70% cost reduction for batch preprocessing

### Performance Considerations

**Throughput**:
- **Ray**: Manual scaling, head node can bottleneck
- **Beam**: THROUGHPUT_BASED auto-scaling, no single bottleneck
- **Expected**: Similar or better throughput

**Latency**:
- **Ray**: Immediate start (cluster already running)
- **Beam**: Startup overhead (workers need to spin up)
- **Mitigation**: Use standard Dataflow (not FlexRS) for latency-sensitive tasks

**Fault Tolerance**:
- **Ray**: Manual retries, preemption handling
- **Beam**: Built-in retries, spot instance support
- **Expected**: Better resilience

### Cost Model Example

**Sample workload**: Transform 10TB of data

**Ray**:
- Cluster: 100 n1-standard-16 (24 hours for safety)
- Cost: 100 workers × $0.76/hour × 24 hours = **$1,824**

**Beam (Standard)**:
- Workers: Auto-scale (avg 50 workers, 12 hours actual)
- Cost: 50 workers × $0.76/hour × 12 hours = **$456** (-75%)

**Beam (FlexRS)**:
- Workers: Auto-scale (avg 50 workers, 16 hours with delay)
- Cost: 50 workers × $0.46/hour × 16 hours = **$368** (-80%)

**Winner**: Beam with FlexRS saves **$1,456 (80%)** per 10TB workload

---

## Open Questions & Decisions

### 1. Tokenization Migration ⚠️

**Question**: Should we migrate Levanter tokenization to Beam?

**Current**: Ray remote function wraps Levanter's `build_or_load_cache()`

**Options**:
A. Keep on Ray (stable, working)
B. Migrate to Beam (consistency)
C. Replace with pure Beam implementation (no Levanter)

**Recommendation**: **Option A** for now
- Levanter cache format is stable
- Training pipeline depends on it
- Investigate Option C in future

### 2. Generation Inference ⚠️

**Question**: Keep vLLM inference on Ray or migrate to Beam?

**Current**: Ray Data with vLLM actors

**Options**:
A. Keep on Ray (current plan)
B. Migrate to Beam RunInference with vLLM
C. Hybrid (Beam for non-vLLM, Ray for vLLM)

**Recommendation**: **Option A**
- vLLM benefits from Ray's actor model
- Long-running inference servers
- Can revisit after Beam migration stabilizes

### 3. Local Development ⚠️

**Question**: How to test locally without Dataflow costs?

**Solution**: Use DirectRunner for unit tests
```python
# Use DirectRunner for local testing
options = PipelineOptions(runner='DirectRunner')

# Use DataflowRunner for integration/production
options = PipelineOptions(runner='DataflowRunner', project='...')
```

**Recommendation**:
- Unit tests: DirectRunner (local, fast, free)
- Integration: Small Dataflow job (minimal cost)
- Production: Full Dataflow with FlexRS

### 4. Executor Framework ⚠️

**Question**: How to handle mixed Ray/Beam steps?

**Solution**: Create `BeamExecutorStep` alongside `ExecutorStep`

```python
# Mixed pipeline
steps = [
    BeamExecutorStep("transform", transform_fn, config),  # Beam
    BeamExecutorStep("classify", classify_fn, config),    # Beam
    ExecutorStep("tokenize", tokenize_fn, config),        # Ray
    ExecutorStep("train", train_fn, config),              # Ray
]

executor.run(steps)  # Handles both types
```

**Recommendation**: Support both, gradually migrate to Beam where beneficial

---

## Risk Mitigation

### Risk 1: Output Parity with Ray

**Risk**: Beam outputs differ from Ray versions

**Mitigation**:
- Parallel validation during migration
- Bit-identical comparison where possible
- Checksum validation
- Gradual rollout (keep Ray running)

**Rollback**: Keep Ray code until Beam validated

### Risk 2: Cost Overrun

**Risk**: Dataflow more expensive than expected

**Mitigation**:
- Start with FlexRS for batch
- Monitor costs closely
- Set budget alerts
- Use quotas to cap spending

**Rollback**: Return to Ray if costs higher

### Risk 3: Performance Regression

**Risk**: Beam slower than Ray

**Mitigation**:
- Benchmark each migration
- Tune auto-scaling parameters
- Profile with Dataflow metrics
- Optimize worker machine types

**Rollback**: Keep Ray for performance-critical paths

### Risk 4: Team Learning Curve

**Risk**: Team unfamiliar with Beam

**Mitigation**:
- Training sessions
- Documentation and examples
- Start with simple migrations
- Pair programming during migration

**Rollback**: N/A (learning investment)

### Risk 5: Levanter Integration Issues

**Risk**: Beam-processed data incompatible with Levanter

**Mitigation**:
- Keep Levanter tokenization on Ray initially
- Validate cache format compatibility
- Test with small training runs

**Rollback**: Keep tokenization on Ray

---

## Success Metrics

### Technical Metrics

1. **Migration Coverage**:
   - Transform: 100% (40+ scripts)
   - Crawl: 100% (15+ files)
   - Classification: 100% (inference pipeline)
   - Generation: 50% (data prep only)

2. **Output Quality**:
   - 100% parity with Ray versions
   - All tests passing

3. **Performance**:
   - Throughput: ≥ Ray baseline
   - Latency: < 2x Ray (acceptable for batch)

### Cost Metrics

1. **Direct Savings**:
   - 40-60% via FlexRS
   - 30% via auto-scaling
   - **Target: 50% overall reduction**

2. **Operational Savings**:
   - Reduced maintenance (delete 426 lines autoscaler code)
   - No manual cluster management
   - Built-in monitoring

### Team Metrics

1. **Adoption**:
   - All new preprocessing uses Beam
   - Team trained on ml-flow API

2. **Maintenance**:
   - ml-flow library maintained
   - Documentation up-to-date

---

## Timeline Summary

| Phase | Duration | Target | Key Deliverable |
|-------|----------|--------|-----------------|
| 0 | 2 weeks | Library | `lib/ml-flow` core API |
| 1 | 6 weeks | Transform | 40+ scripts migrated |
| 2 | 4 weeks | Crawl | 15+ files migrated |
| 3 | 5 weeks | Classification | AutoscalingActorPool → RunInference |
| 4 | 2 weeks | Generation | Data prep (partial) |
| 5 | 3 weeks | Integration | Executor + docs |
| **Total** | **22 weeks** | **~5 months** | **Complete migration** |

---

## Next Steps

### Immediate (Week 1)
1. Create `lib/ml-flow` package structure
2. Set up testing infrastructure (DirectRunner)
3. Implement core `Dataset` API
4. Write first unit tests

### Short-term (Weeks 2-4)
1. Implement `FileProcessor`
2. Implement `ModelInference`
3. Test first simple transform migration
4. Validate on small Dataflow job

### Medium-term (Weeks 5-12)
1. Migrate all transform scripts (Phase 1)
2. Begin crawl migration (Phase 2)
3. Parallel validation
4. Cost analysis

### Long-term (Weeks 13-22)
1. Classification inference (Phase 3)
2. Partial generation migration (Phase 4)
3. Integration and optimization (Phase 5)
4. Complete documentation

---

## Conclusion

This migration plan provides a comprehensive strategy for moving Marin's preprocessing from Ray to Apache Beam/Dataflow. The module-by-module approach minimizes risk while delivering incremental value:

✅ **50%+ cost savings** via FlexRS and auto-scaling
✅ **Simpler code** (eliminate custom autoscaler, manual throttling)
✅ **Better scaling** (throughput-based, no bottlenecks)
✅ **Maintained stability** (training/RL stays on Ray)

The `lib/ml-flow` library provides a familiar Dataset API that wraps Beam's power while maintaining compatibility with existing Marin patterns.

**Recommended start**: Begin with Phase 0 (library setup) immediately, targeting first production migration in Week 8 (simple transforms with FlexRS).
