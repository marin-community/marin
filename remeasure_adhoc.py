import io, json, sys
import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from shellbox.backends.daytona.machine import DaytonaMachineFactory
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.backends.local.machine import check_local_environment
from taskcompendium.backend_requirements import RuntimeProfile
from experiments.post_training.task_curation.backend_capabilities import production_matrix, source_map
from experiments.post_training.task_curation.release_stages.plan_derivation import _resolve_population
from experiments.post_training.task_curation.sources import all_sources

RUNTIME = RuntimeProfile(network="deny", cpus=None, memory_mb=None, storage_mb=None, gpus=0, user=None, startup_timeout=900, cleanup_timeout=120)
configure_coreweave_s3()
sources = all_sources()
checkers = {"docker": DockerMachineFactory().check_environment, "local": check_local_environment,
            "daytona": DaytonaMachineFactory().check_environment, "iris": IrisMachineFactory(controller_url="http://survey.invalid").check_environment}
matrix = production_matrix(checkers)
for target in sys.argv[1:]:
    source = sources[target]
    path, count, _ = _resolve_population(source.info.id, source, "s3://marin-us-east-02a/marin/rl-data/")
    candidates = [str(path).rstrip("/") + "/specs", str(path).rsplit("/population", 1)[0].rstrip("/") + "/normalized/normalize", str(path).rstrip("/") + "/outputs/main"]
    parts = []
    for candidate in candidates:
        try:
            found = [str(i) for i in StoragePath(candidate.rstrip("/") + "/").ls() if str(i).endswith(".parquet")]
            if found:
                parts = found
                break
        except Exception:
            continue
    documents = []
    for part in parts:
        schema = pq.read_schema(io.BytesIO(StoragePath(part).read_bytes()))
        column = "task_spec" if "task_spec" in schema.names else "task_json"
        table = pq.read_table(io.BytesIO(StoragePath(part).read_bytes()), columns=[column, "runtime"])
        for index in range(table.num_rows):
            runtime = table.column("runtime")[index].as_py()
            documents.append({"task": json.loads(table.column(column)[index].as_py()), "runtime": json.loads(runtime) if runtime else {}})
    mapped = source_map(source.info.id, documents, matrix, checkers, {}, {"agent": RUNTIME, "grader": RUNTIME})
    for backend, fit in sorted(mapped.backends.items()):
        print(f"{target} {backend} support={fit.support.value} na={fit.not_applicable} errors={list(fit.validation_errors)[:1]}", flush=True)
