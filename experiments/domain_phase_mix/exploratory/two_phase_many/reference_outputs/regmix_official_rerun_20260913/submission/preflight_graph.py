# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Resolve the four-run graph without executing steps or writing remote state.

Run from the Marin checkout with ``uv run python <this-file>``.
"""

import dataclasses
import hashlib
import json
import os
import re
from pathlib import Path

from marin.execution.context import executor_context
from marin.execution.executor import Executor
from marin.processing.tokenize import step_to_lm_mixture_component

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base
from experiments.domain_phase_mix import launch_delphi_one_phase_dsp_epoch_cap_sweep_3e18 as sweep
from experiments.domain_phase_mix import launch_delphi_regmix_reference_3e18 as launch
from experiments.llama import llama3_tokenizer

PREFIX = "gs://marin-us-east5"
OUTPUT = Path(__file__).resolve().parent
URI_PATTERN = re.compile(r"gs://[^\s\"'<>\\]+")


def gcs_uris(value):
    if isinstance(value, str):
        yield from URI_PATTERN.findall(value)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from gcs_uris(getattr(value, field.name))
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from gcs_uris(key)
            yield from gcs_uris(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from gcs_uris(item)


def main():
    os.environ["MARIN_PREFIX"] = PREFIX
    template_path = launch.LOCAL_ARTIFACT_DIR / "uncheatable_seed666200_t0/run_specs.json"
    template = base.DelphiSwarmRunSpec(**json.loads(template_path.read_text())[0])
    groups = launch.build_validation_run_specs(
        template=template, tpu_type="v6e-8", tpu_region="us-east5", tpu_zone="us-east5-b"
    )
    validation_steps = base._default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False)
        for name, step in validation_steps.items()
    }
    artifacts = []
    training_data_configs = []
    with executor_context():
        for definition, candidates, run_specs in groups:
            training_data_configs.extend(base._build_mixture_data(spec) for spec in run_specs)
            artifacts.append(
                sweep.build_launch_artifacts(
                    run_specs=run_specs,
                    candidates=candidates,
                    candidate_weights_path=launch.DEFAULT_CANDIDATE_WEIGHTS,
                    candidate_weights_sha256=launch.EXPECTED_CANDIDATE_WEIGHTS_SHA256,
                    analysis_output_path=base.DEFAULT_ANALYSIS_OUTPUT_PATH,
                    validation_configs=validation_configs,
                    definition=definition,
                )
            )
        steps = [step for artifact in artifacts for step in artifact.steps]
        executor = Executor(PREFIX, f"{PREFIX}/executor/graph-preflight-not-executed")
        for step in steps:
            executor.compute_version(step, is_pseudo_dep=False)
        resolved = executor._resolve_steps(executor.steps)
    by_name = {step.name: step for step in resolved}
    trains = [step for artifact in artifacts for step in artifact.training_steps]
    evals = [step for artifact in artifacts for step in artifact.eval_steps]
    assert len(trains) == len(evals) == 4
    records = []
    for train, evaluation in zip(trains, evals, strict=True):
        training_config = executor.configs[train]
        evaluation_config = executor.configs[evaluation]
        train_resolved = by_name[train.name]
        eval_resolved = by_name[evaluation.name]
        training_resources = train_resolved.resources
        evaluation_resources = evaluation_config.resources
        for resources in (training_resources, evaluation_resources):
            assert resources.regions == ["us-east5"], resources
            assert resources.zone == "us-east5-b", resources
        checkpoint = f"{executor.output_paths[train]}/hf/step-3006"
        assert str(evaluation_config.eval_config.checkpoint_path) == checkpoint, evaluation_config
        assert executor.output_paths[train] in eval_resolved.dep_paths, eval_resolved.dep_paths
        inline_names = sorted(training_config.validation_configs)
        uncheatable = [name for name in inline_names if name.startswith("uncheatable_eval/")]
        assert len(uncheatable) == 7, uncheatable
        spec = training_config.run_spec
        assert spec.expected_checkpoint_step == 3006 and spec.train_steps == 3007
        assert spec.trainer_seed == 0
        assert spec.data_seed == (666200 if "_u_" in spec.source_run_name else 662009)
        assert spec.phase_weights["phase_0"] == spec.phase_weights["phase_1"]
        records.append({
            "candidate_id": spec.source_run_name,
            "run_id": spec.run_id,
            "run_name": spec.run_name,
            "data_seed": spec.data_seed,
            "trainer_seed": spec.trainer_seed,
            "train_name": train.name,
            "training_output_path": executor.output_paths[train],
            "train_regions": training_resources.regions,
            "train_zone": training_resources.zone,
            "native_eval_name": evaluation.name,
            "eval_regions": evaluation_resources.regions,
            "eval_zone": evaluation_resources.zone,
            "request_set_dir": str(evaluation_config.eval_config.request_set_dir),
            "eval_checkpoint": checkpoint,
            "native_eval_output_path": executor.output_paths[evaluation],
            "eval_dependency_paths": eval_resolved.dep_paths,
            "inline_validation_names": inline_names,
            "inline_uncheatable_count": len(uncheatable),
        })
    uris = sorted(set(gcs_uris([
        list(executor.configs.values()), executor.output_paths, resolved, training_data_configs
    ])))
    assert uris and all(uri == PREFIX or uri.startswith(PREFIX + "/") for uri in uris), uris
    result = {
        "status": "pass",
        "training_count": len(trains),
        "native_eval_count": len(evals),
        "all_resolved_steps": len(resolved),
        "regional_gcs_uri_count": len(uris),
        "all_gcs_uris_in_east5": True,
        "training_data_configs_audited": len(training_data_configs),
        "submission_performed": False,
        "remote_mutation_performed": False,
        "executor_run_called": False,
        "template_path": str(template_path),
        "template_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(launch.DEFAULT_CANDIDATE_WEIGHTS.read_bytes()).hexdigest(),
        "launcher_sha256": hashlib.sha256(Path(launch.__file__).read_bytes()).hexdigest(),
        "runs": records,
        "resolved_gcs_uris": uris,
    }
    destination = OUTPUT / "full_graph_regional_preflight.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status", "training_count", "native_eval_count", "all_resolved_steps", "regional_gcs_uri_count"
    )}))


if __name__ == "__main__":
    main()
