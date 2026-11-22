"""One-time best-16 postprocessing of the failed .12.5 RL job."""
import hashlib
import json
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path

import click
import draccus
import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import DeviceConfig, ResourceConfig
from marin.evaluation.model_config import ModelConfig
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.execution.remote import remote
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import SkyRLRun, SkyRLRolePlan
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import prefix_join
from rigging.timing import Duration

from experiments.post_training.long_loop.archive import write_once
from experiments.post_training.long_loop.archive_stage import archive_graph_run
from experiments.post_training.long_loop.eval_runtime.audit_results import audit, require_evaluation_coverage
from experiments.post_training.long_loop.run_spec import resolve_spec
from experiments.post_training.long_loop.runtime_timing import report_step, timed
from experiments.post_training.long_loop.spec_pipeline import runtime_execution, stage_execution

EXPORT_JOB='/benfeuer/long-loop-best16-export-20261112-post1'
RL_JOB='/benfeuer/long-loop-full-2026.11.12.5/users-benfeuer-checkpoints-long-loop-full-rl-2026.11.12.5-e5a93d16551c'
VERSION='2026.11.12.14'
SPEC=Path(__file__).with_name('round1.yaml')
EVIDENCE='s3://marin-us-east-08a/marin/users/benfeuer/long-loop/evidence/20261112-rl5-terminal'


ORIGINAL_EVAL='s3://marin-us-east-08a/marin/users/benfeuer/long-loop/long-loop/micro-full/2026.11.12.9'
ORIGINAL_INPUTS='s3://marin-us-east-08a/marin/users/benfeuer/long-loop/long-loop/round1/eval-inputs/2026.11.12.8'
APPROVAL='s3://marin-us-east-08a/marin/users/benfeuer/long-loop/evidence/20261112-micro-eval9-relay-recovery/manifest.json'

RESULT_SUFFIX = "exports/dumped_evals/eval_only"
ACCEPTED_UNSCORED = [{"data_source": "tb2-recovery", "uid": "399", "exception_type": "AddTestsDirError"}]
RECOVERY = "s3://marin-us-east-08a/marin/users/benfeuer/long-loop/long-loop/micro-relay-recovery/2026.11.12.10"
RECOVERY_INPUTS = "s3://marin-us-east-08a/marin/users/benfeuer/long-loop/long-loop/round1/eval-relay-recovery-inputs/2026.11.12.10"
LIMITATION = ("DNA assembly original UID399/recovery UID5 is explicitly unscored. Its 1800-second agent exec "
              "hit its deadline, then cancellation stop_command received HTTP400 SANDBOX_NOT_RUNNING/no container IP. "
              "Fail-closed cleanup closed the machine and verifier upload raised AddTestsDirError. No workspace "
              "snapshot/archive remained, so verifier-only recovery was unavailable. Shellbox omitted provider "
              "idle settings: Daytona defaults to 15-minute container auto-stop, and internal guest processes "
              "do not refresh that timer. This is the source-level lifecycle explanation; the deleted sandbox "
              "prevents direct confirmation of its state transition. The other439attempts are scored; missing "
              "grade is excluded from means, never scored zero. No further GPU retry was submitted.")



def read_json(uri):
    fs, key = filesystem_for(uri)
    return json.loads(fs.cat_file(key))


def prepare_recovery(*, output, source, manifest):
    approval = read_json(manifest)
    if not approval["original_evaluation_complete"]:
        raise ValueError("Wait for the complete original evaluation before recovery")
    fs, key = filesystem_for(prefix_join(source, "rollouts.parquet"))
    raw = fs.cat_file(key)
    if hashlib.sha256(raw).hexdigest() != approval["source_parquet_sha256"]:
        raise ValueError("The original evaluation dataset changed")
    table = pq.read_table(pa.BufferReader(raw))
    selected = approval["attempts"]
    indexes = [int(row["original_uid"]) for row in selected]
    if len(set(indexes)) != len(indexes) or not indexes:
        raise ValueError("Recovery must identify distinct failed original attempts")
    rows = table.take(pa.array(indexes)).to_pylist()
    identities = []
    for index, (row, original) in enumerate(zip(rows, selected, strict=True)):
        if row["task_id"] != original["task_id"] or row["repetition_index"] != original["repetition_index"]:
            raise ValueError("Recovery task or repetition differs from the failed attempt")
        for field in ("task_spec", "task_rollout_spec"):
            if hashlib.sha256(row[field].encode()).hexdigest() != original[field + "_sha256"]:
                raise ValueError("Recovery changed the task or environment definition")
        identities.append({**original, "recovery_uid": str(index)})
    target, root = filesystem_for(output)
    target.makedirs(root, exist_ok=True)
    with target.open(prefix_join(root, "rollouts.parquet"), "wb") as destination:
        pq.write_table(table.take(pa.array(indexes)), destination)
    selection = {
        "attempts": len(rows),
        "source": source,
        "approval_manifest": manifest,
        "identities": identities,
        "attempts_per_data_source": dict(Counter(json.loads(row["task_spec"])["source"]["dataset"] for row in rows)),
    }
    write_once(prefix_join(output, "selection.json"), json.dumps(selection, indent=2).encode())


def result_rows(uri):
    fs, root = filesystem_for(prefix_join(uri, RESULT_SUFFIX))
    result = {}
    for path in sorted(fs.glob(prefix_join(root, "*.jsonl"))):
        name = path.rsplit("/", 1)[-1]
        if name != "aggregated_results.jsonl":
            result[name] = [json.loads(line) for line in fs.cat_file(path).splitlines() if line.strip()]
    return result


def merge_recovery(*, output, original, recovery, inputs, expected_selection):
    lineage = read_json(prefix_join(inputs, "selection.json"))
    original_rows = result_rows(original)
    recovery_rows = result_rows(recovery)
    retry_list = [row for rows in recovery_rows.values() for row in rows]
    retry = {str(row["uid"]): row for row in retry_list}
    if len(retry) != len(retry_list) or len(retry) != lineage["attempts"]:
        raise ValueError("Recovery results are incomplete or duplicated")
    by_original = {entry["original_uid"]: entry for entry in lineage["identities"]}
    preserved = []
    replaced = []
    for name, rows in original_rows.items():
        for index, row in enumerate(rows):
            uid = str(row["uid"])
            if uid not in by_original:
                preserved.append({"uid": uid, "sha256": hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()})
                continue
            entry = by_original[uid]
            if row["verification"]["status"] == "verified" or row.get("exception_type") != "RuntimeServiceError":
                raise ValueError("Refusing to replace a successful or unrelated original attempt")
            replacement = retry.pop(entry["recovery_uid"])
            if replacement["data_source"] != row["data_source"]:
                raise ValueError("Recovery is unverified or changed data source")
            rows[index] = {**replacement, "uid": row["uid"]}
            replaced.append({**entry, "original_verification": row["verification"], "recovery_verification": replacement["verification"]})
        write_once(prefix_join(prefix_join(output, RESULT_SUFFIX), name), b"\n".join(json.dumps(row).encode() for row in rows) + b"\n")
    if retry or len(replaced) != len(by_original):
        raise ValueError("Recovery did not replace exactly the approved failed identities")
    expected = read_json(prefix_join(expected_selection, "selection.json"))
    original_summary = audit(original, expected["attempts"], expected["attempts_per_data_source"], expected["identities"])
    merged_summary = audit(output, expected["attempts"], expected["attempts_per_data_source"], expected["identities"])
    merged_summary["accepted_unscored_attempts"] = ACCEPTED_UNSCORED
    require_evaluation_coverage(merged_summary)
    for filename in ("resolved-launch.yaml", "terminal.json"):
        fs, key = filesystem_for(prefix_join(original, filename))
        write_once(prefix_join(output, filename), fs.cat_file(key))
    write_once(prefix_join(output, "original-unscored-evaluation.json"), json.dumps(original_summary, indent=2).encode())
    write_once(prefix_join(output, "evaluation.json"), json.dumps(merged_summary, indent=2).encode())
    write_once(prefix_join(output, "recovery-lineage.json"), json.dumps({
        "original": original, "recovery": recovery, "original_status_preserved": True,
        "approval": lineage["approval_manifest"], "replaced": replaced, "preserved": preserved,
        "preserved_count": len(preserved), "replaced_count": len(replaced),
    }, indent=2).encode())


def archive_recovered(*, output, settings, resolved_spec, artifacts, trained, evaluated, completed_report, recovery):
    settings = json.loads(json.dumps(settings))
    extra = settings['additional_evidence']
    diagnosis = read_json("s3://marin-us-east-08a/marin/users/benfeuer/long-loop/evidence/20261113-micro-recovery-post5/manifest.json")
    for item in diagnosis['objects']:
        extra.append({'source':item['uri'],'name':'eval-recovery-diagnosis/'+item['name'],'category':'eval_trace','sha256':item['sha256']})
    for label, source in (('original', ORIGINAL_EVAL), ('recovery', recovery)):
        source_fs, source_root = filesystem_for(prefix_join(source, RESULT_SUFFIX))
        for key in sorted(source_fs.glob(prefix_join(source_root, '*.jsonl'))):
            raw = source_fs.cat_file(key)
            extra.append({
                'source': source_fs.unstrip_protocol(key),
                'name': 'eval-' + label + '-native/' + key.rsplit('/', 1)[-1],
                'category': 'eval_trace',
                'sha256': hashlib.sha256(raw).hexdigest(),
            })
    for filename in ('recovery-lineage.json', 'original-unscored-evaluation.json'):
        uri = prefix_join(evaluated, filename)
        fs, key = filesystem_for(uri)
        extra.append({'source':uri, 'name':'eval-recovery/'+filename, 'category':'eval_trace', 'sha256':hashlib.sha256(fs.cat_file(key)).hexdigest()})
    # The original consumed launch continues to describe all 440 original attempts.
    # Preserve the recovery launch and every retained replacement trace separately.
    for filename in ('resolved-launch.yaml', 'terminal.json'):
        uri=prefix_join(recovery,filename); fs,key=filesystem_for(uri)
        extra.append({'source':uri, 'name':'eval-recovery/'+filename, 'category':'config', 'sha256':hashlib.sha256(fs.cat_file(key)).hexdigest()})
    launch=read_json(prefix_join(recovery,'terminal.json'))['config']
    data = launch['inputs']['validation_data']
    if len(data) != 1:
        raise ValueError('Recovery must consume one exact subset of the original evaluation')
    for name, uri, category in (
        ('approval.json', APPROVAL, 'config'),
        ('input-selection.json', prefix_join(data[0]['uri'], 'selection.json'), 'dataset'),
        ('input-rollouts.parquet', prefix_join(data[0]['uri'], data[0]['relative_path']), 'dataset'),
    ):
        fs, key = filesystem_for(uri)
        extra.append({'source': uri, 'name': 'eval-recovery/' + name, 'category': category,
                      'sha256': hashlib.sha256(fs.cat_file(key)).hexdigest()})
    retention=launch['skyrl']['generator']['trajectory_retention']['output_path']
    uri=prefix_join(retention,'_retention_ledger.json'); fs,key=filesystem_for(uri)
    raw=fs.cat_file(key); ledger=json.loads(raw)
    extra.append({'source':uri, 'name':'eval-recovery/retention-ledger.json', 'category':'eval_trace', 'sha256':hashlib.sha256(raw).hexdigest()})
    for name,item in ledger['archives'].items():
        uri=prefix_join(retention,name); fs,key=filesystem_for(uri)
        raw=fs.cat_file(key)
        if len(raw)!=item['bytes']:raise ValueError('Recovery retention archive changed')
        extra.append({'source':uri,'name':'eval-recovery/retention/'+name,'category':'eval_trace','sha256':hashlib.sha256(raw).hexdigest()})
    return archive_graph_run(output=output,settings=settings,resolved_spec=resolved_spec,artifacts=artifacts,trained=trained,evaluated=evaluated,completed_report=completed_report)


@click.command()
@rl_build_options
def main():
    resolved=resolve_spec(SPEC,smoke=False)
    os.environ['MARIN_PREFIX']=resolved['storage']['output_prefix']
    os.environ['MARIN_TEMP_PREFIX']=resolved['storage']['temporary_prefix']
    configure_coreweave_s3()
    coordinator=replace(runtime_execution(resolved['execution']['coordinator']),coordinator_name='long-loop-best16-finalize-20261113-post9')
    adopted=lambda key,uri:ArtifactStep.adopt('long-loop/round1/reused-'+key,VERSION,uri)
    learned=ArtifactStep.adopt('long-loop/round1/selected-rl5',VERSION,'s3://marin-us-east-08a/marin/users/benfeuer/long-loop/long-loop/round1/selected-rl5/2026.11.12.7',kind=SkyRLRun,config={'source_job':RL_JOB,'selected_step':16,'export_job':EXPORT_JOB})
    inputs=adopted('full-eval-inputs',ORIGINAL_INPUTS)
    original=adopted('original-eval',ORIGINAL_EVAL)
    recovery_inputs=adopted('recovery-inputs',RECOVERY_INPUTS)
    evaluation=resolved['evaluation']
    recovered=adopted('recovery-eval',RECOVERY)
    merged=adopted('merged-eval','s3://marin-us-east-08a/marin/users/benfeuer/long-loop/long-loop/round1/eval-recovered/2026.11.12.11')
    sft=adopted('sft',resolved['adopted_sft']['uri']); handoff=adopted('handoff',resolved['adopted_handoff']['uri']); endpoint=adopted('endpoint',resolved['adopted_endpoint']['uri'])
    settings=resolved['archive']
    manifest=read_json(EVIDENCE+'/manifest.json')
    settings['additional_evidence'] += [{'source':row['uri'],'name':'rl-terminal/'+row['name'],'category':'eval_trace' if row['name']=='validation16-aggregated-results.jsonl' else 'config' if row['name'].endswith('.json') else 'timing','sha256':row['sha256']} for row in manifest['objects']]
    journals=[row for row in settings['additional_evidence']+resolved['adopted_sft'].get('additional_evidence',[]) if row['category']=='timing' and row['source'].endswith('.jsonl')]
    report=ArtifactStep.adopt('long-loop/report', '2026.11.12.12', 's3://marin-us-east-08a/marin/users/benfeuer/long-loop/long-loop/report/2026.11.12.12')
    resource=resolved['execution']['archive']
    resource=ResourceConfig(**{**resource,'device':draccus.decode(DeviceConfig,resource['device'])})
    terminal=apply('long-loop/round1/archive',remote(archive_recovered,resources=resource,timeout=Duration.from_seconds(259200)),version=VERSION,output=OUT,settings=settings,resolved_spec=resolved,artifacts={'sft':sft,'export':handoff,'endpoint':endpoint,'rl':learned,'eval_inputs':inputs,'eval':merged},trained=learned,evaluated=merged,completed_report=report,recovery=recovered)
    return replace(terminal,runtime_args={**terminal.runtime_args,'skyrl_execution':coordinator})


if __name__=='__main__':main()
