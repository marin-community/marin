# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import textwrap
from pathlib import Path

from marin.rl.skyrl import skyrl_cpu_command


def test_source_admission_checks_real_machine_bindings_without_inference_or_sandboxes(tmp_path):
    result = subprocess.run(
        skyrl_cpu_command(
            "python", "-m", "experiments.post_training.long_loop.tests.source_machine_probe", str(tmp_path)
        ),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert "thin-actor" in observed["unbound_error"]
    assert [row["task_id"] for row in observed["receipts"]] == ["plain", "thin-actor"]
    assert observed["receipts"][1]["task_machine"]["backend"] == "daytona"
    assert observed["receipts"][0]["combined_input_features"]["task_spec"]["dtype"] == "large_string"
    assert observed["incompatible_schema_rejected"] is True


def test_queue_schema_probe_runs_without_jax_and_preserves_footer_features(tmp_path):
    probe = Path(__file__).with_name("campaign_queue_schema_probe.py")
    program = textwrap.dedent(
        """
        import hashlib
        import json
        from pathlib import Path
        import sys
        import fsspec
        import pyarrow as pa
        import pyarrow.parquet as pq
        sys.modules['jax'] = None
        namespace = {'__name__': 'queue_schema_regression'}
        exec(compile(Path(sys.argv[1]).read_bytes(), '<landed-probe>', 'exec'), namespace)
        root = Path(sys.argv[2])
        catalog = root / 'catalog.json'
        catalog.write_text(json.dumps({'sources': {name: {
            'dataset_id': name, 'dataset_revision': '1', 'actor_census_input': {'measured_tasks': 1}
        } for name in ['narrow', 'wide']}}))
        for name, arrow_type in [('narrow', pa.string()), ('wide', pa.large_string())]:
            pq.write_table(pa.table({'task_spec': pa.array(['fixture'], type=arrow_type)}), root / (name+'.parquet'))
        # These are external Iris identity, HF storage and named-secret boundaries.
        namespace['get_job_info'] = lambda: {'job': 'cpu-fixture'}
        namespace['require_secret'] = lambda name: 'fixture-token'
        def local_hf(uri, token):
            name = 'wide.parquet' if 'wide@' in uri else 'narrow.parquet'
            return fsspec.filesystem('file'), str(root / name)
        namespace['url_to_fs'] = local_hf
        selection = root / 'selection.json'
        selection.write_text(json.dumps({
            'catalog_sha256': hashlib.sha256(catalog.read_bytes()).hexdigest(),
            'mix_ids': ['fixture'],
            'mixes': {'fixture': {'sources': {
                name: {'repo': name, 'revision': '1', 'count': 1} for name in ['narrow', 'wide']
            }}}
        }))
        namespace['main'](selection, str(root / 'result.json'), catalog)
        print((root / 'result.json').read_text())
        """
    )
    result = subprocess.run(
        skyrl_cpu_command("python", "-c", program, str(probe), str(tmp_path)),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["status"] == "ALL_SELECTED_QUEUE_SCHEMAS_PASS"
    assert len(observed["sources"]) == 2
    assert observed["mixes"]["fixture"]["combined_features"]["task_spec"]["dtype"] == "large_string"
