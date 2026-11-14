# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Non-actor censuses run without optional provider or Harbor runtime packages."""

import subprocess
import sys
import textwrap


def test_reply_and_local_grader_census_without_harbor_or_daytona_runtime():
    program = textwrap.dedent(
        """
        import sys
        class MissingOptionalPackages:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'harbor', 'harbor_config', 'daytona'}:
                    raise ModuleNotFoundError(fullname, name=fullname)
        sys.meta_path.insert(0, MissingOptionalPackages())
        from experiments.post_training.task_curation.published_census import measure_envelopes
        reply = {'id': 'reply', 'environment_requirements': {}, 'grader': {'kind': 'none'}}
        local = {'id': 'local-grader', 'environment_requirements': {},
                 'grader': {'environment': {'command_semantics': 'linux_process', 'packages_lock': 'grader.lock'}}}
        measured = measure_envelopes([(reply, {}), (local, {})])
        assert measured.converted == 2
        assert measured.machine_slots == 0 and not measured.snapshot_sizes
        assert measured.execution_failure_tasks == 0
        assert measured.roles['grader']['local']['machine_slots'] == 1
        assert not measured.roles['grader']['local']['failures']
    """
    )
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
