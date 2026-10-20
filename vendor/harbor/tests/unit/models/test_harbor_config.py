import json
from copy import deepcopy

import pytest
import yaml
from pydantic import ValidationError

from harbor.models.job.config import JobConfig as HarborJobConfig
from harbor.models.metric.config import MetricConfig as HarborMetricConfig
from harbor.models.task.config import TaskConfig as HarborTaskConfig
from harbor.models.trial.config import TrialConfig as HarborTrialConfig
from harbor_config import (
    JobConfig,
    apply_overlay,
    canonical_json,
    load_config_document,
    load_config_path,
    schema_fingerprint,
)
from harbor_config.models.metric.config import MetricConfig
from harbor_config.models.task.config import TaskConfig
from harbor_config.models.trial.config import TrialConfig


def test_harbor_compatibility_imports_reexport_config_package_classes() -> None:
    assert HarborJobConfig is JobConfig
    assert HarborMetricConfig is MetricConfig
    assert HarborTaskConfig is TaskConfig
    assert HarborTrialConfig is TrialConfig


def test_config_api_matches_harbor_job_config_with_legacy_migrations() -> None:
    document = yaml.safe_load(
        """
job_name: config-parity
jobs_dir: gs://example-bucket/jobs
orchestrator:
  n_concurrent_trials: 16
  quiet: true
  retry:
    max_retries: 6
    exclude_exceptions: [AgentTimeoutError, VerifierTimeoutError]
environment:
  type: daytona
  override_cpus: 8
  override_memory_mb: 16384
agents:
  - name: opencode
    model_name: served-model
    kwargs:
      compaction:
        enabled: true
      model_info:
        context_window: 65536
    env:
      DAYTONA_API_KEY: ${DAYTONA_API_KEY}
datasets:
  - name: terminal-bench
    registry:
      url: https://example.invalid/registry.json
      path: /tmp/registry
"""
    )

    full_harbor = HarborJobConfig.model_validate(deepcopy(document))
    config_only = load_config_document(deepcopy(document))

    assert canonical_json(full_harbor) == canonical_json(config_only)
    assert config_only.n_concurrent_trials == 16
    assert config_only.retry.max_retries == 6
    assert (
        config_only.datasets[0].registry_url == "https://example.invalid/registry.json"
    )
    assert config_only.datasets[0].registry_path is not None


def test_config_api_matches_harbor_validation_errors() -> None:
    invalid_document = {"datasets": [{}]}

    with pytest.raises(ValidationError) as harbor_error:
        HarborJobConfig.model_validate(invalid_document)
    with pytest.raises(ValidationError) as config_error:
        load_config_document(invalid_document)

    assert json.loads(harbor_error.value.json()) == json.loads(
        config_error.value.json()
    )


def test_config_path_matches_harbor_cli_input_rules(tmp_path) -> None:
    config_path = tmp_path / "job.yaml"
    config_path.write_text("agents:\n  - name: opencode\n")

    expected = HarborJobConfig.model_validate(yaml.safe_load(config_path.read_text()))

    assert canonical_json(load_config_path(config_path)) == canonical_json(expected)
    unsupported_path = config_path.with_suffix(".yml")
    unsupported_path.write_text(config_path.read_text())
    with pytest.raises(ValueError, match="Unsupported config file format"):
        load_config_path(unsupported_path)


def test_overlay_revalidates_and_canonicalizes_nested_values() -> None:
    config = load_config_document(
        {
            "agents": [
                {"name": "opencode", "kwargs": {"compaction": {"enabled": True}}}
            ],
            "environment": {"type": "daytona", "override_cpus": 4},
        }
    )

    resolved = apply_overlay(
        config,
        {
            "agents": [
                {
                    "name": "opencode",
                    "kwargs": {"compaction": {"max_tokens": 4096}},
                }
            ],
            "environment": {"override_memory_mb": 8192},
        },
    )

    assert resolved.environment.override_cpus == 4
    assert resolved.environment.override_memory_mb == 8192
    # Overlay lists intentionally replace whole model entries, while mappings
    # within an entry merge recursively.
    assert resolved.agents[0].kwargs == {"compaction": {"max_tokens": 4096}}
    assert canonical_json(resolved) == canonical_json(resolved)
    assert len(schema_fingerprint()) == 64
