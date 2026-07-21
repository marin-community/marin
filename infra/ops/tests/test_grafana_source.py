# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime

import pytest
from ops_workflow.grafana_source import snapshot_from_rows

NOW = datetime(2026, 7, 21, 16, 0, tzinfo=UTC)


def _row(*, labels_hash: str = "2b05ef3b1641c79a") -> dict[str, object]:
    return {
        "rule_org_id": 1,
        "rule_uid": "dns-config-forming",
        "labels": '[["cluster","cw-us-east-08a"],["namespace","kube-system"],["kind","Pod"]]',
        "labels_hash": labels_hash,
        "current_state_since": 1_784_647_897,
        "last_eval_time": 1_784_647_957,
        "fired_at": 1_784_647_897,
        "instance_annotations": '{"summary":"Nameserver limits were exceeded"}',
        "last_result": '{"condition":"C","values":{"A":6548,"C":1}}',
        "rule_title": "DNSConfigForming",
        "rule_labels": '{"severity":"warning"}',
        "rule_annotations": '{"runbook_url":"https://example.test/runbook"}',
    }


def test_snapshot_uses_grafana_rule_and_instance_identity_for_grouping() -> None:
    snapshot = snapshot_from_rows([_row()], observed_at=NOW)

    assert snapshot.observed_at == NOW
    assert len(snapshot.alerts) == 1
    item = snapshot.alerts[0]
    assert item.alert.fingerprint == "1:dns-config-forming:2b05ef3b1641c79a"
    assert item.alert.labels == {
        "severity": "warning",
        "cluster": "cw-us-east-08a",
        "namespace": "kube-system",
        "kind": "Pod",
        "alertname": "DNSConfigForming",
        "grafana_rule_uid": "dns-config-forming",
    }
    assert item.alert.annotations["summary"] == "Nameserver limits were exceeded"
    assert item.alert.values == {"A": 6548, "C": 1}
    assert item.group_labels == {"alertname": "DNSConfigForming", "cluster": "cw-us-east-08a"}
    assert item.group_key == '{"alertname":"DNSConfigForming","cluster":"cw-us-east-08a"}'


def test_snapshot_rejects_duplicate_grafana_instance_identity() -> None:
    with pytest.raises(ValueError, match="duplicate alert fingerprints"):
        snapshot_from_rows([_row(), _row()], observed_at=NOW)
