-- Copyright The Marin Authors
-- SPDX-License-Identifier: Apache-2.0

CREATE TABLE alert_rule (
    org_id bigint NOT NULL,
    uid varchar(40) NOT NULL,
    title varchar(190) NOT NULL,
    labels text,
    annotations text,
    PRIMARY KEY (org_id, uid)
);

CREATE TABLE alert_instance (
    rule_org_id bigint NOT NULL,
    rule_uid varchar(40) NOT NULL,
    labels text NOT NULL,
    labels_hash varchar(190) NOT NULL,
    current_state varchar(190) NOT NULL,
    current_state_since bigint NOT NULL,
    last_eval_time bigint NOT NULL,
    fired_at bigint,
    annotations text,
    last_result text,
    PRIMARY KEY (rule_org_id, rule_uid, labels_hash)
);

INSERT INTO alert_rule (org_id, uid, title, labels, annotations) VALUES (
    1,
    'dns-config-forming',
    'DNSConfigForming',
    '{"severity":"warning"}',
    '{"description":"Kubernetes omitted nameservers beyond the pod DNS limit."}'
);

INSERT INTO alert_instance (
    rule_org_id, rule_uid, labels, labels_hash, current_state,
    current_state_since, last_eval_time, fired_at, annotations, last_result
) VALUES
    (
        1,
        'dns-config-forming',
        '[["cluster","cw-us-east-08a"],["kind","Pod"],["name","node-local-dns-dcb4s"],["namespace","kube-system"]]',
        '2b05ef3b1641c79a',
        'Alerting',
        1784647897,
        1784647957,
        1784647897,
        '{"summary":"Nameserver limits were exceeded; validate node-local DNS configuration"}',
        '{"condition":"C","values":{"A":6548,"C":1}}'
    ),
    (
        1,
        'dns-config-forming',
        '[["cluster","cw-us-east-08a"],["kind","Pod"],["name","nvidia-imex-xkljx"],["namespace","cw-nvidia-imex"]]',
        'ef356383208c86c5',
        'Alerting',
        1784647895,
        1784647957,
        1784647895,
        '{"summary":"Nameserver limits were exceeded; validate pod DNS configuration"}',
        '{"condition":"C","values":{"A":6536,"C":1}}'
    );
