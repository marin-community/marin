#!/bin/bash
# Shared settings for the curriculum-n25 scripts; source it. Exports CW_KEY_ID/CW_KEY_SECRET from Mark's own
# CoreWeave key in ~/openathena/.env (CW_STORAGE_KEY_*; the Iris handoff rejects jobs without it, #10475).
# Never echoes a key.
WT=/Users/k3sc0re/openathena/branches/marin-taskforge-n25-run
ROOT=s3://marin-us-east-02a/marin/taskforge/proposals/curriculum-v3-n25-20261102
CATALOG=s3://marin-us-east-02a/marin/task-curriculum/catalogs/2026.09.20-subject-local-v3-0ce32038771d/curriculum.json
RELAY_EVEN=/muchanem/glm53-relay-rno2a-g-20261027T0414Z
RELAY_ODD=/muchanem/glm53-relay-rno2a-f-20261027T0147Z
ENV_FILE=/Users/k3sc0re/openathena/.env

CW_KEY_ID=$(sed -n 's/^CW_STORAGE_KEY_ID=//p' "$ENV_FILE")
CW_KEY_SECRET=$(sed -n 's/^CW_STORAGE_KEY_SECRET=//p' "$ENV_FILE")
[ -n "$CW_KEY_ID" ] && [ -n "$CW_KEY_SECRET" ] || { echo "CW_STORAGE_KEY_ID/SECRET missing from $ENV_FILE" >&2; exit 1; }
export CW_KEY_ID CW_KEY_SECRET
