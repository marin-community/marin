# Cluster Config Migration Log

## 2026-02-03

- Updated `config.proto` to introduce platform/defaults schema, `VmType`, and
  proto3 `optional` fields for scalar overrides.

- Regenerated protobufs after schema change.
- Added `platform.py` with Platform/PlatformOps and updated autoscaler creation and CLI stop cleanup to use platform ops.
- Updated controller/config/client/cluster manager code to use `platform` and `controller` fields instead of legacy fields.
- Updated README and `examples/eu-west4.yaml` to new schema.
- Converted example/README duration literals to `milliseconds` to match proto `Duration` JSON/YAML parsing.
- Updated controller test to patch `_find_controller_vm_name` after controller lookup rename.
