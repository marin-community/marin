# Plan: Config management cleanup (iris)

## Goals
- Create `config.proto` as the single schema for Iris configuration knobs.
- Stop building derived config helpers like `to_timeout_config(...)`; timeouts/other knobs become sub-messages on the primary config.
- Thread structured config objects through callers/descendants instead of flattening or re-mapping fields.

## Assumptions (confirmed)
- Create `config.proto` alongside existing protos (in `lib/iris/src/iris/rpc/`).
- Move **all configuration protos** into `config.proto` (including config-like messages currently in `cluster.proto` and `vm.proto`).
- YAML and all code can change; no backward compatibility is required.

## Proposed design
- New file `lib/iris/src/iris/rpc/config.proto`:
  - `IrisClusterConfig` becomes the **root** config.
  - Move **all config-ish messages** from `vm.proto` and `cluster.proto` into `config.proto`.
    - From `vm.proto`: `BootstrapConfig`, `TimeoutConfig`, `ControllerVmConfig`, `GcpControllerConfig`,
      `ManualControllerConfig`, `ProviderConfig`, `TpuProvider`, `ManualProvider`, `ScaleGroupConfig`,
      and any other config messages found during inventory.
    - From `cluster.proto`: move any config messages (e.g., environment, scheduling, or request config
      messages) as discovered in the inventory step.
  - Add explicit nested config fields on `IrisClusterConfig`:
    - `bootstrap: BootstrapConfig`
    - `timeouts: TimeoutConfig`
    - `ssh: SshConfig` (new message for user/key/connect defaults)
    - `label_prefix`, `provider_type`, `project_id`, `region`, `zone`, `controller_vm`, `scale_groups`, etc.
- `vm.proto` keeps runtime state/messages (e.g., `VmInfo`, `ScaleGroupStatus`) and *imports* `config.proto` for config message types.
- `cluster.proto` keeps runtime/job/control-plane messages, importing `config.proto` where it needs config types.

Example snippet (shape only):
```proto
// config.proto
package iris.config;

message TimeoutConfig {
  int32 boot_timeout_seconds = 1;        // Default: 300
  int32 init_timeout_seconds = 2;        // Default: 600
  int32 ssh_connect_timeout_seconds = 3; // Default: 30
  int32 ssh_poll_interval_seconds = 4;   // Default: 5
}

message SshConfig {
  string user = 1;        // Default: "root"
  string private_key = 2;
  int32 port = 3;         // Default: 22
}

message IrisClusterConfig {
  string provider_type = 1;
  string project_id = 2;
  string region = 3;
  string zone = 4;

  BootstrapConfig bootstrap = 10;
  TimeoutConfig timeouts = 11;
  SshConfig ssh = 12;

  ControllerVmConfig controller_vm = 20;
  map<string, ScaleGroupConfig> scale_groups = 30;

  string label_prefix = 40; // Default: "iris"
}
```

## Work plan
1. **Inventory current config usage**
   - Search for all config message usages across `iris/rpc/*.proto`, `iris/cluster/**`, and `iris/cli/**`.
   - Identify config-like messages in `cluster.proto` to relocate into `config.proto`.

2. **Define `config.proto` and move config messages**
   - Create `lib/iris/src/iris/rpc/config.proto` with **all configuration schema** (from `vm.proto` and `cluster.proto`).
   - Remove/move config messages from the other protos (keep runtime-only messages there).
   - Update package names consistently; regenerate imports in dependent protos if needed.

3. **Wire protos together**
   - Update `vm.proto` to import `config.proto` and reference config types from there.
   - Update `cluster.proto` to import `config.proto` for any config types it references.

4. **Refactor `config.py` to use nested config objects**
   - Replace `to_bootstrap_config` / `to_timeout_config` with direct use of `config.bootstrap` and `config.timeouts`.
   - Replace `to_ssh_config` to read `config.ssh` and per-group overrides.
   - Thread config objects down to `_create_manager` / `ManualVmManager` / `TpuVmManager` without re-mapping.

   Example changes (shape only):
   ```py
   manager = _create_manager(
       provider=provider_type,
       config=group_config,
       bootstrap_config=config.bootstrap,
       timeouts=config.timeouts,
       ssh_config=ssh_config or to_ssh_config(config),
       label_prefix=config.label_prefix or "iris",
       ...,
   )
   ```

5. **Update YAML parsing & config serialization**
   - Adjust `load_config` docstrings/examples to reflect nested config structure and new config fields.
   - Ensure `ParseDict` expects nested fields (`bootstrap`, `timeouts`, `ssh`, etc.), not flat fields.
   - Update any config serialization paths to handle nested objects and moved config messages.

6. **Thread configs through descendants**
   - Follow references from `config.py` into `VmManager` implementations and any callers.
   - Make sure downstream components accept the new nested proto shape and do not rebuild copies.

7. **Update docs/recipes**
   - Update any YAML examples in `docs/` or recipes to the new schema.
   - Remove mentions of flat timeout fields and config messages that moved from `vm.proto`/`cluster.proto`.

8. **Tests & validation**
   - Update existing tests that instantiate `IrisClusterConfig` or use YAML fixtures.
   - Add integration-style test that loads YAML with nested config and spins a minimal autoscaler config (if one exists already).

## Checklist of likely touch points
- `lib/iris/src/iris/rpc/vm.proto`
- `lib/iris/src/iris/rpc/cluster.proto`
- `lib/iris/src/iris/rpc/config.proto` (new)
- `lib/iris/src/iris/cluster/vm/config.py`
- `lib/iris/src/iris/cluster/vm/` managers that use `TimeoutConfig` and `BootstrapConfig`
- Docs under `docs/` referencing YAML config

## Notes
- I’m assuming **no backwards compatibility** (per repo guidance). If you want a transitional parser for old YAML, call it out explicitly and I’ll fold that into the plan.
