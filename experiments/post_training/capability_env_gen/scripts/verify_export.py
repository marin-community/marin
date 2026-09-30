"""Verify a completed run actually exported a validated task.

Usage: verify_export.py <local-terminal-pull-dir>
Checks, in order: snapshot completeness, item state, export presence,
export == harbor bytes, and TaskSpec schema conformance.
"""
import hashlib, json, sys, pathlib

root = pathlib.Path(sys.argv[1])
report = {"pull": str(root)}

man = json.loads((root / "pull-manifest.json").read_text())
report["complete_manifest"] = man.get("complete_manifest")
report["complete_snapshot"] = man.get("complete_snapshot")
report["omitted"] = len(man.get("omitted") or {})
report["files"] = len(man.get("files") or {})

stage = next((d for d in (root / "checkpoint-revalidation", root / "construction", root)
              if (d / "items").is_dir()), None)
item = next((stage / "items").glob("*"))
status = json.loads((item / "status.json").read_text())
report["item"] = item.name
report["item_state"] = status.get("state")
report["issues"] = status.get("issues")
report["export_recorded"] = status.get("export")
report["quality_review"] = (status.get("quality_review") or {}).get("state")

validated = stage / "validated" / item.name
report["export_dir_present"] = validated.is_dir()
if validated.is_dir():
    def tree(d):
        return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(d.rglob("*")) if p.is_file()}
    exported, harbor = tree(validated), tree(item / "harbor")
    report["export_files"] = len(exported)
    report["export_matches_harbor_bytes"] = exported == harbor
    spec = validated / "specification.json"
    if spec.is_file():
        import jsonschema
        schema = json.loads(pathlib.Path(
            "/Users/k3sc0re/openathena/capability_env_gen/vendor/task_spec/task-spec-v0.9.json"
        ).read_text())
        errs = sorted(jsonschema.Draft202012Validator(schema).iter_errors(
            json.loads(spec.read_text())), key=lambda e: list(e.path))
        report["taskspec_schema_errors"] = [f"{list(e.path)}: {e.message[:120]}" for e in errs]
        report["taskspec_valid"] = not errs
        report["spec_id"] = json.loads(spec.read_text()).get("id")

report["EXPORTED"] = bool(
    report.get("item_state") == "quality_accepted"
    and report.get("export_dir_present")
    and report.get("export_matches_harbor_bytes")
    and report.get("taskspec_valid")
    and report.get("complete_snapshot")
)
print(json.dumps(report, indent=2))
