"""Construct one streaming FineStore checkpoint namespace per source task."""

from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import distributions
from pathlib import Path
from finestore.store import DataStore
from packaging.utils import canonicalize_name

from eval.contracts.finestore_output import DEFAULT_SOURCE_PREFIX
from eval.contracts.finestore_resume import FineStoreResumeManager, OPERATION_PROTOCOL
from eval.execution.run_spec import RunSpec
from eval.limits import parse_key_value_args
from eval.resume.fingerprint import RunFingerprint, digest_files, resolve_model_revision
from eval.runtime.checkpoints import input_digest

MODEL_IDENTITY_FIELDS = {
    "pretrained", "model", "revision", "tokenizer", "tokenizer_backend", "tokenizer_revision",
    "chat_template", "chat_template_args", "chat_template_kwargs", "extra_body", "completion_content_policy",
    "truncate", "max_length", "max_model_len", "dtype",
    "lora_local_path", "peft", "add_bos_token", "model_url_suffix",
}
# Inference base_url is transport: replacement servers must serve the same
# declared model/revision. Judge endpoints remain part of judge identity below.


def installed_dependency_versions() -> dict[str, str]:
    """Record installed versions in import order, excluding wheel-owned vendors."""
    installed = list(distributions())
    owned_roots = set()
    for distribution in installed:
        root = Path(str(distribution.locate_file(""))).resolve()
        # Vendored metadata belongs to its enclosing wheel. Importing that
        # wheel must not add its private copies as installed dependencies.
        for file in distribution.files or ():
            if file.name in {"METADATA", "PKG-INFO"} and file.parent.suffix in {".dist-info", ".egg-info"}:
                nested = Path(str(distribution.locate_file(file))).resolve().parent.parent
                if nested != root and nested.is_relative_to(root):
                    owned_roots.add(nested)
    versions = {}
    for distribution in installed:
        if Path(str(distribution.locate_file(""))).resolve() not in owned_roots:
            name = canonicalize_name(distribution.metadata["Name"])
            # Independent overlays follow the same first-match precedence as
            # importlib.metadata.distribution(), rather than last-wins order.
            versions.setdefault(name, distribution.version)
    return versions


def build_resume_wiring(args, lm, run: RunSpec, *, store: DataStore, judge_configs=None):
    model_args = parse_key_value_args(args.model_args or "")
    identity = {name: value for name, value in model_args.items() if name in MODEL_IDENTITY_FIELDS}
    repository = identity.get("pretrained", identity.get("model"))
    identity["resolved_revision"] = resolve_model_revision(repository, identity.get("revision"), allow_network=False)
    identity["adapter"] = args.model
    identity["template"] = lm.chat_template(run.apply_chat_template)
    root = Path(__file__).resolve().parents[1]
    # Optional benchmark packages build inside the source tree. Their generated
    # copies must not change whether a checkpoint can resume.
    code = digest_files(
        path for path in [*root.rglob("*.py"), *root.rglob("*.yaml")]
        if not {"build", "dist", "__pycache__"}.intersection(path.relative_to(root).parts)
        and not any(part.endswith(".egg-info") for part in path.relative_to(root).parts)
    )
    # Wheels have no checkout lockfile. Record the versions actually executing
    # the run in both installed and source environments.
    dependencies = installed_dependency_versions()

    def factory(task):
        judge = (judge_configs or {}).get(task)
        fingerprint = RunFingerprint(inputs={
            "protocol": OPERATION_PROTOCOL, "task": task, "model": identity,
            "run": asdict(run), "implementation": code,
            "dependencies": dependencies,
            "judge": {"model": judge.model, "endpoint": input_digest(judge.base_url)} if judge else None,
            "include_path": args.include_path,
        })
        return FineStoreResumeManager(
            store=store,
            source_prefix=getattr(args, "finestore_output_prefix", DEFAULT_SOURCE_PREFIX),
            task_name=task, fingerprint=fingerprint, mode=args.resume_mode,
        )

    return factory
