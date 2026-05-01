# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GAIR/daVinci-Dev dataset download and transform.

Two subsets, registered separately because they share neither layout nor
rendering:

- ``ctx-native`` is the LLM-enhanced PR corpus
  (``ctx-native/llm_enhanced_prs/*.parquet``). Each row is a structured PR
  (title, body, related issue, file snapshots, commits with diffs and a
  refined message, and an LLM-written summary). We render it to Markdown
  sections approximating the upstream Go renderer's layout — simpler than
  the search-and-replace edit format the original training used, but a
  reasonable first pass.
- ``env-native`` is the executable-rollouts JSONL (``env-native.jsonl``):
  SWE-Agent + GLM-4.6 trajectories on SWE-rebench tasks. Same shape as the
  ``swe_rebench_openhands`` rollouts; rendered with the same role-tagged
  format.

The HF dataset is gated (auto-approve); ``HF_TOKEN`` must be set locally
for ``download_hf_step`` to authenticate.
"""

import hashlib
import json

from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext, counters, load_jsonl

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import load_parquet_batched
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "GAIR/daVinci-Dev"
HF_REVISION = "7df0a81"


# ---------------------------------------------------------------------------
# ctx-native: PR-derived structured rows → templated Markdown
# ---------------------------------------------------------------------------

CTX_GLOBS = ["ctx-native/llm_enhanced_prs/*.parquet"]


def _render_issue(issue: dict | None) -> str:
    if not issue:
        return ""
    parts = [f"# Issue\n## {issue.get('title') or ''}"]
    body = issue.get("body") or ""
    if body:
        parts.append(body)
    for c in issue.get("comments") or []:
        author = c.get("author") or "unknown"
        body = c.get("body") or ""
        parts.append(f"### Comment by {author}\n{body}")
    return "\n\n".join(parts).strip()


def _render_relevant_files(files: list[dict] | None) -> str:
    if not files:
        return ""
    parts = ["# Relevant Files Found"]
    for f in files:
        path = f.get("path") or ""
        content = f.get("content") or ""
        parts.append(f"## {path}\n```\n{content}\n```")
    return "\n\n".join(parts)


def _render_commits(commits: list[dict] | None) -> str:
    if not commits:
        return ""
    parts = ["# Edits"]
    for c in commits:
        msg = c.get("refined_message") or c.get("message") or ""
        if msg:
            parts.append(msg)
        for d in c.get("diffs") or []:
            path = d.get("path") or ""
            patch = d.get("patch") or ""
            parts.append(f"## {path}\n```diff\n{patch}\n```")
    return "\n\n".join(parts)


def ctx_row_to_doc(row: dict) -> list[dict]:
    repo_name = row.get("repo_name") or ""
    title = row.get("title") or ""
    if not repo_name or not title:
        counters.increment("davinci_dev/ctx/dropped")
        return []

    sections = [
        f"# Repository Context\nName: {repo_name}\nDescription: {row.get('repo_description') or ''}",
        _render_issue(row.get("related_issue")),
        f"# Pull Request\n## {title}\n{row.get('body') or ''}".rstrip(),
        f"# Summary\n{row.get('pr_summary')}" if row.get("pr_summary") else "",
        _render_relevant_files(row.get("relevant_files")),
        _render_commits(row.get("commits")),
    ]
    text = "\n\n".join(s for s in sections if s).strip()
    if not text:
        counters.increment("davinci_dev/ctx/dropped")
        return []

    counters.increment("davinci_dev/ctx/kept")
    return [
        {
            "id": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text": text,
            "source": "GAIR/daVinci-Dev/ctx-native",
        }
    ]


def transform_ctx_native(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(f"{input_path}/**/*.parquet")
        .flat_map(load_parquet_batched)
        .flat_map(ctx_row_to_doc)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="davinci-dev-ctx-transform", resources=ResourceConfig(cpu=1, ram="32g"))
    ctx.execute(pipeline)


def download_davinci_dev_ctx_native_step() -> StepSpec:
    """Download and render daVinci-Dev's PR-derived (llm_enhanced_prs) subset."""
    dl = download_hf_step(
        "raw/davinci-dev-ctx-native",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=CTX_GLOBS,
    )
    return StepSpec(
        name="processed/davinci-dev-ctx-native",
        deps=[dl],
        fn=lambda output_path: transform_ctx_native(
            input_path=dl.output_path,
            output_path=output_path,
        ),
        hash_attrs={"version": "v1"},
    )


def davinci_dev_ctx_native_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the full ``(download+transform, normalize)`` chain for daVinci-Dev ctx-native."""
    processed = download_davinci_dev_ctx_native_step()
    return (
        processed,
        normalize_step(name="normalized/davinci-dev-ctx-native", download=processed),
    )


# ---------------------------------------------------------------------------
# env-native: SWE-Agent multi-turn rollouts → role-tagged transcript
# ---------------------------------------------------------------------------

ENV_GLOBS = ["env-native.jsonl"]


def _render_tool_call(tc: dict) -> str:
    func = tc.get("function") or {}
    name = func.get("name") or "unknown"
    args = func.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    parts = [f"<tool_call:{name}>"]
    if isinstance(args, dict):
        for k, v in args.items():
            parts.append(f"  {k}: {v}")
    elif args is not None:
        parts.append(f"  {args}")
    parts.append(f"</tool_call:{name}>")
    return "\n".join(parts)


def _render_env_message(msg: dict) -> str:
    role = msg.get("role") or "unknown"
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls")
    parts = [f"<{role}>"]
    if content:
        parts.append(content)
    if tool_calls:
        for tc in tool_calls:
            parts.append(_render_tool_call(tc))
    parts.append(f"</{role}>")
    return "\n".join(parts)


def _success_to_tag(success: bool | None) -> str | None:
    if success is None:
        return None
    if success:
        return "This trajectory solved the task successfully."
    return "This trajectory failed to solve the task."


def env_row_to_doc(row: dict) -> list[dict]:
    messages = row.get("messages")
    if not messages:
        counters.increment("davinci_dev/env/dropped")
        return []
    tag = _success_to_tag(row.get("success") if "success" in row else None)
    rendered = "\n\n".join(_render_env_message(m) for m in messages)
    text = f"{tag}\n\n{rendered}" if tag else rendered

    counters.increment("davinci_dev/env/kept")
    return [
        {
            "id": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text": text,
            "source": "GAIR/daVinci-Dev/env-native",
        }
    ]


def transform_env_native(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(f"{input_path}/env-native.jsonl")
        .flat_map(load_jsonl)
        .flat_map(env_row_to_doc)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="davinci-dev-env-transform", resources=ResourceConfig(cpu=1, ram="16g"))
    ctx.execute(pipeline)


def download_davinci_dev_env_native_step() -> StepSpec:
    """Download and render daVinci-Dev's executable-rollouts (env-native) subset."""
    dl = download_hf_step(
        "raw/davinci-dev-env-native",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=ENV_GLOBS,
    )
    return StepSpec(
        name="processed/davinci-dev-env-native",
        deps=[dl],
        fn=lambda output_path: transform_env_native(
            input_path=dl.output_path,
            output_path=output_path,
        ),
        hash_attrs={"version": "v1"},
    )


def davinci_dev_env_native_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the full ``(download+transform, normalize)`` chain for daVinci-Dev env-native."""
    processed = download_davinci_dev_env_native_step()
    return (
        processed,
        normalize_step(
            name="normalized/davinci-dev-env-native",
            download=processed,
            # Transform emits one ~3 GB parquet; nested message structs blow up
            # the default 16 GiB worker on load. Bump to 64 GiB.
            worker_resources=ResourceConfig(cpu=2, ram="64g", disk="10g"),
        ),
    )
