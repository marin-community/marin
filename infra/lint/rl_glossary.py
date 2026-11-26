# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read the current RL loop vocabulary from its source Google Doc."""

import hashlib
from dataclasses import dataclass

import google.auth
from google.auth import compute_engine, impersonated_credentials
from google.auth.transport.requests import AuthorizedSession, Request

DOC_ID = "1hi-EZaEDPMlpmCoPAJVfdVxsvTXy3gPFVFraiYol-Uc"
DOC_URL = f"https://docs.google.com/document/d/{DOC_ID}/edit"
DOCS_SCOPE = "https://www.googleapis.com/auth/documents.readonly"
DOCS_API_URL = f"https://docs.googleapis.com/v1/documents/{DOC_ID}"


@dataclass(frozen=True)
class GlossarySource:
    title: str
    content_sha: str
    text: str


def _text_runs(node: object) -> list[str]:
    if isinstance(node, list):
        return [run for item in node for run in _text_runs(item)]
    if not isinstance(node, dict):
        return []
    if "textRun" in node:
        run = node["textRun"]
        content = run.get("content", "")
        if run.get("textStyle", {}).get("strikethrough"):
            return [f"~~{content}~~"]
        return [content]
    return [run for value in node.values() for run in _text_runs(value)]


def glossary_from_document(document: dict) -> GlossarySource:
    """Keep proposed rename strikethroughs visible to the lint reviewer."""
    content = document.get("tabs") or document["body"]
    text = "".join(_text_runs(content)).strip()
    if not text:
        raise ValueError("RL Loop Glossary has no readable text")
    content_sha = hashlib.sha256(text.encode()).hexdigest()[:12]
    return GlossarySource(document["title"], content_sha, text)


def fetch_glossary() -> GlossarySource:
    """Read the current doc through ADC, impersonating the GCE account for a Docs scope."""
    credentials, _ = google.auth.default(scopes=[DOCS_SCOPE])
    if isinstance(credentials, compute_engine.Credentials):
        request = Request()
        credentials.refresh(request)
        credentials = impersonated_credentials.Credentials(
            source_credentials=credentials,
            target_principal=credentials.service_account_email,
            target_scopes=[DOCS_SCOPE],
        )
    response = AuthorizedSession(credentials).get(DOCS_API_URL, params={"includeTabsContent": "true"}, timeout=20)
    response.raise_for_status()
    return glossary_from_document(response.json())
