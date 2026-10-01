# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "filelock", "pyarrow"]
# ///
"""Validate and attach a completed review and its evidence to the private Atlas."""

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from make_review import validate_collection
from native_revision_attestation import revision_attestation

ROOT = Path(__file__).parent
MARIN = ROOT.parents[1]
APPLET = "fb11c931-5861-4878-8bb5-a964d652b45f"
SQL_CLIENT = """import json
import sys
from urllib.parse import urlsplit
from marina.client import marina_request

payload = json.load(sys.stdin)
applet = payload.pop("applet")
details = marina_request("https://marina.oa.dev", "GET", f"/api/marina/applets/{applet}")
url = urlsplit(details["authenticated_url"])
origin = f"{url.scheme}://{url.netloc}"
result = marina_request(origin, "POST", f"/a/{applet}/query", json_body=payload)
json.dump(result, sys.stdout)
"""


def quality(collection):
    """Return the source rating, or None for a supplemental small-model review."""
    source_ids = {s["id"] for s in collection["subjects"] if s["level"] == "source"}
    source_syntheses = [r for r in collection["reviews"] if r["method"] == "synthesis" and r["subject_id"] in source_ids]
    if len(source_syntheses) != 1:
        raise ValueError("Publish one Atlas source population per review collection")
    source = source_syntheses[0]
    judges = [r for r in collection["reviews"] if r["method"] == "model_judgment"]
    if any(r["reviewer"]["id"] == "model:Qwen/Qwen3-Coder-30B-A3B-Instruct" for r in judges):
        return None
    runtime = [r for r in collection["reviews"] if r["method"] == "runtime_execution"]
    severe = any(
        f["kind"] == "issue" and f["severity"] in {"high", "critical"} for r in [*judges, source] for f in r["findings"]
    )
    rejects = sum(r["verdict"] == "reject" for r in judges)
    if source["verdict"] == "reject" and rejects * 2 > len(judges) and severe:
        return "bad"
    issues = any(
        f["kind"] == "issue" and f["severity"] in {"medium", "high", "critical"}
        for r in [*judges, source]
        for f in r["findings"]
    )
    if (
        source["verdict"] == "keep"
        and runtime
        and all(r["attributes"]["verification"]["status"] == "verified" for r in runtime)
        and not issues
    ):
        return "good"
    return "some_issues"


def sql(statement, parameters):
    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-c",
            SQL_CLIENT,
        ],
        cwd=MARIN,
        input=json.dumps({"applet": APPLET, "sql": statement, "parameters": parameters}, ensure_ascii=False),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def review_url(review_id: str) -> str:
    result = subprocess.run(
        ["uv", "run", "--no-sync", "marina", "applets", "versions", APPLET, "--json"],
        cwd=MARIN,
        capture_output=True,
        text=True,
        check=True,
    )
    details = json.loads(result.stdout)
    return f"{details['authenticated_url'].rstrip('/')}/v/{details['current_version']}/review.html?id={review_id}"


def upload_artifacts(artifacts):
    sql(
        """INSERT INTO review_artifacts (review_id,path,content,sha256)
        SELECT review_id,path,content,sha256 FROM jsonb_to_recordset(CAST(:items AS JSONB))
        AS item(review_id TEXT,path TEXT,content TEXT,sha256 TEXT)
        ON CONFLICT (review_id,path) DO NOTHING""",
        {"items": json.dumps(artifacts)},
    )


@dataclass(frozen=True)
class ReviewPublication:
    root: Path
    atlas_id: str
    collection: dict
    payload: dict
    subject: dict
    attestation_path: Path | None


def validated_publication(root: Path, atlas_id: str) -> ReviewPublication:
    path = root / "quality-reviews.json"
    collection = json.loads(path.read_text())
    schema = json.loads((root / "quality-review.schema.json").read_text())
    validate_collection(collection, root, schema)
    provenance = collection["execution_provenance"]
    if not provenance or not provenance["marinskyrl_commit"]:
        raise ValueError("Native reviews must record the MSkyRL commit")
    live = sql("SELECT payload FROM catalog_sources WHERE id=:id AND active", {"id": atlas_id})["rows"]
    if len(live) != 1:
        raise ValueError("Atlas source is absent or inactive")
    payload = live[0]["payload"]
    subjects = [s for s in collection["subjects"] if s["level"] == "source"]
    if len(subjects) != 1 or subjects[0]["source_id"] != atlas_id:
        raise ValueError("Review subject does not identify the requested Atlas population")
    if subjects[0]["dataset_revision"] != (payload.get("dataset_revision") or payload.get("revision")):
        raise ValueError("Reviewed data revision differs from the current Atlas source")
    attestation_path = None
    if payload["origin"] == "MarinSkyRL" and payload["revision"] != provenance["marinskyrl_commit"]:
        attestation = revision_attestation(root, payload["revision"], payload["verifier_path"])
        attestation_path = root / f"publication/native-revision-attestation-{payload['revision']}.json"
        attestation_path.parent.mkdir(parents=True, exist_ok=True)
        attestation_path.write_text(json.dumps(attestation, indent=2))
    return ReviewPublication(root, atlas_id, collection, payload, subjects[0], attestation_path)


def archive_evidence(publication: ReviewPublication, review_id: str) -> None:
    root, collection, attestation_path = publication.root, publication.collection, publication.attestation_path
    evidence_paths = {item["snapshot_path"] for review in collection["reviews"] for item in review["evidence"]}
    if attestation_path:
        evidence_paths.add(str(attestation_path.relative_to(root)))
    evidence_paths.update(
        filename for filename in ["run.json", "quality-review.schema.json"] if (root / filename).is_file()
    )
    evidence_paths.update(
        str(path.relative_to(root)) for path in root.glob("tasks/*/execution-*/solver/turn-*/response.json")
    )
    for filename in ["trajectory.json", "offline-tooling.json"]:
        evidence_paths.update(
            str(path.relative_to(root)) for path in root.glob(f"tasks/*/execution-*/*/agent/{filename}")
        )
    artifacts = []
    for relative in sorted(evidence_paths):
        path = root / relative
        content = path.read_text()
        artifacts.append(
            {
                "review_id": review_id,
                "path": relative,
                "content": content,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    batch = []
    for artifact in artifacts:
        if batch and len(json.dumps([*batch, artifact]).encode()) > 60_000:
            upload_artifacts(batch)
            batch = []
        batch.append(artifact)
    if batch:
        upload_artifacts(batch)


def publish_review(root: Path, atlas_id: str) -> dict:
    publication = validated_publication(root, atlas_id)
    collection, payload = publication.collection, publication.payload
    provenance = collection["execution_provenance"]
    path = root / "quality-reviews.json"
    review_id = hashlib.sha256(path.read_bytes()).hexdigest()
    updated = max(r["reviewed_at"] for r in collection["reviews"] if r["reviewed_at"])
    rating = quality(collection)
    sql(
        """INSERT INTO catalog_reviews (id,source_id,collection,updated_at)
        VALUES (:id,:source,CAST(:collection AS JSONB),:date) ON CONFLICT (id) DO NOTHING""",
        {"id": review_id, "source": atlas_id, "collection": json.dumps(collection), "date": updated},
    )
    archive_evidence(publication, review_id)
    native = [review for review in collection["reviews"] if review["method"] == "runtime_execution"]
    sql(
        """UPDATE catalog_sources SET quality=:quality,review_id=:review,review_date=:date,
        review_source_revision=:revision,review_verifier_revision=:verifier,traces=:traces WHERE id=:source""",
        {
            "quality": rating,
            "review": review_id,
            "date": updated,
            "revision": publication.subject["dataset_revision"],
            "verifier": payload.get("verifier_revision"),
            "traces": len(native),
            "source": atlas_id,
        },
    )
    result = {
        "atlas_id": atlas_id,
        "review_id": review_id,
        "quality": rating,
        "review_date": updated,
        "marinskyrl_commit": provenance["marinskyrl_commit"],
        "url": review_url(review_id),
    }
    (root / "atlas-publication.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--atlas-id", required=True)
    args = parser.parse_args()
    print(json.dumps(publish_review(args.run_dir.resolve(), args.atlas_id)))


if __name__ == "__main__":
    main()
