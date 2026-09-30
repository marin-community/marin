import hashlib
import io
import tarfile

import pytest

from capability_pipeline.rootfs_review import RootfsReviewError, review_rootfs


def archive(entries):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as tar:
        for name, content in entries:
            member = tarfile.TarInfo(name)
            if isinstance(content, bytes):
                member.size = len(content)
                tar.addfile(member, io.BytesIO(content))
            else:
                member.type, member.linkname = content
                tar.addfile(member)
    output.seek(0)
    return output


def review(entries, **kwargs):
    return review_rootfs(
        archive(entries),
        expected_files={"/workspace/README.md": hashlib.sha256(b"rule 8").hexdigest()},
        task_roots=["/workspace", "/opt/task", "/fixtures"],
        reject_paths=["/opt/evaluator", "**/ground_truth.json"],
        **kwargs,
    )


PUBLIC = ("./workspace/README.md", b"rule 8")


def test_review_binds_actual_public_copy_and_allows_empty_mounts():
    entries = [
        PUBLIC,
        ("./var/lib/postgresql/data", (tarfile.DIRTYPE, "")),
        ("./bin", (tarfile.SYMTYPE, "usr/bin")),
        ("./usr/bin/tool", b"base"),
    ]
    result = review(entries)
    assert result["state"] == "passed"
    assert result["members"] == 4
    assert result["task_acceptance"] == "not_evaluated"
    assert len(result["inventory_sha256"]) == 64
    with pytest.raises(RootfsReviewError, match="content mismatch"):
        review([(PUBLIC[0], b"old rules")])
    with pytest.raises(RootfsReviewError, match="lacks required"):
        review([("./usr/bin/tool", b"base")])


@pytest.mark.parametrize(
    "path",
    [
        "./etc/hostname",
        "./etc/hosts",
        "./etc/resolv.conf",
        "./usr/local/bin/daytona",
        "./root/.daytona/state",
        "./var/lib/postgresql/data/PG_VERSION",
        "./tmp/capture_job.json",
    ],
)
def test_provider_and_live_database_payloads_are_rejected(path):
    with pytest.raises(RootfsReviewError, match="runtime or provider"):
        review([PUBLIC, (path, b"must not ship")])


@pytest.mark.parametrize(
    "extra, message",
    [
        (("./opt/evaluator/check.py", b"private"), "private path"),
        (("./hidden/ground_truth.json", b"private"), "private path"),
        (("./workspace/solution.py", b"private"), "undeclared task"),
        (("./workspace", (tarfile.SYMTYPE, "/somewhere")), "undeclared task"),
        (
            ("./innocent", (tarfile.SYMTYPE, "/opt/evaluator/check.py")),
            "link to private",
        ),
        (("../outside", b"bad"), "unsafe archive"),
        (PUBLIC, "duplicate archive"),
    ],
)
def test_private_payload_and_ambiguous_paths_fail(extra, message):
    with pytest.raises(RootfsReviewError, match=message):
        review([PUBLIC, extra])


def test_expected_file_cannot_be_a_link():
    with pytest.raises(RootfsReviewError, match="not a regular"):
        review([(PUBLIC[0], (tarfile.LNKTYPE, "usr/share/old-readme"))])


def test_marker_matching_crosses_read_boundaries_and_ignores_base_examples():
    content = b"a" * ((1 << 20) - 5) + b"private_marker" + b"z"
    with pytest.raises(RootfsReviewError, match="private marker"):
        review_rootfs(
            archive([("./workspace/task", content)]),
            expected_files={"/workspace/task": hashlib.sha256(content).hexdigest()},
            task_roots=["/workspace"],
            reject_paths=[],
            content_markers=["private_marker"],
        )
    assert (
        review(
            [PUBLIC, ("./usr/share/examples/doc", b"private_marker")],
            content_markers=["private_marker"],
        )["state"]
        == "passed"
    )


def test_limits_and_invalid_tar_fail():
    with pytest.raises(RootfsReviewError, match="member limit"):
        review([PUBLIC, ("./base", b"a")], max_members=1)
    with pytest.raises(RootfsReviewError, match="file-byte limit"):
        review([PUBLIC], max_file_bytes=2)
    with pytest.raises(RootfsReviewError, match="invalid rootfs"):
        review_rootfs(
            io.BytesIO(b"not gzip"),
            expected_files={"/a": "a" * 64},
            reject_paths=[],
            task_roots=[],
        )


def test_required_content_under_late_symlink_parent_is_rejected():
    with pytest.raises(RootfsReviewError, match="nondirectory"):
        review_rootfs(
            archive(
                [
                    ("./opt/runtime/start", b"start"),
                    ("./opt/runtime", (tarfile.SYMTYPE, "/elsewhere")),
                ]
            ),
            expected_files={"/opt/runtime/start": hashlib.sha256(b"start").hexdigest()},
            task_roots=[],
            reject_paths=[],
        )
