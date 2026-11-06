import hashlib
from concurrent.futures import ThreadPoolExecutor

import pytest

from autopost.contracts import (
    ArtifactComponent,
    ArtifactManifest,
    ArtifactRef,
    StagedArtifact,
)
from autopost.store.artifacts import LocalArtifactStore
from autopost.store.errors import (
    ConflictError,
    CorruptArtifactError,
    MissingArtifactError,
    StoreError,
)


def make_manifest(
    identity="output",
    *,
    payload=b"immutable payload",
    producer="job-fixture",
    name="component",
):
    component = ArtifactComponent(
        name=name,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        media_type="application/octet-stream",
    )
    return ArtifactManifest.create(
        artifact_id=identity,
        kind="report",
        producer_job_id=producer,
        components=(component,),
    )


@pytest.fixture
def artifacts(tmp_path):
    return LocalArtifactStore(tmp_path / "artifacts")


def test_staging_is_invisible_until_atomic_finalization_and_survives_reopen(artifacts):
    manifest = make_manifest()
    staged = artifacts.stage(manifest.artifact.artifact_id)
    artifacts.write(staged, "component", b"immutable payload")
    with pytest.raises(MissingArtifactError):
        artifacts.resolve(manifest.artifact)
    artifacts.validate(staged, manifest)
    ref = artifacts.finalize(staged, manifest)
    reopened = LocalArtifactStore(artifacts.root)
    assert reopened.resolve(ref) == manifest
    assert reopened.read(ref, "component") == b"immutable payload"
    assert reopened.finalize(staged, manifest) == ref


def test_retry_finalization_and_writes_never_overwrite_immutable_content(artifacts):
    manifest = make_manifest()
    for _ in range(2):
        staged = artifacts.stage("output")
        artifacts.write(staged, "component", b"immutable payload")
        artifacts.write(staged, "component", b"immutable payload")
        assert artifacts.finalize(staged, manifest) == manifest.artifact
    staged = artifacts.stage("output")
    artifacts.write(staged, "component", b"different payload")
    with pytest.raises(ConflictError):
        artifacts.write(staged, "component", b"overwrite attempt")
    with pytest.raises(ConflictError):
        artifacts.finalize(staged, make_manifest(payload=b"different payload"))
    assert artifacts.read(manifest.artifact, "component") == b"immutable payload"


@pytest.mark.parametrize(
    "defect", ["missing", "extra", "size", "hash", "descriptor", "identity"]
)
def test_manifest_or_component_defects_cannot_be_finalized(artifacts, defect):
    manifest = make_manifest()
    staged = artifacts.stage("output" if defect != "identity" else "other-output")
    if defect != "missing":
        artifacts.write(
            staged,
            "component",
            b"wrong"
            if defect == "size"
            else b"x" * len(b"immutable payload")
            if defect == "hash"
            else b"immutable payload",
        )
    if defect == "extra":
        artifacts.write(staged, "undeclared", b"extra bytes")
    if defect == "descriptor":
        manifest = manifest.model_copy(update={"producer_job_id": "altered-producer"})
    error = ConflictError if defect == "identity" else CorruptArtifactError
    with pytest.raises(error):
        artifacts.finalize(staged, manifest)
    with pytest.raises(MissingArtifactError):
        artifacts.resolve(manifest.artifact)


def test_corrupt_manifest_and_payload_are_detected_on_every_read(artifacts):
    manifest = make_manifest()
    staged = artifacts.stage("output")
    artifacts.write(staged, "component", b"immutable payload")
    artifacts.finalize(staged, manifest)
    directory = artifacts._object_directory("output")
    payload = next((directory / "payload").iterdir())
    payload.write_bytes(b"corrupt")
    with pytest.raises(CorruptArtifactError):
        artifacts.read(manifest.artifact, "component")
    payload.write_bytes(b"immutable payload")
    (directory / "manifest.json").write_bytes(b'{"schema_version":999}')
    with pytest.raises(CorruptArtifactError):
        artifacts.resolve(manifest.artifact)


def test_component_and_handle_path_traversal_is_rejected(artifacts):
    staged = artifacts.stage("valid")
    for name in ("../outside", "C:/outside", "/outside", "..\\outside"):
        with pytest.raises(ValueError):
            artifacts.write(staged, name, b"bytes")
    forged = StagedArtifact(staging_id="../escape", expected="valid")
    with pytest.raises(StoreError):
        artifacts.write(forged, "safe", b"bytes")
    swapped = StagedArtifact(staging_id=staged.staging_id, expected="other")
    with pytest.raises(ConflictError):
        artifacts.write(swapped, "safe", b"bytes")


def test_content_identity_binds_provenance_and_component_names(artifacts):
    first = make_manifest()
    assert first.artifact.sha256 == first.content_digest
    assert make_manifest(producer="other-job").artifact.sha256 != first.artifact.sha256
    assert make_manifest(name="other-name").artifact.sha256 != first.artifact.sha256
    with pytest.raises(ValueError):
        ArtifactRef(artifact_id="physical/path", sha256="a" * 64)


def test_concurrent_identical_publication_has_one_immutable_outcome(artifacts):
    manifest = make_manifest()

    def publish(_):
        backend = LocalArtifactStore(artifacts.root)
        staged = backend.stage("output")
        backend.write(staged, "component", b"immutable payload")
        return backend.finalize(staged, manifest)

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(publish, range(8))) == [manifest.artifact] * 8
    assert artifacts.resolve(manifest.artifact) == manifest


def test_windows_reserved_and_case_sensitive_names_remain_opaque(artifacts):
    for name in ("CON", "Case", "case"):
        manifest = make_manifest(identity=name, name=name)
        staged = artifacts.stage(name)
        artifacts.write(staged, name, b"immutable payload")
        artifacts.finalize(staged, manifest)
        assert artifacts.read(manifest.artifact, name) == b"immutable payload"
