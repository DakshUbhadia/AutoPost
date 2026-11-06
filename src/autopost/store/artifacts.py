"""Immutable local artifacts with isolated staging and atomic publication."""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from pathlib import Path

from pydantic import TypeAdapter

from autopost.contracts import ArtifactManifest, ArtifactRef, StagedArtifact
from autopost.contracts.manifests import ComponentName
from autopost.contracts.refs import ArtifactId

from .errors import (
    ConflictError,
    CorruptArtifactError,
    MissingArtifactError,
    StoreError,
)


def _sync_directory(path: Path) -> None:
    # Python/Windows cannot portably fsync a directory. Atomic same-volume
    # rename still provides process-crash safety; power-loss guarantees depend
    # on the underlying filesystem. Payload/manifest files are always fsynced.
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _write_once(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(path.parent)


class LocalArtifactStore:
    """Only this backend resolves logical IDs to physical filesystem paths.

    Finalized objects are never overwritten through this API. Every resolve
    verifies the manifest and all bytes, detecting out-of-band tampering.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if os.name == "nt" and not str(self.root).startswith("\\\\?\\"):
            # Object/component hashes can exceed MAX_PATH under an ordinary
            # checkout/test directory. Keep extended locators private here.
            value = str(self.root)
            self.root = Path(
                "\\\\?\\UNC\\" + value[2:]
                if value.startswith("\\\\")
                else "\\\\?\\" + value
            )
        self.root.mkdir(parents=True, exist_ok=True)
        self._staging = self.root / "staging"
        self._objects = self.root / "objects"
        for directory in (self._staging, self._objects):
            self._safe(directory)
            directory.mkdir(exist_ok=True)

    def _safe(self, path: Path) -> Path:
        if not path.resolve().is_relative_to(self.root) or path.is_symlink():
            raise CorruptArtifactError("artifact locator escapes its backend root")
        return path

    def _object_directory(self, artifact_id: str) -> Path:
        TypeAdapter(ArtifactId).validate_python(artifact_id)
        key = hashlib.sha256(artifact_id.encode("utf-8")).hexdigest()
        return self._safe(self._objects / key[:2] / key)

    def _stage_directory(self, staged: StagedArtifact) -> Path:
        staged.to_bytes()
        if not re.fullmatch(r"[0-9a-f]{32}", staged.staging_id):
            raise StoreError("invalid staging identity")
        return self._safe(self._staging / staged.staging_id)

    @staticmethod
    def _component_name(name: str) -> str:
        TypeAdapter(ComponentName).validate_python(name)
        # Hash logical names too: avoid Windows device names and case-folding.
        return hashlib.sha256(name.encode("utf-8")).hexdigest()

    def stage(self, expected: ArtifactId) -> StagedArtifact:
        TypeAdapter(ArtifactId).validate_python(expected)
        staged = StagedArtifact(staging_id=uuid.uuid4().hex, expected=expected)
        directory = self._stage_directory(staged)
        directory.mkdir()
        (directory / "payload").mkdir()
        _write_once(directory / "reservation.json", staged.to_bytes())
        _sync_directory(self._staging)
        return staged

    def _check_reservation(self, staged: StagedArtifact) -> Path:
        directory = self._stage_directory(staged)
        if not directory.is_dir():
            raise MissingArtifactError("staged artifact does not exist")
        try:
            reservation = StagedArtifact.from_bytes(
                self._safe(directory / "reservation.json").read_bytes()
            )
        except (OSError, ValueError) as exc:
            raise CorruptArtifactError("staging reservation is corrupt") from exc
        if reservation != staged:
            raise ConflictError("staging reservation belongs to a different artifact")
        return directory

    def write(self, staged: StagedArtifact, component: str, data: bytes) -> None:
        if type(data) is not bytes:
            raise StoreError("artifact payload must be bytes")
        name = self._component_name(component)
        directory = self._check_reservation(staged)
        payload = self._safe(directory / "payload")
        path = self._safe(payload / name)
        try:
            _write_once(path, data)
        except FileExistsError as exc:
            # Safe redelivery of the same component; never overwrite bytes.
            if path.read_bytes() != data:
                raise ConflictError(
                    "staged component already contains different bytes"
                ) from exc

    def _verify(self, directory: Path, manifest: ArtifactManifest) -> None:
        if manifest.content_digest != manifest.artifact.sha256:
            raise CorruptArtifactError("artifact manifest digest mismatch")
        payload = self._safe(directory / "payload")
        if not payload.is_dir():
            raise CorruptArtifactError("artifact payload is missing")
        expected = {self._component_name(item.name) for item in manifest.components}
        if {path.name for path in payload.iterdir()} != expected:
            raise CorruptArtifactError("artifact components are missing or undeclared")
        for component in manifest.components:
            path = self._safe(payload / self._component_name(component.name))
            if not path.is_file():
                raise CorruptArtifactError("artifact component is not a regular file")
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if (
                path.stat().st_size != component.size_bytes
                or actual != component.sha256
            ):
                raise CorruptArtifactError("artifact component hash or size mismatch")

    def validate(self, staged: StagedArtifact, manifest: ArtifactManifest) -> None:
        manifest = ArtifactManifest.from_bytes(manifest.to_bytes())
        directory = self._check_reservation(staged)
        if staged.expected != manifest.artifact.artifact_id:
            raise ConflictError(
                "artifact manifest differs from reserved output identity"
            )
        self._verify(directory, manifest)

    def _load_manifest(self, directory: Path) -> ArtifactManifest:
        try:
            return ArtifactManifest.from_bytes(
                self._safe(directory / "manifest.json").read_bytes()
            )
        except (OSError, ValueError) as exc:
            raise CorruptArtifactError(
                "finalized artifact manifest is missing or corrupt"
            ) from exc

    def _already_finalized(
        self, directory: Path, manifest: ArtifactManifest
    ) -> ArtifactRef:
        existing = self._load_manifest(directory)
        self._verify(directory, existing)
        if existing.to_bytes() != manifest.to_bytes():
            raise ConflictError(
                "immutable artifact identity already has different content"
            )
        return existing.artifact

    def finalize(
        self, staged: StagedArtifact, manifest: ArtifactManifest
    ) -> ArtifactRef:
        manifest = ArtifactManifest.from_bytes(manifest.to_bytes())
        staged_directory = self._stage_directory(staged)
        target = self._object_directory(manifest.artifact.artifact_id)
        if staged.expected != manifest.artifact.artifact_id:
            raise ConflictError(
                "artifact manifest differs from reserved output identity"
            )
        if not staged_directory.exists() and target.is_dir():
            return self._already_finalized(target, manifest)
        self.validate(staged, manifest)
        if target.exists():
            return self._already_finalized(target, manifest)
        manifest_path = self._safe(staged_directory / "manifest.json")
        try:
            _write_once(manifest_path, manifest.to_bytes())
        except FileExistsError:
            if self._load_manifest(staged_directory) != manifest:
                raise CorruptArtifactError(
                    "staged manifest changed before finalization"
                ) from None
        _sync_directory(staged_directory / "payload")
        _sync_directory(staged_directory)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._safe(target.parent)
        _sync_directory(self._objects)
        try:
            # Same filesystem, whole directory published at once. There is no
            # point at which resolve can observe a partially published object.
            os.rename(staged_directory, target)
        except OSError:
            if target.is_dir():
                return self._already_finalized(target, manifest)
            raise
        _sync_directory(target.parent)
        return manifest.artifact

    def resolve(self, ref: ArtifactRef) -> ArtifactManifest:
        ref = ArtifactRef.from_bytes(ref.to_bytes())
        directory = self._object_directory(ref.artifact_id)
        if not directory.is_dir():
            raise MissingArtifactError(f"artifact is not finalized: {ref.artifact_id}")
        manifest = self._load_manifest(directory)
        if manifest.artifact != ref:
            raise CorruptArtifactError(
                "finalized manifest does not match artifact reference"
            )
        self._verify(directory, manifest)
        return manifest

    def read(self, ref: ArtifactRef, component: str) -> bytes:
        manifest = self.resolve(ref)
        if component not in {item.name for item in manifest.components}:
            raise MissingArtifactError("artifact has no such logical component")
        return self._safe(
            self._object_directory(ref.artifact_id)
            / "payload"
            / self._component_name(component)
        ).read_bytes()
