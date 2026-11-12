import json
import subprocess
import sys
from pathlib import Path

from autopost.cli import main
from autopost.contracts import DatasetRef
from autopost.store.artifacts import LocalArtifactStore
from autopost.store.datasets import DatasetRepository


def test_cpu_build_cli_is_reproducible_and_exports_inspectable_review_packets(
    tmp_path, capsys
):
    arguments = ["data", "build-dev", "--root", str(tmp_path)]
    assert main(arguments) == 0
    first = json.loads(capsys.readouterr().out)
    assert (
        first["hidden_review_status"] == "pending_human_review"
        and first["fixture_only"]
    )
    assert (
        not {"hidden_audit", "hidden_sealed", "transfer_sealed"}
        & first["splits"].keys()
    )
    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out) == first
    directory = tmp_path / first["build_id"]
    packet_path = directory / "hidden-review-packets.json"
    packets = json.loads(packet_path.read_text())
    assert len(packets) == 3 and all(packet["report"]["passed"] for packet in packets)
    repo = DatasetRepository(
        tmp_path / "datasets.sqlite", LocalArtifactStore(tmp_path / "artifacts")
    )
    hidden = DatasetRef(**first["splits"]["hidden_dev"])
    for packet in packets:
        assert (
            main(
                [
                    "data",
                    "review-hidden",
                    "--root",
                    str(tmp_path),
                    "--packet",
                    str(packet_path),
                    "--program",
                    packet["program"]["program_id"],
                    "--reviewer",
                    "fixture-test-actor",
                    "--decision",
                    "approved",
                    "--notes",
                    "Synthetic CLI workflow test; not an actual human audit.",
                ]
            )
            == 0
        )
        assert json.loads(capsys.readouterr().out)["decision"] == "approved"
    assert repo.evaluation_tasks(hidden, fixture_only=True)


def test_hidden_preview_script_never_creates_research_partitions(tmp_path):
    output = tmp_path / "previews.json"
    script = (
        Path(__file__).resolve().parents[1] / "scripts/build_hidden_perturbations.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--output", str(output), "--trials", "100"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "require human review" in result.stdout
    packets = json.loads(output.read_text())
    assert all(
        packet["partition"] == "hidden_dev" and packet["review_required"]
        for packet in packets
    )


def test_missing_explicit_embeddings_fail_without_publishing_any_dataset(
    tmp_path, capsys
):
    assert (
        main(
            [
                "data",
                "build-dev",
                "--root",
                str(tmp_path),
                "--source",
                str(tmp_path / "source.jsonl"),
            ]
        )
        == 1
    )
    assert "explicit embedding" in capsys.readouterr().err
    assert not (tmp_path / "datasets.sqlite").exists()


def test_local_snapshot_and_cached_embedding_cli_matches_in_memory_build(
    tmp_path, development_result
):
    from dataclasses import asdict

    from autopost.contracts.serialization import canonical_bytes
    from autopost.data.commands import build_command

    sources, embeddings, config, programs, expected = development_result
    paths = []
    for index, source in enumerate(sources):
        path = tmp_path / f"source-{index}.jsonl"
        path.write_bytes(
            canonical_bytes({"source": source.spec.model_dump(mode="json")})
            + b"\n"
            + b"".join(canonical_bytes(asdict(row)) + b"\n" for row in source.rows)
        )
        paths.append(path)
    embedding_path = tmp_path / "embeddings.json"
    embedding_path.write_bytes(canonical_bytes(asdict(embeddings)))
    program_path = tmp_path / "programs.json"
    program_path.write_bytes(
        canonical_bytes([program.model_dump(mode="json") for program in programs])
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_bytes(config.to_bytes())
    index = build_command(
        tmp_path / "output",
        source_paths=paths,
        embedding_path=embedding_path,
        config_path=config_path,
        programs_path=program_path,
    )
    assert index["build_id"] == expected.build_id
    assert index["splits"] == {
        bundle.manifest.split: bundle.manifest.dataset.model_dump(mode="json")
        for bundle in expected.bundles
    }
