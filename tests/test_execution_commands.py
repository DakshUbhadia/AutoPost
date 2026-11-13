import json

from autopost.cli import main
from autopost.execution import spike
from autopost.execution.capabilities import GPU_PINS


def test_subprocess_cli_roundtrip_is_idempotent(tmp_path, capsys):
    command = ["execution", "fixture", "--root", str(tmp_path)]
    assert main(command) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "succeeded"
    assert first["acceptance"] == "accepted"
    assert first["remaining_budget"] == 9
    assert main(command) == 0
    repeated = json.loads(capsys.readouterr().out)
    assert repeated["acceptance"] == "duplicate"
    assert repeated["remaining_budget"] == 9


def test_gpu_preflight_records_blockers_without_downloading_models(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        spike,
        "runtime_snapshot",
        lambda: {
            "platform": "Windows",
            "versions": {name: None for name in GPU_PINS},
            "cuda_available": False,
            "devices": [],
        },
    )
    report = spike.run_spike(tmp_path, run_gpu=True)
    assert report["status"] == "BLOCKED"
    assert report["checks"] == {}
    assert report["scientific_execution"] is False
    saved = json.loads((tmp_path / "latest-report.json").read_bytes())
    assert saved == report


def test_gpu_available_does_not_run_without_explicit_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(
        spike,
        "runtime_snapshot",
        lambda: {
            "platform": "Linux",
            "versions": dict(GPU_PINS),
            "cuda_available": True,
            "devices": [
                {
                    "name": "synthetic",
                    "capability": [8, 0],
                    "memory_bytes": 24 * 1024**3,
                }
            ],
        },
    )
    report = spike.run_spike(tmp_path)
    assert report["status"] == "READY_NOT_RUN"
    assert report["checks"] == {}


def test_phase_failure_retains_report_and_never_claims_success(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        spike,
        "runtime_snapshot",
        lambda: {
            "platform": "Linux",
            "versions": dict(GPU_PINS),
            "cuda_available": True,
            "devices": [
                {
                    "name": "synthetic",
                    "capability": [8, 0],
                    "memory_bytes": 24 * 1024**3,
                }
            ],
        },
    )

    class FailedProcess:
        pid = 123

        def wait(self, **_):
            return 1

    original_run = spike.subprocess.run
    monkeypatch.setattr(
        spike.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(stdout=b"fixture==1.0\n"),
    )
    monkeypatch.setattr(spike.subprocess, "Popen", lambda *a, **kw: FailedProcess())
    monkeypatch.setattr(spike.os, "killpg", lambda *_: None, raising=False)
    monkeypatch.setattr(spike.signal, "SIGKILL", 9, raising=False)
    report = spike.run_spike(tmp_path, run_gpu=True)
    assert report["status"] == "FAIL"
    assert report["checks"] == {"adapter": {"passed": False}}
    monkeypatch.setattr(spike.subprocess, "run", original_run)
