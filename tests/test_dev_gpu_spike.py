"""Explicit hardware acceptance gate, excluded from ordinary CPU execution."""

import os

import pytest

from autopost.execution.spike import run_spike


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("AUTOPOST_RUN_GPU_SPIKE") != "1",
    reason="GPU campaign requires explicit opt-in",
)
def test_real_dev_gpu_compatibility(tmp_path):
    report = run_spike(
        tmp_path, execution=os.environ.get("AUTOPOST_EXECUTION_PROFILE"), run_gpu=True
    )
    assert report["status"] == "PASS", report
