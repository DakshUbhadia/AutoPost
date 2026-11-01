import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/selfcheck.py"
SPEC = importlib.util.spec_from_file_location("autopost_selfcheck", PATH)
SC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SC)


def test_fast_selfcheck():
    SC.scorer_tests(n_random=500)
    SC.independence_test()
    SC.reward_and_band()
    SC.gate_tests()
    SC.lora_scale_test()
    SC.c1_demo()


@pytest.mark.slow
def test_full_selfcheck():
    SC.main([])
