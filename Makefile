.DEFAULT_GOAL := help
.PHONY: help install install-cpu install-toolchain-probe selfcheck test test-all db-init prereg-verify toolchain-probe spike data hidden phase1 gonogo run report

help:
	@echo install-cpu install-toolchain-probe selfcheck test test-all db-init prereg-verify toolchain-probe spike data hidden phase1 gonogo run report

install: install-cpu

install-cpu:
	python -m pip install --requirement requirements/cpu-dev.lock
	python -m pip install --no-deps --no-build-isolation --editable .

install-toolchain-probe:
	python -m pip install --requirement requirements/toolchain-probe.lock
	python -m pip install --no-deps --no-build-isolation --editable .

selfcheck:
	python scripts/selfcheck.py

test:
	python -m pytest -q -m "not slow" --basetemp=.test-tmp

test-all:
	python -m pytest -q --basetemp=.test-tmp

db-init:
	python -m autopost.cli db init

prereg-verify:
	python -m autopost.cli prereg verify

toolchain-probe:
	python scripts/probe_toolchain_api.py

spike data hidden phase1 gonogo run report:
	@echo not implemented yet (see roadmap)
	@exit 1
